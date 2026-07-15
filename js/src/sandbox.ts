import { ClientError, Status } from "nice-grpc";
import { setTimeout } from "timers/promises";
import {
  FileDescriptor,
  GenericResult,
  GenericResult_GenericStatus,
  PTYInfo,
  PTYInfo_PTYType,
  SandboxTagsGetResponse,
  SandboxCreateRequest,
  SandboxCreateV2Request,
  NetworkAccess,
  NetworkAccess_NetworkAccessType,
  VolumeMount,
  CloudBucketMount as CloudBucketMountProto,
  SchedulerPlacement,
  TunnelType,
  PortSpec,
  Resources,
  PortSpecs,
  Probe as ProbeProto,
  StringMap,
} from "../proto/modal_proto/api";
import {
  TaskExecStartRequest,
  TaskExecStdoutConfig,
  TaskExecStderrConfig,
  TaskMountDirectoryRequest,
  TaskReloadVolumesRequest,
  TaskSnapshotDirectoryRequest,
  TaskSnapshotFilesystemRequest,
  TaskUnmountDirectoryRequest,
  TaskSetNetworkAccessRequest,
} from "../proto/modal_proto/task_command_router";
import { TaskCommandRouterClientImpl } from "./task_command_router_client";
import { v4 as uuidv4 } from "uuid";
import { type ModalClient, isRetryableGrpc, ModalGrpcClient } from "./client";
import { SandboxFilesystem } from "./sandbox_fs";
import { SidecarService } from "./sandbox_sidecar";
import {
  type ModalReadStream,
  type ModalWriteStream,
  streamConsumingIter,
  toModalReadStream,
  toModalWriteStream,
} from "./streams";
import {
  type Secret,
  mergeEnvIntoSecrets,
  hydrateSecrets,
  splitEnvDictAndResolvableSecrets,
  validateEnvVarKeys,
} from "./secret";
import {
  ConflictError,
  InvalidError,
  NotFoundError,
  SandboxTimeoutError,
  AlreadyExistsError,
  ClientClosedError,
} from "./errors";
import { Image } from "./image";
import { checkObjectName } from "./name_utils";
import { volumeToMountProto, type Volume } from "./volume";
import type { Proxy } from "./proxy";
import type { CloudBucketMount } from "./cloud_bucket_mount";
import type { App } from "./app";
import { parseGpuConfig } from "./app";
import { checkForRenamedParams } from "./validation";

// Backoff configuration for SandboxGetLogs retry behavior.
const SB_LOGS_INITIAL_DELAY_MS = 10;
const SB_LOGS_DELAY_FACTOR = 2;
const SB_LOGS_MAX_RETRIES = 10;

const TTL_NO_EXPIRY_SENTINEL = -1;
const CUSTOMER_SUPPLIED_ENCRYPTION_KEY_MIN_LENGTH = 16;
const CUSTOMER_SUPPLIED_ENCRYPTION_KEY_MAX_LENGTH = 512;

/**
 * Resolve the caller-facing `ttlMs` field on snapshot params into the
 * wire-format integer seconds expected by the snapshot RPCs.
 *
 * `undefined` (or omitted) → default 30 days.
 * `null` → opt out of expiry (wire sentinel `-1`).
 * Positive multiple of 1000ms → converted to whole seconds. Sub-second
 * or non-whole-second values are rejected since the wire format would
 * silently strip the sub-second precision.
 */
function resolveTtlSeconds(ttlMs: number | null | undefined): number {
  if (ttlMs === undefined) {
    return 30 * 24 * 3600;
  }
  if (ttlMs === null) {
    return TTL_NO_EXPIRY_SENTINEL;
  }
  if (ttlMs < 1000) {
    throw new InvalidError(`ttlMs must be at least 1000ms, got ${ttlMs}`);
  }
  if (ttlMs % 1000 !== 0) {
    throw new InvalidError(`ttlMs must be a multiple of 1000ms, got ${ttlMs}`);
  }
  return ttlMs / 1000;
}

/**
 * @internal
 * @hidden
 */
export function validateExperimentalEncryptionKey(
  key: Uint8Array | undefined,
): Uint8Array | undefined {
  if (key === undefined) {
    return undefined;
  }
  if (!(key instanceof Uint8Array)) {
    throw new TypeError("experimentalEncryptionKey must be a Uint8Array");
  }
  if (
    key.length < CUSTOMER_SUPPLIED_ENCRYPTION_KEY_MIN_LENGTH ||
    key.length > CUSTOMER_SUPPLIED_ENCRYPTION_KEY_MAX_LENGTH
  ) {
    throw new InvalidError(
      `experimentalEncryptionKey must be between ${CUSTOMER_SUPPLIED_ENCRYPTION_KEY_MIN_LENGTH} and ` +
        `${CUSTOMER_SUPPLIED_ENCRYPTION_KEY_MAX_LENGTH} bytes, got ${key.length} bytes`,
    );
  }
  return key;
}

const V1_SANDBOX_ID_ALPHABET = new Set(
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
);
const ULID_ALPHABET = new Set("0123456789ABCDEFGHJKMNPQRSTVWXYZ");

/** @ignore */
export enum SandboxVersion {
  V1 = "V1",
  V2 = "V2",
}

function isV1SandboxId(sandboxId: string): boolean {
  const [prefix, suffix, ...extra] = sandboxId.split("-");
  return (
    prefix === "sb" &&
    extra.length === 0 &&
    suffix !== undefined &&
    suffix.length === 22 &&
    Array.from(suffix).every((ch) => V1_SANDBOX_ID_ALPHABET.has(ch))
  );
}

function isV2SandboxId(sandboxId: string): boolean {
  const [prefix, suffix, ...extra] = sandboxId.split("-");
  return (
    prefix === "sb" &&
    extra.length === 0 &&
    suffix !== undefined &&
    suffix.length === 26 &&
    "01234567".includes(suffix[0]) &&
    Array.from(suffix).every((ch) => ULID_ALPHABET.has(ch))
  );
}

/** @ignore */
export function getSandboxVersion(sandboxId: string): SandboxVersion {
  if (isV2SandboxId(sandboxId)) {
    return SandboxVersion.V2;
  }
  if (isV1SandboxId(sandboxId)) {
    return SandboxVersion.V1;
  }
  throw new InvalidError(`Invalid Sandbox ID: ${sandboxId}`);
}

/**
 * Stdin is always present, but this option allow you to drop stdout or stderr
 * if you don't need them. The default is "pipe", matching Node.js behavior.
 *
 * If behavior is set to "ignore", the output streams will be empty.
 */
export type StdioBehavior = "pipe" | "ignore";

/**
 * Specifies the type of data that will be read from the Sandbox or container
 * process. "text" means the data will be read as UTF-8 text, while "binary"
 * means the data will be read as raw bytes (Uint8Array).
 */
export type StreamMode = "text" | "binary";

/** Optional parameters for {@link Probe.withTcp} and {@link Probe.withExec}. */
export type ProbeParams = {
  intervalMs: number;
};

/** Probe configuration for sandbox readiness checks. */
export class Probe {
  readonly #tcpPort?: number;
  readonly #execArgv?: string[];
  readonly #intervalMs: number;

  private constructor(params: {
    tcpPort?: number;
    execArgv?: string[];
    intervalMs: number;
  }) {
    const { tcpPort, execArgv, intervalMs } = params;
    if ((tcpPort === undefined) === (execArgv === undefined)) {
      throw new InvalidError(
        "Probe must be created with Probe.withTcp(...) or Probe.withExec(...)",
      );
    }
    this.#tcpPort = tcpPort;
    this.#execArgv = execArgv;
    this.#intervalMs = intervalMs;
  }

  static withTcp(
    port: number,
    params: ProbeParams = { intervalMs: 100 },
  ): Probe {
    if (!Number.isInteger(port)) {
      throw new InvalidError("Probe.withTcp() expects an integer `port`");
    }
    if (port <= 0 || port > 65535) {
      throw new InvalidError(
        `Probe.withTcp() expects \`port\` in [1, 65535], got ${port}`,
      );
    }
    const { intervalMs } = params;
    Probe.#validateIntervalMs("Probe.withTcp", intervalMs);
    return new Probe({ tcpPort: port, intervalMs });
  }

  static withExec(
    argv: string[],
    params: ProbeParams = { intervalMs: 100 },
  ): Probe {
    if (!Array.isArray(argv) || argv.length === 0) {
      throw new InvalidError("Probe.withExec() requires at least one argument");
    }
    if (!argv.every((arg) => typeof arg === "string")) {
      throw new InvalidError(
        "Probe.withExec() expects all arguments to be strings",
      );
    }
    const { intervalMs } = params;
    Probe.#validateIntervalMs("Probe.withExec", intervalMs);
    return new Probe({ execArgv: [...argv], intervalMs });
  }

  /** @ignore */
  toProto(): ProbeProto {
    if (this.#tcpPort !== undefined) {
      return ProbeProto.create({
        tcpPort: this.#tcpPort,
        intervalMs: this.#intervalMs,
      });
    }
    if (this.#execArgv !== undefined) {
      return ProbeProto.create({
        execCommand: { argv: this.#execArgv },
        intervalMs: this.#intervalMs,
      });
    }
    throw new InvalidError(
      "Probe must be created with Probe.withTcp(...) or Probe.withExec(...)",
    );
  }

  static #validateIntervalMs(methodName: string, intervalMs: number) {
    if (!Number.isInteger(intervalMs)) {
      throw new InvalidError(
        `${methodName}() expects an integer \`intervalMs\``,
      );
    }
    if (intervalMs <= 0) {
      throw new InvalidError(
        `${methodName}() expects \`intervalMs\` > 0, got ${intervalMs}`,
      );
    }
  }
}

/** Optional parameters for {@link SandboxService#create client.sandboxes.create()}. */
export type SandboxCreateParams = {
  /** Reservation of physical CPU cores for the Sandbox, can be fractional. */
  cpu?: number;

  /** Hard limit of physical CPU cores for the Sandbox, can be fractional. */
  cpuLimit?: number;

  /** Reservation of memory in MiB. */
  memoryMiB?: number;

  /** Hard limit of memory in MiB. */
  memoryLimitMiB?: number;

  /** GPU reservation for the Sandbox (e.g. "A100", "T4:2", "A100-80GB:4"). */
  gpu?: string;

  /** Maximum lifetime of the Sandbox in milliseconds. Defaults to 5 minutes. */
  timeoutMs?: number;

  /** The amount of time in milliseconds that a Sandbox can be idle before being terminated. */
  idleTimeoutMs?: number;

  /** Working directory of the Sandbox. */
  workdir?: string;

  /**
   * Sequence of program arguments for the main process.
   * Default behavior is to sleep indefinitely until timeout or termination.
   */
  command?: string[]; // default is ["sleep", "48h"]

  /** Environment variables to set in the Sandbox. */
  env?: Record<string, string>;

  /** {@link Secret}s to inject into the Sandbox as environment variables. */
  secrets?: Secret[];

  /** Mount points for Modal {@link Volume}s. */
  volumes?: Record<string, Volume>;

  /** Mount points for {@link CloudBucketMount}s. */
  cloudBucketMounts?: Record<string, CloudBucketMount>;

  /** Enable a PTY for the Sandbox entrypoint command. When enabled, all output (stdout and
   * stderr from the process) is multiplexed into stdout, and the stderr stream is effectively empty. */
  pty?: boolean;

  /** List of ports to tunnel into the Sandbox. Encrypted ports are tunneled with TLS. */
  encryptedPorts?: number[];

  /** List of encrypted ports to tunnel into the Sandbox, using HTTP/2. */
  h2Ports?: number[];

  /** List of ports to tunnel into the Sandbox without encryption. */
  unencryptedPorts?: number[];

  /** Whether to block all network access from the Sandbox. */
  blockNetwork?: boolean;

  /** List of CIDRs the Sandbox is allowed to access. If not set, all CIDRs are allowed. Cannot be used with blockNetwork. */
  outboundCidrAllowlist?: string[];

  /** List of domain names the Sandbox is allowed to access. Supports wildcard prefixes (`*.example.com`). Cannot be used with blockNetwork. */
  outboundDomainAllowlist?: string[];

  /** List of CIDRs allowed to connect inbound to the Sandbox (tunnels and connection tokens). If not set, all IPs are allowed. Cannot be used with blockNetwork. */
  inboundCidrAllowlist?: string[];

  /** Enable private IPv6 networking (i6pn) so Sandboxes in the same workspace can address each
   * other directly at their `i6pn.modal.local` address. Pin every Sandbox in the group to the same specific region
   * (e.g. `regions: ["us-east-1"]`). Cannot be used with blockNetwork. */
  i6pn?: boolean;

  /** Cloud provider to run the Sandbox on. */
  cloud?: string;

  /** Region(s) to run the Sandbox on. */
  regions?: string[];

  /** Enable verbose logging. */
  verbose?: boolean;

  /** Reference to a Modal {@link Proxy} to use in front of this Sandbox. */
  proxy?: Proxy;

  /** Probe used to determine when the Sandbox has become ready. */
  readinessProbe?: Probe;

  /** Optional name for the Sandbox. Unique within an App. */
  name?: string;

  /** Tags to attach to the Sandbox. Filterable via {@link SandboxService#list client.sandboxes.list}. */
  tags?: Record<string, string>;

  /** Optional experimental options. */
  experimentalOptions?: Record<string, any>;

  /** If set, connections to this Sandbox will be subdomains of this domain rather than the default.
   * This requires prior manual setup by Modal and is only available for Enterprise customers.
   */
  customDomain?: string;

  /** If true, the sandbox will receive a MODAL_IDENTITY_TOKEN env var for OIDC-based auth (e.g. to AWS, GCP). */
  includeOidcIdentityToken?: boolean;
};

export async function buildSandboxCreateRequestProto(
  appId: string,
  imageId: string,
  params: SandboxCreateParams = {},
): Promise<SandboxCreateRequest> {
  checkForRenamedParams(params, {
    memory: "memoryMiB",
    memoryLimit: "memoryLimitMiB",
    timeout: "timeoutMs",
    idleTimeout: "idleTimeoutMs",
    cidrAllowlist: "outboundCidrAllowlist",
  });

  const gpuConfig = parseGpuConfig(params.gpu);

  // The gRPC API only accepts a whole number of seconds.
  if (params.timeoutMs != undefined && params.timeoutMs <= 0) {
    throw new Error(`timeoutMs must be positive, got ${params.timeoutMs}`);
  }
  if (params.timeoutMs && params.timeoutMs % 1000 !== 0) {
    throw new Error(
      `timeoutMs must be a multiple of 1000ms, got ${params.timeoutMs}`,
    );
  }
  if (params.idleTimeoutMs != undefined && params.idleTimeoutMs <= 0) {
    throw new Error(
      `idleTimeoutMs must be positive, got ${params.idleTimeoutMs}`,
    );
  }
  if (params.idleTimeoutMs && params.idleTimeoutMs % 1000 !== 0) {
    throw new Error(
      `idleTimeoutMs must be a multiple of 1000ms, got ${params.idleTimeoutMs}`,
    );
  }

  if (params.workdir && !params.workdir.startsWith("/")) {
    throw new Error(`workdir must be an absolute path, got: ${params.workdir}`);
  }

  const volumeMounts: VolumeMount[] = params.volumes
    ? Object.entries(params.volumes).map(([mountPath, volume]) =>
        volumeToMountProto(mountPath, volume),
      )
    : [];

  const cloudBucketMounts: CloudBucketMountProto[] = params.cloudBucketMounts
    ? Object.entries(params.cloudBucketMounts).map(([mountPath, mount]) =>
        mount.toProto(mountPath),
      )
    : [];

  const openPorts: PortSpec[] = [];
  if (params.encryptedPorts) {
    openPorts.push(
      ...params.encryptedPorts.map((port) =>
        PortSpec.create({
          port,
          unencrypted: false,
        }),
      ),
    );
  }
  if (params.h2Ports) {
    openPorts.push(
      ...params.h2Ports.map((port) =>
        PortSpec.create({
          port,
          unencrypted: false,
          tunnelType: TunnelType.TUNNEL_TYPE_H2,
        }),
      ),
    );
  }
  if (params.unencryptedPorts) {
    openPorts.push(
      ...params.unencryptedPorts.map((port) =>
        PortSpec.create({
          port,
          unencrypted: true,
        }),
      ),
    );
  }

  const secretIds = (params.secrets || []).map((secret) => secret.secretId);

  let networkAccess: NetworkAccess;
  if (params.blockNetwork) {
    if (params.i6pn) {
      throw new InvalidError(
        "blockNetwork disables all networking, including i6pn. To keep i6pn while blocking " +
          "public egress, use an empty outbound allowlist (outboundCidrAllowlist: []) instead.",
      );
    }
    if (params.outboundCidrAllowlist) {
      throw new Error(
        "outboundCidrAllowlist cannot be used when blockNetwork is enabled",
      );
    }
    if (params.outboundDomainAllowlist) {
      throw new Error(
        "outboundDomainAllowlist cannot be used when blockNetwork is enabled",
      );
    }
    if (params.inboundCidrAllowlist) {
      throw new Error(
        "inboundCidrAllowlist cannot be used when blockNetwork is enabled",
      );
    }
    networkAccess = NetworkAccess.create({
      networkAccessType: NetworkAccess_NetworkAccessType.BLOCKED,
      allowedCidrs: [],
      allowedDomains: [],
    });
  } else if (params.outboundCidrAllowlist || params.outboundDomainAllowlist) {
    networkAccess = NetworkAccess.create({
      networkAccessType: NetworkAccess_NetworkAccessType.ALLOWLIST,
      allowedCidrs: params.outboundCidrAllowlist || [],
      allowedDomains: params.outboundDomainAllowlist || [],
    });
  } else {
    networkAccess = NetworkAccess.create({
      networkAccessType: NetworkAccess_NetworkAccessType.OPEN,
      allowedCidrs: [],
      allowedDomains: [],
    });
  }

  const schedulerPlacement: SchedulerPlacement | undefined = params.regions
    ?.length
    ? SchedulerPlacement.create({
        regions: params.regions,
      })
    : undefined;

  let ptyInfo: PTYInfo | undefined;
  if (params.pty) {
    ptyInfo = defaultSandboxPTYInfo();
  }

  let milliCpu: number | undefined = undefined;
  let milliCpuMax: number | undefined = undefined;
  if (params.cpu === undefined && params.cpuLimit !== undefined) {
    throw new Error("must also specify cpu when cpuLimit is specified");
  }
  if (params.cpu !== undefined) {
    if (params.cpu <= 0) {
      throw new Error(`cpu (${params.cpu}) must be a positive number`);
    }
    milliCpu = Math.trunc(1000 * params.cpu);
    if (params.cpuLimit !== undefined) {
      if (params.cpuLimit < params.cpu) {
        throw new Error(
          `cpu (${params.cpu}) cannot be higher than cpuLimit (${params.cpuLimit})`,
        );
      }
      milliCpuMax = Math.trunc(1000 * params.cpuLimit);
    }
  }

  let memoryMb: number | undefined = undefined;
  let memoryMbMax: number | undefined = undefined;
  if (params.memoryMiB === undefined && params.memoryLimitMiB !== undefined) {
    throw new Error(
      "must also specify memoryMiB when memoryLimitMiB is specified",
    );
  }
  if (params.memoryMiB !== undefined) {
    if (params.memoryMiB <= 0) {
      throw new Error(
        `the memoryMiB request (${params.memoryMiB}) must be a positive number`,
      );
    }
    memoryMb = params.memoryMiB;
    if (params.memoryLimitMiB !== undefined) {
      if (params.memoryLimitMiB < params.memoryMiB) {
        throw new Error(
          `the memoryMiB request (${params.memoryMiB}) cannot be higher than memoryLimitMiB (${params.memoryLimitMiB})`,
        );
      }
      memoryMbMax = params.memoryLimitMiB;
    }
  }

  // The public interface uses Record<string, any> so that we can add support for any experimental
  // option type in the future. The experimental_options_v2 proto map accepts arbitrary strings, but
  // for now we deliberately restrict values to booleans and strings.
  const protoExperimentalOptions: Record<string, string> =
    params.experimentalOptions
      ? Object.entries(params.experimentalOptions).reduce(
          (acc, [name, value]) => {
            if (typeof value !== "boolean" && typeof value !== "string") {
              throw new Error(
                `experimental option '${name}' must be a boolean or string, got ${value}`,
              );
            }
            acc[name] = String(value);
            return acc;
          },
          {} as Record<string, string>,
        )
      : {};

  const tagsList = params.tags
    ? Object.entries(params.tags).map(([tagName, tagValue]) => ({
        tagName,
        tagValue,
      }))
    : [];

  return SandboxCreateRequest.create({
    appId,
    tags: tagsList,
    definition: {
      entrypointArgs: params.command ?? [],
      imageId,
      timeoutSecs:
        params.timeoutMs != undefined ? params.timeoutMs / 1000 : 300,
      idleTimeoutSecs:
        params.idleTimeoutMs != undefined
          ? params.idleTimeoutMs / 1000
          : undefined,
      workdir: params.workdir ?? undefined,
      networkAccess,
      resources: Resources.create({
        milliCpu,
        milliCpuMax,
        memoryMb,
        memoryMbMax,
        gpuConfig,
      }),
      volumeMounts,
      cloudBucketMounts,
      ptyInfo,
      secretIds,
      openPorts: PortSpecs.create({ ports: openPorts }),
      cloudProviderStr: params.cloud ?? "",
      schedulerPlacement,
      verbose: params.verbose ?? false,
      proxyId: params.proxy?.proxyId,
      readinessProbe: params.readinessProbe?.toProto(),
      name: params.name,
      experimentalOptionsV2: protoExperimentalOptions,
      customDomain: params.customDomain,
      includeOidcIdentityToken: params.includeOidcIdentityToken ?? false,
      inboundCidrAllowlist: params.inboundCidrAllowlist ?? [],
      i6pnEnabled: params.i6pn ?? false,
    },
  });
}

export async function buildSandboxCreateV2RequestProto(
  appId: string,
  imageId: string,
  params: SandboxCreateParams = {},
): Promise<SandboxCreateV2Request> {
  if (params.gpu) {
    throw new Error("GPUs are not supported by experimentalCreate");
  }

  const req = await buildSandboxCreateRequestProto(appId, imageId, params);

  // V2 sandboxes support ephemeral env vars natively, so env vars are passed
  // directly rather than via a server-side Secret.
  const ephemeralSecrets =
    params.env && Object.keys(params.env).length > 0
      ? StringMap.create({ contents: params.env })
      : undefined;

  return SandboxCreateV2Request.create({
    appId: req.appId,
    definition: req.definition,
    ephemeralSecrets,
    tags: req.tags,
  });
}

/**
 * Service for managing {@link Sandbox}es.
 *
 * Normally only ever accessed via the client as:
 * ```typescript
 * const modal = new ModalClient();
 * const sb = await modal.sandboxes.create(app, image);
 * ```
 */
export class SandboxService {
  readonly #client: ModalClient;
  constructor(client: ModalClient) {
    this.#client = client;
  }

  /**
   * Create a new {@link Sandbox} in the {@link App} with the specified {@link Image} and options.
   */
  async create(
    app: App,
    image: Image,
    params: SandboxCreateParams = {},
  ): Promise<Sandbox> {
    await image.build(app);

    const mergedSecrets = await mergeEnvIntoSecrets(
      this.#client,
      params.env,
      params.secrets,
    );

    // The SandboxCreate request only carries secret IDs, so any locally-created
    // Secrets (and env vars) must be hydrated into server-side Secrets first.
    await hydrateSandboxSecrets(
      this.#client,
      mergedSecrets,
      params.cloudBucketMounts,
    );

    const mergedParams = {
      ...params,
      secrets: mergedSecrets,
      env: undefined, // setting env to undefined just to clarify it's not needed anymore
    };

    const createReq = await buildSandboxCreateRequestProto(
      app.appId,
      image.imageId,
      mergedParams,
    );
    let createResp;
    try {
      createResp = await this.#client.cpClient.sandboxCreate(createReq);
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.ALREADY_EXISTS) {
        throw new AlreadyExistsError(err.details || err.message);
      }
      throw err;
    }

    this.#client.logger.debug(
      "Created Sandbox",
      "sandbox_id",
      createResp.sandboxId,
    );
    return new Sandbox(this.#client, createResp.sandboxId);
  }

  /**
   * Create a new {@link Sandbox} using the experimental V2 backend.
   *
   * Supported features include exec, encrypted tunnels, wait/poll/terminate,
   * CPU and memory configuration, region placement, volumes, cloud bucket
   * mounts (with static credentials via {@link Secret} or `oidcAuthRoleArn`),
   * OIDC identity tokens, {@link Proxy proxies}, filesystem snapshots, and
   * custom domains (`customDomain` allows connections to the Sandbox via a
   * subdomain of that parent domain rather than a default Modal domain;
   * requires prior setup by Modal).
   *
   * Features like memory snapshots and GPUs are not supported.
   *
   * V2 sandboxes created with this method are not currently returned by
   * {@link SandboxService#list client.sandboxes.list()}. A named Sandbox can be
   * looked up with
   * {@link SandboxService#experimentalFromName client.sandboxes.experimentalFromName()};
   * otherwise store {@link Sandbox#sandboxId sandbox.sandboxId} and use
   * {@link SandboxService#fromId client.sandboxes.fromId()} to reattach.
   */
  async experimentalCreate(
    app: App,
    image: Image,
    params: SandboxCreateParams = {},
  ): Promise<Sandbox> {
    await image.build(app);

    // V2 supports ephemeral env vars natively (passed via ephemeralSecrets in
    // the request), so unlike create() we don't fold env vars into a server-side
    // Secret. Locally-created Secrets (fromObject) and params.env are sent
    // directly as ephemeral env vars, avoiding a SecretGetOrCreate round-trip;
    // params.env takes precedence on key collisions. Only the remaining
    // resolvable Secrets (e.g. from fromName) need hydrating to secret IDs.
    validateEnvVarKeys(params.env ?? {});
    const [envDict, resolvableSecrets] = splitEnvDictAndResolvableSecrets(
      params.secrets ?? [],
    );
    Object.assign(envDict, params.env ?? {});

    await hydrateSandboxSecrets(
      this.#client,
      resolvableSecrets,
      params.cloudBucketMounts,
    );

    const mergedParams = {
      ...params,
      secrets: resolvableSecrets,
      env: envDict,
    };

    const createReq = await buildSandboxCreateV2RequestProto(
      app.appId,
      image.imageId,
      mergedParams,
    );
    let createResp;
    try {
      createResp = await this.#client.cpClient.sandboxCreateV2(createReq);
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.ALREADY_EXISTS) {
        throw new AlreadyExistsError(err.details || err.message);
      }
      throw err;
    }

    this.#client.logger.debug(
      "Created experimental V2 Sandbox",
      "sandbox_id",
      createResp.sandboxId,
    );

    const tunnels =
      createResp.tunnels.length > 0
        ? Object.fromEntries(
            createResp.tunnels.map((t) => [
              t.containerPort,
              new Tunnel(t.host, t.port, t.unencryptedHost, t.unencryptedPort),
            ]),
          )
        : undefined;

    return new Sandbox(this.#client, createResp.sandboxId, {
      isV2: true,
      taskId: createResp.taskId,
      tunnels,
    });
  }

  /** Returns a running {@link Sandbox} object from an ID.
   *
   * @returns Sandbox with ID
   */
  async fromId(sandboxId: string): Promise<Sandbox> {
    const sandboxVersion = getSandboxVersion(sandboxId);
    const isV2 = sandboxVersion === SandboxVersion.V2;
    return new Sandbox(this.#client, sandboxId, { isV2 });
  }

  /** Get a running {@link Sandbox} by name from a deployed {@link App}.
   *
   * Raises a {@link NotFoundError} if no running Sandbox is found with the given name.
   * A Sandbox's name is the `name` argument passed to {@link SandboxService#create sandboxes.create()}.
   *
   * @param appName - Name of the deployed App
   * @param name - Name of the Sandbox
   * @param params - Optional parameters for getting the Sandbox
   * @returns Promise that resolves to a Sandbox
   */
  async fromName(
    appName: string,
    name: string,
    params?: SandboxFromNameParams,
  ): Promise<Sandbox> {
    try {
      const resp = await this.#client.cpClient.sandboxGetFromName({
        sandboxName: name,
        appName,
        environmentName: this.#client.environmentName(params?.environment),
      });
      return new Sandbox(this.#client, resp.sandboxId);
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(
          `Sandbox with name '${name}' not found in App '${appName}'`,
        );
      throw err;
    }
  }

  /**
   * Get a running V2 {@link Sandbox} by name from a deployed {@link App}.
   *
   * This looks up V2 Sandboxes, i.e. Sandboxes created via
   * {@link SandboxService#experimentalCreate client.sandboxes.experimentalCreate()}.
   *
   * EXPERIMENTAL: the API is subject to change.
   *
   * @param appName - Name of the deployed App
   * @param name - Name of the Sandbox
   * @param params - Optional parameters for getting the Sandbox
   * @returns Promise that resolves to a Sandbox
   */
  async experimentalFromName(
    appName: string,
    name: string,
    params?: SandboxExperimentalFromNameParams,
  ): Promise<Sandbox> {
    try {
      // SandboxGetFromNameV2 only returns V2 Sandboxes and authenticates via
      // the auth-token metadata (attached automatically by the client), like
      // the other V2 Sandbox RPCs.
      const resp = await this.#client.cpClient.sandboxGetFromNameV2({
        sandboxName: name,
        appName,
        environmentName: this.#client.environmentName(params?.environment),
      });
      return new Sandbox(this.#client, resp.sandboxId, { isV2: true });
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(
          `Sandbox with name '${name}' not found in App '${appName}'`,
        );
      throw err;
    }
  }

  /**
   * List all {@link Sandbox}es for the current Environment or App ID (if specified).
   * If tags are specified, only Sandboxes that have at least those tags are returned.
   */
  async *list(
    params: SandboxListParams = {},
  ): AsyncGenerator<Sandbox, void, unknown> {
    const env = this.#client.environmentName(params.environment);
    const tagsList = params.tags
      ? Object.entries(params.tags).map(([tagName, tagValue]) => ({
          tagName,
          tagValue,
        }))
      : [];

    let beforeTimestamp: number | undefined = undefined;
    while (true) {
      try {
        const resp = await this.#client.cpClient.sandboxList({
          appId: params.appId,
          beforeTimestamp,
          environmentName: env,
          includeFinished: false,
          tags: tagsList,
        });
        if (!resp.sandboxes || resp.sandboxes.length === 0) {
          return;
        }
        for (const info of resp.sandboxes) {
          yield new Sandbox(this.#client, info.id);
        }
        beforeTimestamp = resp.sandboxes[resp.sandboxes.length - 1].createdAt;
      } catch (err) {
        if (
          err instanceof ClientError &&
          err.code === Status.INVALID_ARGUMENT
        ) {
          throw new InvalidError(err.details || err.message);
        }
        throw err;
      }
    }
  }

  /**
   * List the V2 {@link Sandbox}es in an {@link App}.
   *
   * This lists V2 Sandboxes, i.e. Sandboxes created via
   * {@link SandboxService#experimentalCreate client.sandboxes.experimentalCreate()}.
   * Such Sandboxes are not returned by
   * {@link SandboxService#list client.sandboxes.list()}. If tags are specified,
   * only Sandboxes that have at least those tags are returned.
   *
   * Yields {@link Sandbox} objects that are currently running in the App.
   *
   * EXPERIMENTAL: the API is subject to change.
   */
  async *experimentalList(
    params: SandboxExperimentalListParams,
  ): AsyncGenerator<Sandbox, void, unknown> {
    if (!params?.appId) {
      throw new InvalidError(
        "experimentalList requires an `appId`:\n\n" +
          'const app = await modal.apps.fromName("my-app");\n' +
          "modal.sandboxes.experimentalList({ appId: app.appId });",
      );
    }

    const tagsList = params.tags
      ? Object.entries(params.tags).map(([tagName, tagValue]) => ({
          tagName,
          tagValue,
        }))
      : [];

    let beforeTimestamp: number | undefined = undefined;
    while (true) {
      // Fetches a batch of Sandboxes. SandboxListV2 authenticates via the
      // auth-token metadata (attached automatically by the client), like the
      // other V2 Sandbox RPCs.
      const resp = await this.#client.cpClient.sandboxListV2({
        appId: params.appId,
        beforeTimestamp,
        includeFinished: false,
        tags: tagsList,
      });
      if (!resp.sandboxes || resp.sandboxes.length === 0) {
        return;
      }
      for (const info of resp.sandboxes) {
        // SandboxListV2 only returns V2 Sandboxes; mark them as such so
        // operations like wait/terminate/exec use the V2 RPCs.
        yield new Sandbox(this.#client, info.id, { isV2: true });
      }
      // Fetch the next batch starting from the end of the current one.
      beforeTimestamp = resp.sandboxes[resp.sandboxes.length - 1].createdAt;
    }
  }
}

/** Optional parameters for {@link SandboxService#list client.sandboxes.list()}. */
export type SandboxListParams = {
  /** Filter Sandboxes for a specific {@link App}. */
  appId?: string;
  /** Only return Sandboxes that include all specified tags. */
  tags?: Record<string, string>;
  /** Override environment for the request; defaults to current profile. */
  environment?: string;
};

/** Parameters for {@link SandboxService#experimentalList client.sandboxes.experimentalList()}. */
export type SandboxExperimentalListParams = {
  /** The App to list Sandboxes under. */
  appId: string;
  /** Only return Sandboxes that include all specified tags. */
  tags?: Record<string, string>;
};

/** Optional parameters for {@link SandboxService#fromName client.sandboxes.fromName()}. */
export type SandboxFromNameParams = {
  environment?: string;
};

/** Optional parameters for {@link SandboxService#experimentalFromName client.sandboxes.experimentalFromName()}. */
export type SandboxExperimentalFromNameParams = {
  environment?: string;
};

/** Optional parameters for {@link Sandbox#exec Sandbox.exec()}. */
export type SandboxExecParams = {
  /** Specifies text or binary encoding for input and output streams. */
  mode?: StreamMode;
  /** Whether to pipe or ignore standard output. */
  stdout?: StdioBehavior;
  /** Whether to pipe or ignore standard error. */
  stderr?: StdioBehavior;
  /** Working directory to run the command in. */
  workdir?: string;
  /** Timeout for the process in milliseconds. Defaults to 0 (no timeout). */
  timeoutMs?: number;
  /** Environment variables to set for the command. */
  env?: Record<string, string>;
  /** {@link Secret}s to inject as environment variables for the commmand.*/
  secrets?: Secret[];
  /** Enable a PTY for the command. When enabled, all output (stdout and stderr from the
   * process) is multiplexed into stdout, and the stderr stream is effectively empty. */
  pty?: boolean;
};

/** Optional parameters for {@link Sandbox#terminate Sandbox.terminate()}. */
export type SandboxTerminateParams = {
  /** If true, wait for the Sandbox to finish and return the exit code. */
  wait?: boolean;
};

/** Optional parameters for {@link Sandbox#snapshotFilesystem Sandbox.snapshotFilesystem()}. */
export type SandboxSnapshotFilesystemParams = {
  /**
   * Overall budget for the snapshot call, in milliseconds. Defaults to
   * 55000. If it elapses before a snapshot completes, the call is cancelled
   * and an error is thrown.
   */
  timeoutMs?: number;
  /**
   * Lifetime of the resulting image in milliseconds, as a hard cutoff
   * measured from creation. Defaults to 30 days. Pass `null` to retain
   * the image indefinitely.
   */
  ttlMs?: number | null;
};

/** Optional parameters for {@link Sandbox#reloadVolumes Sandbox.reloadVolumes()}. */
export type SandboxReloadVolumesParams = {
  /**
   * Overall budget for the reload call, in milliseconds. Defaults to 55000.
   * If the reload does not complete within this window, the call is cancelled
   * and a `TimeoutError` is thrown; note that the reload may still complete in
   * the background.
   */
  timeoutMs?: number;
};

/** Optional parameters for {@link Sandbox#snapshotDirectory Sandbox.snapshotDirectory()}. */
export type SandboxSnapshotDirectoryParams = {
  /**
   * Overall budget for the snapshot call, in milliseconds. Defaults to
   * 55000. If it elapses before a snapshot completes, the call is cancelled
   * and an error is thrown.
   */
  timeoutMs?: number;
  /**
   * Lifetime of the resulting image in milliseconds, as a hard cutoff
   * measured from creation. Defaults to 30 days. Pass `null` to retain
   * the image indefinitely.
   */
  ttlMs?: number | null;
  /**
   * Experimental customer-supplied encryption key used to encrypt the
   * resulting snapshot. The same key is required when mounting the image.
   * Modal does not persist the key.
   */
  experimentalEncryptionKey?: Uint8Array;
};

/** Optional parameters for {@link Sandbox#mountImage Sandbox.mountImage()}. */
export type SandboxMountImageParams = {
  /**
   * Experimental customer-supplied encryption key used to decrypt the image.
   * Use the same key that encrypted the snapshot.
   */
  experimentalEncryptionKey?: Uint8Array;
};

/** Optional parameters for {@link Sandbox#createConnectToken Sandbox.createConnectToken()}. */
export type SandboxCreateConnectTokenParams = {
  /** Optional user-provided metadata string that will be added to the headers by the proxy when forwarding requests to the Sandbox. */
  userMetadata?: string;
  /** Container port that requests are routed to when using this token. Defaults to 8080. */
  port?: number;
};

/**
 * Parameters for {@link Sandbox#updateNetworkPolicy Sandbox.updateNetworkPolicy()}.
 *
 * Each dimension is independent: `undefined` leaves that dimension unchanged,
 * while a defined value replaces it (an empty array blocks all egress for that
 * dimension; a wildcard entry such as `"0.0.0.0/0"` or `"*"` allows everything).
 *
 * Currently, both dimensions must be provided (the underlying transport does
 * not yet support partial updates). This requirement will be relaxed in a
 * future release.
 */
export type SandboxUpdateNetworkPolicyParams = {
  outboundCidrAllowlist?: string[];
  outboundDomainAllowlist?: string[];
};

/** Credentials returned by {@link Sandbox#createConnectToken Sandbox.createConnectToken()}. */
export type SandboxCreateConnectCredentials = {
  url: string;
  token: string;
};

/** A port forwarded from within a running Modal {@link Sandbox}. */
export class Tunnel {
  /** @ignore */
  constructor(
    public host: string,
    public port: number,
    public unencryptedHost?: string,
    public unencryptedPort?: number,
  ) {}

  /** Get the public HTTPS URL of the forwarded port. */
  get url(): string {
    let value = `https://${this.host}`;
    if (this.port !== 443) {
      value += `:${this.port}`;
    }
    return value;
  }

  /** Get the public TLS socket as a [host, port] tuple. */
  get tlsSocket(): [string, number] {
    return [this.host, this.port];
  }

  /** Get the public TCP socket as a [host, port] tuple. */
  get tcpSocket(): [string, number] {
    if (!this.unencryptedHost || this.unencryptedPort === undefined) {
      throw new InvalidError(
        "This tunnel is not configured for unencrypted TCP.",
      );
    }
    return [this.unencryptedHost, this.unencryptedPort];
  }
}

export function defaultSandboxPTYInfo(): PTYInfo {
  return PTYInfo.create({
    enabled: true,
    winszRows: 24,
    winszCols: 80,
    envTerm: "xterm-256color",
    envColorterm: "truecolor",
    envTermProgram: "",
    ptyType: PTYInfo_PTYType.PTY_TYPE_SHELL,
    noTerminateOnIdleStdin: true,
  });
}

// The maximum number of bytes that can be passed to an exec on Linux.
// Though this is technically a 'server side' limit, it is unlikely to change.
// getconf ARG_MAX will show this value on a host.
//
// By probing in production, the limit is 131072 bytes (2**17).
// We need some bytes of overhead for the rest of the command line besides the args,
// e.g. 'runsc exec ...'. So we use 2**16 as the limit.
export function validateExecArgs(args: string[]): void {
  const ARG_MAX_BYTES = 2 ** 16;

  // Avoid "[Errno 7] Argument list too long" errors.
  const totalArgLen = args.reduce((sum, arg) => sum + arg.length, 0);
  if (totalArgLen > ARG_MAX_BYTES) {
    throw new InvalidError(
      `Total length of CMD arguments must be less than ${ARG_MAX_BYTES} bytes (ARG_MAX). ` +
        `Got ${totalArgLen} bytes.`,
    );
  }
}

export function validateWorkdir(workdir: string | undefined): void {
  if (workdir !== undefined && !workdir.startsWith("/")) {
    throw new InvalidError(`workdir must be an absolute path, got: ${workdir}`);
  }
}

/**
 * Hydrate the given Secrets together with the credential Secrets of any cloud
 * bucket mounts, so all their secretIds are available before building the
 * SandboxCreate request. Gathers them into a fresh list to avoid mutating the
 * caller's secrets array.
 *
 * @internal
 * @hidden
 */
export async function hydrateSandboxSecrets(
  client: ModalClient,
  secrets: Secret[],
  mounts?: Record<string, CloudBucketMount>,
): Promise<void> {
  const toHydrate = [...secrets];
  if (mounts) {
    for (const mount of Object.values(mounts)) {
      if (mount?.secret) {
        toHydrate.push(mount.secret);
      }
    }
  }
  await hydrateSecrets(client, toHydrate);
}

export function getReturnCode(
  result: GenericResult | undefined,
): number | null {
  if (
    result === undefined ||
    result.status === GenericResult_GenericStatus.GENERIC_STATUS_UNSPECIFIED
  ) {
    return null;
  }

  // Statuses are converted to exitcodes so we can conform to subprocess API.
  if (result.status === GenericResult_GenericStatus.GENERIC_STATUS_TIMEOUT) {
    return 124;
  } else if (
    result.status === GenericResult_GenericStatus.GENERIC_STATUS_TERMINATED
  ) {
    return 137;
  } else {
    return result.exitcode;
  }
}

export function buildTaskExecStartRequestProto(
  taskId: string,
  execId: string,
  command: string[],
  params?: SandboxExecParams,
  containerId?: string,
): TaskExecStartRequest {
  checkForRenamedParams(params, { timeout: "timeoutMs" });

  if (params?.timeoutMs != undefined && params.timeoutMs <= 0) {
    throw new Error(`timeoutMs must be positive, got ${params.timeoutMs}`);
  }
  if (params?.timeoutMs && params.timeoutMs % 1000 !== 0) {
    throw new Error(
      `timeoutMs must be a multiple of 1000ms, got ${params.timeoutMs}`,
    );
  }
  validateWorkdir(params?.workdir);

  const secretIds = (params?.secrets || []).map((secret) => secret.secretId);

  const stdout = params?.stdout ?? "pipe";
  const stderr = params?.stderr ?? "pipe";

  let stdoutConfig: TaskExecStdoutConfig;
  if (stdout === "pipe") {
    stdoutConfig = TaskExecStdoutConfig.TASK_EXEC_STDOUT_CONFIG_PIPE;
  } else if (stdout === "ignore") {
    stdoutConfig = TaskExecStdoutConfig.TASK_EXEC_STDOUT_CONFIG_DEVNULL;
  } else {
    throw new Error(`Unsupported stdout behavior: ${stdout}`);
  }

  let stderrConfig: TaskExecStderrConfig;
  if (stderr === "pipe") {
    stderrConfig = TaskExecStderrConfig.TASK_EXEC_STDERR_CONFIG_PIPE;
  } else if (stderr === "ignore") {
    stderrConfig = TaskExecStderrConfig.TASK_EXEC_STDERR_CONFIG_DEVNULL;
  } else {
    throw new Error(`Unsupported stderr behavior: ${stderr}`);
  }

  let ptyInfo: PTYInfo | undefined;
  if (params?.pty) {
    ptyInfo = defaultSandboxPTYInfo();
  }

  return TaskExecStartRequest.create({
    taskId,
    execId,
    commandArgs: command,
    stdoutConfig,
    stderrConfig,
    timeoutSecs: params?.timeoutMs ? params.timeoutMs / 1000 : undefined,
    workdir: params?.workdir,
    secretIds,
    env: params?.env ?? {},
    ptyInfo,
    runtimeDebug: false,
    containerId: containerId ?? "",
  });
}

/** @ignore */
export function buildTaskMountDirectoryRequestProto(
  taskId: string,
  path: string,
  imageId: string,
  params?: SandboxMountImageParams,
): TaskMountDirectoryRequest {
  return TaskMountDirectoryRequest.create({
    taskId,
    path: new TextEncoder().encode(path),
    imageId,
    customerSuppliedEncryptionKey: validateExperimentalEncryptionKey(
      params?.experimentalEncryptionKey,
    ),
  });
}

/** @ignore */
export function buildTaskSnapshotDirectoryRequestProto(
  taskId: string,
  path: string,
  snapshotId: string,
  ttlSeconds: number,
  params?: SandboxSnapshotDirectoryParams,
): TaskSnapshotDirectoryRequest {
  return TaskSnapshotDirectoryRequest.create({
    taskId,
    path: new TextEncoder().encode(path),
    snapshotId,
    ttlSeconds,
    customerSuppliedEncryptionKey: validateExperimentalEncryptionKey(
      params?.experimentalEncryptionKey,
    ),
  });
}

/** Sandboxes are secure, isolated containers in Modal that boot in seconds. */
export class Sandbox {
  readonly #client: ModalClient;
  readonly sandboxId: string;

  #stdin?: ModalWriteStream<string>;
  #stdout?: ModalReadStream<string>;
  #stderr?: ModalReadStream<string>;
  #stdoutAbort?: AbortController;
  #stderrAbort?: AbortController;

  #taskId: string | undefined;
  #tunnels: Record<number, Tunnel> | undefined;
  #isV2: boolean;
  #commandRouterClient: TaskCommandRouterClientImpl | undefined;
  #commandRouterClientPromise: Promise<TaskCommandRouterClientImpl> | undefined;
  #attached: boolean = true;
  #filesystem?: SandboxFilesystem;
  #sidecars?: SidecarService;

  /** @ignore */
  constructor(
    client: ModalClient,
    sandboxId: string,
    params: {
      isV2?: boolean;
      taskId?: string;
      tunnels?: Record<number, Tunnel>;
    } = {},
  ) {
    this.#client = client;
    this.sandboxId = sandboxId;
    this.#isV2 = params.isV2 ?? false;
    this.#taskId = params.taskId || undefined;
    this.#tunnels = params.tunnels;
  }

  get stdin(): ModalWriteStream<string> {
    this.#ensureV1("stdin");
    if (!this.#stdin) {
      this.#stdin = toModalWriteStream(
        inputStreamSb(this.#client.cpClient, this.sandboxId),
      );
    }
    return this.#stdin;
  }

  get stdout(): ModalReadStream<string> {
    this.#ensureV1("stdout");
    if (!this.#stdout) {
      this.#stdoutAbort = new AbortController();
      const bytesStream = streamConsumingIter(
        outputStreamSb(
          this.#client.cpClient,
          this.sandboxId,
          FileDescriptor.FILE_DESCRIPTOR_STDOUT,
          this.#stdoutAbort.signal,
        ),
        () => this.#stdoutAbort?.abort(),
      );
      this.#stdout = toModalReadStream(
        bytesStream.pipeThrough(new TextDecoderStream()),
      );
    }
    return this.#stdout;
  }

  get stderr(): ModalReadStream<string> {
    this.#ensureV1("stderr");
    if (!this.#stderr) {
      this.#stderrAbort = new AbortController();
      const bytesStream = streamConsumingIter(
        outputStreamSb(
          this.#client.cpClient,
          this.sandboxId,
          FileDescriptor.FILE_DESCRIPTOR_STDERR,
          this.#stderrAbort.signal,
        ),
        () => this.#stderrAbort?.abort(),
      );
      this.#stderr = toModalReadStream(
        bytesStream.pipeThrough(new TextDecoderStream()),
      );
    }
    return this.#stderr;
  }

  get filesystem(): SandboxFilesystem {
    this.#ensureAttached();
    if (!this.#filesystem) {
      this.#filesystem = new SandboxFilesystem((command, params) =>
        this.#execInternal(command, params),
      );
    }
    return this.#filesystem;
  }

  /**
   * Operations for managing sidecar containers that run alongside the
   * Sandbox's main container.
   *
   * EXPERIMENTAL: the API is subject to change.
   */
  get experimentalSidecars(): SidecarService {
    this.#ensureAttached();
    if (!this.#sidecars) {
      this.#sidecars = new SidecarService({
        client: this.#client,
        exec: (command, params, containerId) =>
          this.#execInternal(command, params, containerId),
        commandRouter: () => this.#getCommandRouter(),
      });
    }
    return this.#sidecars;
  }

  /**
   * Set tags (key-value pairs) on the Sandbox. Tags can be used to filter results in {@link SandboxService#list client.sandboxes.list}.
   *
   * Setting tags replaces the Sandbox's entire tag set; passing an empty object clears all tags.
   */
  async setTags(tags: Record<string, string>): Promise<void> {
    this.#ensureAttached();
    const tagsList = Object.entries(tags).map(([tagName, tagValue]) => ({
      tagName,
      tagValue,
    }));
    try {
      if (this.#isV2) {
        await this.#client.cpClient.sandboxTagsSetV2({
          sandboxId: this.sandboxId,
          tags: tagsList,
        });
      } else {
        await this.#client.cpClient.sandboxTagsSet({
          environmentName: this.#client.environmentName(),
          sandboxId: this.sandboxId,
          tags: tagsList,
        });
      }
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.INVALID_ARGUMENT) {
        throw new InvalidError(err.details || err.message);
      }
      throw err;
    }
  }

  /** Get tags (key-value pairs) currently attached to this Sandbox from the server. */
  async getTags(): Promise<Record<string, string>> {
    this.#ensureAttached();
    let resp: SandboxTagsGetResponse;
    try {
      if (this.#isV2) {
        resp = await this.#client.cpClient.sandboxTagsGetV2({
          sandboxId: this.sandboxId,
        });
      } else {
        resp = await this.#client.cpClient.sandboxTagsGet({
          sandboxId: this.sandboxId,
        });
      }
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.INVALID_ARGUMENT) {
        throw new InvalidError(err.details || err.message);
      }
      throw err;
    }

    const tags: Record<string, string> = {};
    for (const tag of resp.tags) {
      tags[tag.tagName] = tag.tagValue;
    }
    return tags;
  }

  /**
   * Assign a name to a running V2 {@link Sandbox} that was created without one.
   *
   * This is only supported for V2 Sandboxes, i.e. Sandboxes created via
   * {@link SandboxService#experimentalCreate client.sandboxes.experimentalCreate()}.
   * A name may only be set once, and only on a Sandbox that has never had one;
   * afterwards the Sandbox can be looked up with
   * {@link SandboxService#experimentalFromName client.sandboxes.experimentalFromName()}.
   *
   * EXPERIMENTAL: the API is subject to change.
   *
   * @param name - Name to assign to the Sandbox. Must be unique within the App.
   * @throws {AlreadyExistsError} If another running Sandbox in the App already holds the name.
   * @throws {InvalidError} If the server rejects the name as invalid.
   * @throws {ConflictError} If the Sandbox already has a name or is no longer running.
   */
  async experimentalSetName(name: string): Promise<void> {
    this.#ensureAttached();
    this.#ensureV2("experimentalSetName");
    checkObjectName(name, "Sandbox");
    try {
      await this.#client.cpClient.sandboxSetName({
        sandboxId: this.sandboxId,
        name,
      });
    } catch (err) {
      if (err instanceof ClientError) {
        const message = err.details || err.message;
        switch (err.code) {
          case Status.ALREADY_EXISTS:
            throw new AlreadyExistsError(message);
          case Status.INVALID_ARGUMENT:
            throw new InvalidError(message);
          case Status.FAILED_PRECONDITION:
            throw new ConflictError(message);
        }
      }
      throw err;
    }
  }

  async exec(
    command: string[],
    params?: SandboxExecParams & { mode?: "text" },
  ): Promise<ContainerProcess<string>>;

  async exec(
    command: string[],
    params: SandboxExecParams & { mode: "binary" },
  ): Promise<ContainerProcess<Uint8Array>>;

  async exec(
    command: string[],
    params?: SandboxExecParams,
  ): Promise<ContainerProcess> {
    return this.#execInternal(command, params);
  }

  async #execInternal(
    command: string[],
    params: SandboxExecParams | undefined,
    containerId?: string,
  ): Promise<ContainerProcess> {
    this.#ensureAttached();
    validateExecArgs(command);

    // Locally-created Secrets (fromObject) are passed directly to the worker as
    // environment variables, so only the remaining Secrets need hydrating. This
    // avoids a SecretGetOrCreate round-trip for env-dict Secrets.
    validateEnvVarKeys(params?.env ?? {});
    const [envDict, resolvableSecrets] = splitEnvDictAndResolvableSecrets(
      params?.secrets ?? [],
    );
    Object.assign(envDict, params?.env ?? {});
    await hydrateSecrets(this.#client, resolvableSecrets);

    const execParams: SandboxExecParams = {
      ...params,
      env: envDict,
      secrets: resolvableSecrets,
    };

    const [taskId, commandRouterClient] = await this.#getCommandRouter();

    const execId = uuidv4();
    const request = buildTaskExecStartRequestProto(
      taskId,
      execId,
      command,
      execParams,
      containerId,
    );

    await commandRouterClient.execStart(request);

    this.#client.logger.debug(
      "Created ContainerProcess",
      "exec_id",
      execId,
      "sandbox_id",
      this.sandboxId,
      "container_id",
      containerId ?? "",
      "command",
      command,
    );

    const deadline = params?.timeoutMs ? Date.now() + params.timeoutMs : null;

    return new ContainerProcess(
      taskId,
      execId,
      commandRouterClient,
      params,
      deadline,
    );
  }

  async #getCommandRouter(): Promise<[string, TaskCommandRouterClientImpl]> {
    this.#ensureAttached();
    const taskId = await this.#getTaskId();
    const client = await this.#getOrCreateCommandRouterClient(taskId);
    return [taskId, client];
  }

  #ensureAttached(): void {
    if (!this.#attached) {
      throw new ClientClosedError();
    }
  }

  #ensureV1(methodName: string): void {
    if (this.#isV2) {
      throw new InvalidError(
        `Sandbox.${methodName} is not supported for V2 sandboxes`,
      );
    }
  }

  #ensureV2(methodName: string): void {
    if (!this.#isV2) {
      throw new InvalidError(
        `Sandbox.${methodName} is only supported for V2 sandboxes`,
      );
    }
  }

  #sandboxWait(timeout: number) {
    const req = { sandboxId: this.sandboxId, timeout };
    if (this.#isV2) {
      return this.#client.cpClient.sandboxWaitV2(req);
    }
    return this.#client.cpClient.sandboxWait(req);
  }

  #sandboxGetTaskId() {
    const req = { sandboxId: this.sandboxId };
    if (this.#isV2) {
      return this.#client.cpClient.sandboxGetTaskIdV2(req);
    }
    return this.#client.cpClient.sandboxGetTaskId(req);
  }

  #sandboxGetTunnels(timeoutMs: number) {
    const req = { sandboxId: this.sandboxId, timeout: timeoutMs / 1000 };
    if (this.#isV2) {
      return this.#client.cpClient.sandboxGetTunnelsV2(req);
    }
    return this.#client.cpClient.sandboxGetTunnels(req);
  }

  async #sandboxTerminate(): Promise<void> {
    const req = { sandboxId: this.sandboxId };
    if (this.#isV2) {
      await this.#client.cpClient.sandboxTerminateV2(req);
      return;
    }
    await this.#client.cpClient.sandboxTerminate(req);
  }

  static readonly #maxGetTaskIdAttempts = 600; // 5 minutes at 500ms intervals

  async #getTaskId(): Promise<string> {
    if (this.#taskId !== undefined) {
      return this.#taskId;
    }
    for (let i = 0; i < Sandbox.#maxGetTaskIdAttempts; i++) {
      const resp = await this.#sandboxGetTaskId();
      if (resp.taskResult) {
        if (
          resp.taskResult.status ===
            GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS ||
          !resp.taskResult.exception
        ) {
          throw new Error(`Sandbox ${this.sandboxId} has already completed`);
        }
        throw new Error(
          `Sandbox ${this.sandboxId} has already completed with result: exception:"${resp.taskResult.exception}"`,
        );
      }
      if (resp.taskId) {
        this.#taskId = resp.taskId;
        return this.#taskId;
      }
      await setTimeout(500);
    }
    throw new Error(
      `Timed out waiting for task ID for Sandbox ${this.sandboxId}`,
    );
  }

  async #getOrCreateCommandRouterClient(
    taskId: string,
  ): Promise<TaskCommandRouterClientImpl> {
    if (this.#commandRouterClient !== undefined) {
      return this.#commandRouterClient;
    }

    if (this.#commandRouterClientPromise !== undefined) {
      return this.#commandRouterClientPromise;
    }

    const promise = (async () => {
      const client = await TaskCommandRouterClientImpl.tryInit(
        this.#client.cpClient,
        taskId,
        this.sandboxId,
        this.#isV2,
        this.#client.logger,
        this.#client.profile,
      );
      if (!client) {
        throw new Error(
          "Command router access is not available for this sandbox",
        );
      }
      if (!this.#attached) {
        client.close();
        throw new ClientClosedError();
      }
      this.#commandRouterClient = client;
      return client;
    })();
    this.#commandRouterClientPromise = promise;

    try {
      return await promise;
    } catch (err) {
      // clear the Promise so subsequent calls can retry
      if (this.#commandRouterClientPromise === promise) {
        this.#commandRouterClientPromise = undefined;
      }
      throw err;
    }
  }

  /**
   * Create a token for making HTTP connections to the Sandbox.
   */
  async createConnectToken(
    params?: SandboxCreateConnectTokenParams,
  ): Promise<SandboxCreateConnectCredentials> {
    this.#ensureAttached();
    this.#ensureV1("createConnectToken");
    const port = params?.port ?? 8080;
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      throw new InvalidError(
        `createConnectToken() expects \`port\` in [1, 65535], got ${port}`,
      );
    }
    const resp = await this.#client.cpClient.sandboxCreateConnectToken({
      sandboxId: this.sandboxId,
      userMetadata: params?.userMetadata,
      port,
    });
    return { url: resp.url, token: resp.token };
  }

  async terminate(): Promise<void>;
  async terminate(params: { wait: true }): Promise<number>;
  async terminate(params?: SandboxTerminateParams): Promise<number | void> {
    this.#ensureAttached();
    await this.#sandboxTerminate();

    let exitCode: number | undefined;
    if (params?.wait) {
      exitCode = await this.wait();
    }

    this.#taskId = undefined;
    this.detach();
    return exitCode;
  }

  /**
   * Disconnect from the Sandbox, cleaning up local resources.
   * The Sandbox continues running on Modal's infrastructure.
   * After calling detach(), most operations on this Sandbox object will throw.
   */
  detach(): void {
    this.#commandRouterClient?.close();
    this.#attached = false;
    this.#commandRouterClient = undefined;
    this.#commandRouterClientPromise = undefined;
  }

  async wait(): Promise<number> {
    while (true) {
      const resp = await this.#sandboxWait(10);
      if (resp.result) {
        const returnCode = getReturnCode(resp.result)!;
        this.#client.logger.debug(
          "Sandbox wait completed",
          "sandbox_id",
          this.sandboxId,
          "status",
          resp.result.status,
          "return_code",
          returnCode,
        );
        return returnCode;
      }
    }
  }

  /**
   * Wait until the Sandbox readiness probe reports the Sandbox is ready.
   *
   * This method only works for Sandboxes configured with a readiness probe.
   *
   * @param timeoutMs - Maximum total time to wait, in milliseconds.
   * @returns A promise that resolves once the Sandbox is ready.
   * @throws {@link TimeoutError} If readiness is not reported before `timeoutMs`.
   */
  async waitUntilReady(timeoutMs = 300_000): Promise<void> {
    this.#ensureAttached();
    if (timeoutMs <= 0) {
      throw new InvalidError(`timeoutMs must be positive, got ${timeoutMs}`);
    }

    // Route to the task command router for both V1 and V2 sandboxes.
    const [taskId, commandRouterClient] = await this.#getCommandRouter();
    await commandRouterClient.sandboxWaitUntilReady(taskId, timeoutMs);
  }

  /** Get {@link Tunnel} metadata for the Sandbox.
   *
   * Raises {@link SandboxTimeoutError} if the tunnels are not available after the timeout.
   *
   * @returns A dictionary of {@link Tunnel} objects which are keyed by the container port.
   */
  async tunnels(timeoutMs = 50000): Promise<Record<number, Tunnel>> {
    this.#ensureAttached();
    if (this.#tunnels) {
      return this.#tunnels;
    }

    const resp = await this.#sandboxGetTunnels(timeoutMs);

    if (
      resp.result?.status === GenericResult_GenericStatus.GENERIC_STATUS_TIMEOUT
    ) {
      throw new SandboxTimeoutError();
    }

    this.#tunnels = {};
    for (const t of resp.tunnels) {
      this.#tunnels[t.containerPort] = new Tunnel(
        t.host,
        t.port,
        t.unencryptedHost,
        t.unencryptedPort,
      );
    }

    return this.#tunnels;
  }

  /**
   * Snapshot the filesystem of the Sandbox.
   *
   * Returns an {@link Image} object which can be used to spawn a new Sandbox with the same filesystem.
   *
   * The call has an overall `timeoutMs` budget (default: 55000). If it
   * elapses before a snapshot completes, the call is cancelled and an
   * error is thrown.
   *
   * @param params - Optional parameters; see {@link SandboxSnapshotFilesystemParams}.
   * @returns Promise that resolves to an {@link Image}
   */
  async snapshotFilesystem(
    params?: SandboxSnapshotFilesystemParams,
  ): Promise<Image> {
    this.#ensureAttached();
    const wireTtlSeconds = resolveTtlSeconds(params?.ttlMs);
    // Treat both undefined and 0 as "use default", matching the Go
    // `Timeout time.Duration` zero-value convention on SandboxSnapshotFilesystemParams.
    const timeoutMs = params?.timeoutMs || 55000;
    const [taskId, commandRouterClient] = await this.#getCommandRouter();

    const request = TaskSnapshotFilesystemRequest.create({
      taskId,
      snapshotId: uuidv4(),
      ttlSeconds: wireTtlSeconds,
    });

    const response = await commandRouterClient.snapshotFilesystem(request, {
      timeoutMs,
    });

    if (!response.imageId) {
      throw new Error("Sandbox snapshot filesystem response missing `imageId`");
    }

    return new Image(this.#client, response.imageId, "");
  }

  /**
   * Mount an {@link Image} at a path in the Sandbox filesystem.
   *
   * @param path - The path where the directory should be mounted
   * @param image - Optional {@link Image} to mount. If undefined, mounts an empty directory.
   * @param params - Optional parameters; see {@link SandboxMountImageParams}.
   */
  async mountImage(
    path: string,
    image?: Image,
    params?: SandboxMountImageParams,
  ): Promise<void> {
    this.#ensureAttached();
    const [taskId, commandRouterClient] = await this.#getCommandRouter();

    if (image && !image.imageId) {
      throw new Error(
        "Image must be built before mounting. Call `image.build(app)` first.",
      );
    }

    const imageId = image?.imageId ?? "";
    const request = buildTaskMountDirectoryRequestProto(
      taskId,
      path,
      imageId,
      params,
    );
    await commandRouterClient.mountDirectory(request);
  }

  /**
   * Unmounts an {@link Image} previously mounted at a path in the Sandbox filesystem.
   *
   * @param path - The mount path to unmount
   */
  async unmountImage(path: string): Promise<void> {
    this.#ensureAttached();
    const [taskId, commandRouterClient] = await this.#getCommandRouter();

    const pathBytes = new TextEncoder().encode(path);
    const request = TaskUnmountDirectoryRequest.create({
      taskId,
      path: pathBytes,
    });
    await commandRouterClient.unmountDirectory(request);
  }

  /**
   * Updates the outbound network policy of a running Sandbox.
   *
   * Established connections that the new policy no longer permits are terminated.
   *
   * @param params - Both `outboundCidrAllowlist` and `outboundDomainAllowlist` must be provided.
   */
  async updateNetworkPolicy(
    params: SandboxUpdateNetworkPolicyParams,
  ): Promise<void> {
    this.#ensureAttached();
    if (
      params.outboundCidrAllowlist === undefined ||
      params.outboundDomainAllowlist === undefined
    ) {
      throw new InvalidError(
        "updateNetworkPolicy currently requires both outboundCidrAllowlist and outboundDomainAllowlist to be set",
      );
    }
    const [taskId, commandRouterClient] = await this.#getCommandRouter();

    const request = TaskSetNetworkAccessRequest.create({
      taskId,
      networkAccess: NetworkAccess.create({
        networkAccessType: NetworkAccess_NetworkAccessType.ALLOWLIST,
        allowedCidrs: params.outboundCidrAllowlist,
        allowedDomains: params.outboundDomainAllowlist,
      }),
    });
    await commandRouterClient.setNetworkAccess(request);
  }

  /**
   * Reload all Volumes mounted in the Sandbox.
   *
   * Blocks until the Volumes have been reloaded, bounded by `timeoutMs` (55000
   * by default). If the reload does not complete within that window, a
   * `TimeoutError` is thrown; note that the reload may still complete in the
   * background.
   *
   * @param params - Optional parameters; see {@link SandboxReloadVolumesParams}.
   */
  async reloadVolumes(params?: SandboxReloadVolumesParams): Promise<void> {
    this.#ensureAttached();
    if (params?.timeoutMs !== undefined && params.timeoutMs < 0) {
      throw new InvalidError("`timeoutMs` must not be negative");
    }
    // Both undefined and 0 fall back to the default, matching the Go
    // `Timeout time.Duration` zero-value convention.
    const timeoutMs = params?.timeoutMs || 55000;
    const [taskId, commandRouterClient] = await this.#getCommandRouter();
    await commandRouterClient.reloadVolumes(
      TaskReloadVolumesRequest.create({ taskId }),
      { timeoutMs },
    );
  }

  /**
   * Snapshots and creates a new {@link Image} from a directory in the running sandbox.
   *
   * The resulting Image is retained for `ttlMs` (default: 30 days),
   * as a hard cutoff measured from creation — usage does not extend
   * the lifetime. Pass `ttlMs: null` to retain indefinitely.
   *
   * The call has an overall `timeoutMs` budget (default: 55000). If it
   * elapses before a snapshot completes, the call is cancelled and an
   * error is thrown.
   *
   * @param path - The path of the directory to snapshot
   * @param params - Optional parameters; see {@link SandboxSnapshotDirectoryParams}.
   * @returns Promise that resolves to an {@link Image}
   */
  async snapshotDirectory(
    path: string,
    params?: SandboxSnapshotDirectoryParams,
  ): Promise<Image> {
    this.#ensureAttached();
    const wireTtlSeconds = resolveTtlSeconds(params?.ttlMs);
    // Treat both undefined and 0 as "use default", matching the Go
    // `Timeout time.Duration` zero-value convention on SandboxSnapshotDirectoryParams.
    const timeoutMs = params?.timeoutMs || 55000;
    const [taskId, commandRouterClient] = await this.#getCommandRouter();

    // snapshotId guarantees idempotency under retries.
    const request = buildTaskSnapshotDirectoryRequestProto(
      taskId,
      path,
      uuidv4(),
      wireTtlSeconds,
      params,
    );
    const response = await commandRouterClient.snapshotDirectory(request, {
      timeoutMs,
    });

    if (!response.imageId) {
      throw new Error("Sandbox snapshot directory response missing `imageId`");
    }

    return new Image(this.#client, response.imageId, "");
  }

  /**
   * Check if the Sandbox has finished running.
   *
   * Returns `null` if the Sandbox is still running, else returns the exit code.
   */
  async poll(): Promise<number | null> {
    this.#ensureAttached();
    const resp = await this.#sandboxWait(0);

    return getReturnCode(resp.result);
  }
}

export class ContainerProcess<R extends string | Uint8Array = any> {
  stdin: ModalWriteStream<R>;
  stdout: ModalReadStream<R>;
  stderr: ModalReadStream<R>;

  readonly #taskId: string;
  readonly #execId: string;
  readonly #commandRouterClient: TaskCommandRouterClientImpl;
  readonly #deadline: number | null;

  /** @ignore */
  constructor(
    taskId: string,
    execId: string,
    commandRouterClient: TaskCommandRouterClientImpl,
    params?: SandboxExecParams,
    deadline?: number | null,
  ) {
    this.#taskId = taskId;
    this.#execId = execId;
    this.#commandRouterClient = commandRouterClient;
    this.#deadline = deadline ?? null;

    const mode = params?.mode ?? "text";
    const stdout = params?.stdout ?? "pipe";
    const stderr = params?.stderr ?? "pipe";

    this.stdin = toModalWriteStream(
      inputStreamCp<R>(commandRouterClient, taskId, execId),
    );

    const stdoutStream =
      stdout === "ignore"
        ? ReadableStream.from([])
        : streamConsumingIter(
            outputStreamCp(
              commandRouterClient,
              taskId,
              execId,
              FileDescriptor.FILE_DESCRIPTOR_STDOUT,
              this.#deadline,
            ),
          );

    const stderrStream =
      stderr === "ignore"
        ? ReadableStream.from([])
        : streamConsumingIter(
            outputStreamCp(
              commandRouterClient,
              taskId,
              execId,
              FileDescriptor.FILE_DESCRIPTOR_STDERR,
              this.#deadline,
            ),
          );

    if (mode === "text") {
      this.stdout = toModalReadStream(
        stdoutStream.pipeThrough(new TextDecoderStream()),
      ) as ModalReadStream<R>;
      this.stderr = toModalReadStream(
        stderrStream.pipeThrough(new TextDecoderStream()),
      ) as ModalReadStream<R>;
    } else {
      this.stdout = toModalReadStream(stdoutStream) as ModalReadStream<R>;
      this.stderr = toModalReadStream(stderrStream) as ModalReadStream<R>;
    }
  }

  /**
   * @ignore
   * Send stdin EOF directly, bypassing the WritableStream state machine.
   *
   * Useful for when an stdin write has failed.
   */
  async closeStdin(): Promise<void> {
    await this.#commandRouterClient
      .execStdinWrite(this.#taskId, this.#execId, 0, new Uint8Array(0), true)
      .catch(() => {});
  }

  /** Wait for process completion and return the exit code. */
  async wait(): Promise<number> {
    const resp = await this.#commandRouterClient.execWait(
      this.#taskId,
      this.#execId,
      this.#deadline,
    );
    if (resp.code !== undefined) {
      return resp.code;
    } else if (resp.signal !== undefined) {
      return 128 + resp.signal;
    } else {
      throw new InvalidError("Unexpected exit status");
    }
  }
}

// Like _StreamReader with object_type == "sandbox".
async function* outputStreamSb(
  cpClient: ModalGrpcClient,
  sandboxId: string,
  fileDescriptor: FileDescriptor,
  signal?: AbortSignal,
): AsyncIterable<Uint8Array> {
  let lastIndex = "0-0";
  let completed = false;
  let retriesRemaining = SB_LOGS_MAX_RETRIES;
  let delayMs = SB_LOGS_INITIAL_DELAY_MS;
  while (!completed) {
    try {
      const outputIterator = cpClient.sandboxGetLogs(
        {
          sandboxId,
          fileDescriptor,
          timeout: 55,
          lastEntryId: lastIndex,
        },
        { signal },
      );
      for await (const batch of outputIterator) {
        // Successful read - reset backoff counters.
        delayMs = SB_LOGS_INITIAL_DELAY_MS;
        retriesRemaining = SB_LOGS_MAX_RETRIES;
        lastIndex = batch.entryId;
        yield* batch.items.map((item) => new TextEncoder().encode(item.data));
        if (batch.eof) {
          completed = true;
          break;
        }
        if (signal?.aborted) {
          return;
        }
      }
    } catch (err) {
      // If cancelled, exit cleanly regardless of error type.
      if (signal?.aborted) {
        return;
      }
      if (isRetryableGrpc(err) && retriesRemaining > 0) {
        // Short exponential backoff to avoid tight retry loops.
        try {
          await setTimeout(delayMs, undefined, { signal });
        } catch {
          // Abort during sleep - exit cleanly.
          return;
        }
        delayMs *= SB_LOGS_DELAY_FACTOR;
        retriesRemaining--;
        continue;
      } else {
        throw err;
      }
    }
  }
}

function inputStreamSb(
  cpClient: ModalGrpcClient,
  sandboxId: string,
): WritableStream<string> {
  let index = 1;
  return new WritableStream<string>({
    async write(chunk) {
      await cpClient.sandboxStdinWrite({
        sandboxId,
        input: encodeIfString(chunk),
        index,
      });
      index++;
    },
    async close() {
      await cpClient.sandboxStdinWrite({
        sandboxId,
        index,
        eof: true,
      });
    },
  });
}

async function* outputStreamCp(
  commandRouterClient: TaskCommandRouterClientImpl,
  taskId: string,
  execId: string,
  fileDescriptor: FileDescriptor,
  deadline: number | null,
): AsyncIterable<Uint8Array> {
  for await (const batch of commandRouterClient.execStdioRead(
    taskId,
    execId,
    fileDescriptor,
    deadline,
  )) {
    yield batch.data;
  }
}

function inputStreamCp<R extends string | Uint8Array>(
  commandRouterClient: TaskCommandRouterClientImpl,
  taskId: string,
  execId: string,
): WritableStream<R> {
  let offset = 0;
  return new WritableStream<R>({
    async write(chunk) {
      const data = encodeIfString(chunk);
      await commandRouterClient.execStdinWrite(
        taskId,
        execId,
        offset,
        data,
        false, // eof
      );
      offset += data.length;
    },
    async close() {
      await commandRouterClient.execStdinWrite(
        taskId,
        execId,
        offset,
        new Uint8Array(0),
        true, // eof
      );
    },
  });
}

function encodeIfString(chunk: Uint8Array | string): Uint8Array {
  return typeof chunk === "string" ? new TextEncoder().encode(chunk) : chunk;
}
