import {
  FileDescriptor,
  GenericResult,
  GenericResult_GenericStatus,
  PTYInfo,
  PTYInfo_PTYType,
  SandboxCreateRequest,
  SandboxCreateV2Request,
  NetworkAccess,
  NetworkAccess_NetworkAccessType,
  VolumeMount,
  SchedulerPlacement,
  TunnelType,
  PortSpec,
  Resources,
  PortSpecs,
  Probe as ProbeProto,
  SecretSource,
  SandboxRestoreRequest_SandboxNameOverrideType,
} from "../proto/modal_proto/api";
import {
  TaskExecStartRequest,
  TaskExecStdoutConfig,
  TaskExecStderrConfig,
  TaskMountDirectoryRequest,
  TaskSnapshotDirectoryRequest,
} from "../proto/modal_proto/task_command_router";
import {
  TaskCommandRouterClientImpl,
  type CommandRouterAccess,
  type StdinSource,
} from "./task_command_router_client";
import { ExperimentalOutboundPolicy } from "./outbound_policy";
import { type ModalClient } from "./client";
import { SandboxFilesystem } from "./sandbox_fs";
import { SidecarService } from "./sandbox_sidecar";
import {
  decodeTextIter,
  type ModalReadStream,
  type ModalWriteStream,
  streamConsumingIter,
  toModalReadStream,
  toModalWriteStream,
} from "./streams";
import { type Secret, hydrateSecrets, secretSources } from "./secret";
import { InvalidError, NotFoundError, ClientClosedError } from "./errors";
import { Image } from "./image";
import { SandboxSnapshot } from "./sandbox_snapshot";
import { checkObjectName } from "./name_utils";
import { volumeToMountProto, type Volume } from "./volume";
import type { Proxy } from "./proxy";
import {
  buildCloudBucketMountProtos,
  type CloudBucketMount,
} from "./cloud_bucket_mount";
import type { App } from "./app";
import { parseGpuConfig } from "./app";
import { checkForRenamedParams } from "./validation";
import { SandboxLogsManager } from "./logs";
import type { SandboxImpl } from "./sandbox_impl";
import { SandboxImplV1 } from "./sandbox_impl_v1";
import { SandboxImplV2 } from "./sandbox_impl_v2";

// How long create() waits for capacity before giving up.
/** @internal @hidden */
export const SANDBOX_SCHEDULING_TIMEOUT_MS = 21 * 60 * 1000;

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
/** @internal @hidden */
export function resolveTtlSeconds(ttlMs: number | null | undefined): number {
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

/** @ignore
 * Determines which backend a Sandbox ID belongs to. IDs that don't match the
 * V1 shape are assumed to be V2, so that Sandbox IDs minted in newer formats
 * route to the newer backend instead of failing client-side.
 */
export function getSandboxVersion(sandboxId: string): SandboxVersion {
  if (isV1SandboxId(sandboxId)) {
    return SandboxVersion.V1;
  }
  return SandboxVersion.V2;
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

/** Runtime a Sandbox runs in: a virtual machine, or gVisor. */
export type SandboxRuntime = "gvisor" | "vm";

/** The value for the Sandbox definition's runtime field, or undefined to let Modal pick. */
function sandboxRuntimeToProto(
  runtime: SandboxRuntime | undefined,
): string | undefined {
  switch (runtime) {
    case undefined:
      return undefined;
    case "gvisor":
    case "vm":
      return runtime;
    default:
      throw new Error(
        `runtime must be "gvisor" or "vm", got ${JSON.stringify(runtime)}`,
      );
  }
}

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

  /** Runtime under which the Sandbox executes, or undefined to let Modal pick. */
  runtime?: SandboxRuntime;

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

  /** Configuration for replacing headers in outbound HTTPS requests from the Sandbox.
   * Secrets referenced by the policy are resolved outside the Sandbox and are never
   * visible to the workload. See {@link ExperimentalOutboundPolicy}.
   * EXPERIMENTAL: the API is subject to change. */
  experimentalOutboundPolicy?: ExperimentalOutboundPolicy;

  /** List of CIDRs allowed to connect inbound to the Sandbox (encrypted tunnels and connection tokens). If not set, all IPs are allowed. Cannot be used with blockNetwork. */
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

  /**
   * Optional experimental options. Values must be booleans or strings.
   */
  experimentalOptions?: Record<string, any>;

  /** If set, connections to this Sandbox will be subdomains of this domain rather than the default.
   * This requires prior manual setup by Modal and is only available for Enterprise customers.
   */
  customDomain?: string;

  /** If true, the sandbox will receive a MODAL_IDENTITY_TOKEN env var for OIDC-based auth (e.g. to AWS, GCP). */
  includeOidcIdentityToken?: boolean;

  /** Enable memory snapshots. */
  experimentalEnableSnapshot?: boolean;
};

/**
 * Build the outbound {@link NetworkAccess} for the given params, defaulting to
 * open access when nothing is specified. Shared by the Sandbox and sidecar
 * create paths.
 * @internal
 * @hidden
 */
export function buildOutboundNetworkAccess(
  blockNetwork: boolean | undefined,
  outboundCidrAllowlist: string[] | undefined,
  outboundDomainAllowlist: string[] | undefined,
): NetworkAccess {
  if (blockNetwork) {
    if (outboundCidrAllowlist) {
      throw new InvalidError(
        "outboundCidrAllowlist cannot be used when blockNetwork is enabled",
      );
    }
    if (outboundDomainAllowlist) {
      throw new InvalidError(
        "outboundDomainAllowlist cannot be used when blockNetwork is enabled",
      );
    }
    return NetworkAccess.create({
      networkAccessType: NetworkAccess_NetworkAccessType.BLOCKED,
      allowedCidrs: [],
      allowedDomains: [],
    });
  }
  if (outboundCidrAllowlist || outboundDomainAllowlist) {
    return NetworkAccess.create({
      networkAccessType: NetworkAccess_NetworkAccessType.ALLOWLIST,
      allowedCidrs: outboundCidrAllowlist || [],
      allowedDomains: outboundDomainAllowlist || [],
    });
  }
  return NetworkAccess.create({
    networkAccessType: NetworkAccess_NetworkAccessType.OPEN,
    allowedCidrs: [],
    allowedDomains: [],
  });
}

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

  if (params.name) {
    checkObjectName(params.name, "Sandbox");
  }

  const runtime = sandboxRuntimeToProto(params.runtime);

  const volumeMounts: VolumeMount[] = params.volumes
    ? Object.entries(params.volumes).map(([mountPath, volume]) =>
        volumeToMountProto(mountPath, volume),
      )
    : [];

  const cloudBucketMounts = buildCloudBucketMountProtos(
    params.cloudBucketMounts,
  );

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

  if (params.blockNetwork) {
    if (params.i6pn) {
      throw new InvalidError(
        "blockNetwork disables all networking, including i6pn. To keep i6pn while blocking " +
          "public egress, use an empty outbound allowlist (outboundCidrAllowlist: []) instead.",
      );
    }
    if (params.inboundCidrAllowlist) {
      throw new Error(
        "inboundCidrAllowlist cannot be used when blockNetwork is enabled",
      );
    }
  }
  if (params.experimentalOutboundPolicy !== undefined) {
    params.experimentalOutboundPolicy._validate();
    if (params.blockNetwork) {
      throw new InvalidError(
        "experimentalOutboundPolicy cannot be used when blockNetwork is enabled",
      );
    }
    if (
      params.outboundDomainAllowlist &&
      params.outboundDomainAllowlist.length > 0
    ) {
      throw new InvalidError(
        "experimentalOutboundPolicy cannot be used with outboundDomainAllowlist",
      );
    }
  }
  const networkAccess = buildOutboundNetworkAccess(
    params.blockNetwork,
    params.outboundCidrAllowlist,
    params.outboundDomainAllowlist,
  );

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
      experimentalOutboundPolicy: params.experimentalOutboundPolicy?._toProto(),
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
      enableSnapshot: params.experimentalEnableSnapshot ?? false,
      runtime,
    },
  });
}

export async function buildSandboxCreateV2RequestProto(
  appId: string,
  imageId: string,
  params: SandboxCreateParams = {},
  secretSources: SecretSource[] = [],
): Promise<SandboxCreateV2Request> {
  if (params.gpu) {
    throw new Error("GPUs are not supported by experimentalCreate");
  }

  const req = await buildSandboxCreateRequestProto(appId, imageId, params);

  return SandboxCreateV2Request.create({
    appId: req.appId,
    definition: req.definition,
    secretSources,
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
    const implClass = implClassForCreateParams(this.#client, params);
    const impl = await implClass.create(this.#client, app, image, params);
    return new Sandbox(impl);
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
   * Pass `experimentalEnableSnapshot: true` to create a sandbox that can be
   * snapshotted with {@link Sandbox#experimentalSnapshot}. Features like network
   * file systems and GPUs are not supported.
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
    const impl = await SandboxImplV2.create(this.#client, app, image, params);
    return new Sandbox(impl);
  }

  /** Returns a running {@link Sandbox} object from an ID.
   *
   * @returns Sandbox with ID
   */
  async fromId(sandboxId: string): Promise<Sandbox> {
    return new Sandbox(newSandboxImpl(this.#client, sandboxId));
  }

  /** Get a running {@link Sandbox} by name from a deployed {@link App}.
   *
   * Raises a {@link NotFoundError} if no running Sandbox is found with the given name.
   * If `includeTerminated` is set to `true` and no Sandbox with the name is currently
   * running, the most recent terminated Sandbox will be returned instead.
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
    if (params?.includeTerminated || this.#client.profile.sandboxV2) {
      try {
        return await this.experimentalFromName(appName, name, params);
      } catch (err) {
        if (!(err instanceof NotFoundError) || params?.includeTerminated)
          throw err;
      }
    }

    const impl = await SandboxImplV1.fromName(
      this.#client,
      appName,
      name,
      params,
    );
    return new Sandbox(impl);
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
    const impl = await SandboxImplV2.fromName(
      this.#client,
      appName,
      name,
      params,
    );
    return new Sandbox(impl);
  }

  /**
   * Restore a {@link Sandbox} from a memory snapshot.
   *
   * The restore targets the same backend the snapshot was taken from. A V1
   * snapshot restores as a V1 sandbox, a V2 snapshot as a V2 sandbox.
   *
   * EXPERIMENTAL: the API is subject to change.
   */
  async experimentalFromSnapshot(
    snapshot: SandboxSnapshot,
    params?: SandboxExperimentalFromSnapshotParams,
  ): Promise<Sandbox> {
    await snapshot._hydrate();
    const useV2 = snapshot._isV2 ?? false;

    let overrideType: SandboxRestoreRequest_SandboxNameOverrideType;
    let nameOverride: string | undefined;
    if (params?.name === undefined) {
      overrideType =
        SandboxRestoreRequest_SandboxNameOverrideType.SANDBOX_NAME_OVERRIDE_TYPE_UNSPECIFIED;
    } else if (params.name === null) {
      overrideType =
        SandboxRestoreRequest_SandboxNameOverrideType.SANDBOX_NAME_OVERRIDE_TYPE_NONE;
    } else {
      checkObjectName(params.name, "Sandbox");
      overrideType =
        SandboxRestoreRequest_SandboxNameOverrideType.SANDBOX_NAME_OVERRIDE_TYPE_STRING;
      nameOverride = params.name;
    }

    const implClass = useV2 ? SandboxImplV2 : SandboxImplV1;
    const impl = await implClass.fromSnapshot(
      this.#client,
      snapshot,
      overrideType,
      nameOverride,
    );
    return new Sandbox(impl);
  }

  /**
   * List all {@link Sandbox}es for the current Environment or App ID (if specified).
   * If tags are specified, only Sandboxes that have at least those tags are returned.
   */
  async *list(
    params: SandboxListParams = {},
  ): AsyncGenerator<Sandbox, void, unknown> {
    if (this.#client.profile.sandboxV2) {
      yield* this.experimentalList(params);
      return;
    }

    for await (const impl of SandboxImplV1.list(this.#client, params)) {
      yield new Sandbox(impl);
    }
  }

  /**
   * List V2 {@link Sandbox}es.
   *
   * This lists all Sandboxes (both v1 and v2).
   * Pass `appId` to scope to an App; omit it to list across the current
   * environment (deprecated — prefer scoping by `appId`). If tags are specified,
   * only Sandboxes that have at least those tags are returned.
   *
   * Yields {@link Sandbox} objects that are currently running.
   *
   * EXPERIMENTAL: the API is subject to change.
   */
  async *experimentalList(
    params: SandboxExperimentalListParams = {},
  ): AsyncGenerator<Sandbox, void, unknown> {
    for await (const impl of SandboxImplV2.list(this.#client, params)) {
      yield new Sandbox(impl);
    }
  }
}

/**
 * Select the implementation backing the Sandbox with the given ID.
 *
 * @internal
 * @hidden
 */
export function implClassForSandboxId(
  sandboxId: string,
): typeof SandboxImplV1 | typeof SandboxImplV2 {
  if (getSandboxVersion(sandboxId) === SandboxVersion.V1) {
    return SandboxImplV1;
  }
  return SandboxImplV2;
}

/**
 * Construct the implementation backing the Sandbox with the given ID.
 *
 * Server responses can return a Sandbox of either backend regardless of which
 * RPC was called, so the backend is always deduced from the returned ID.
 *
 * @internal
 * @hidden
 */
export function newSandboxImpl(
  client: ModalClient,
  sandboxId: string,
  params: {
    taskId?: string;
    tunnels?: Record<number, Tunnel>;
    initCommandRouterAccess?: CommandRouterAccess;
  } = {},
): SandboxImpl {
  const implClass = implClassForSandboxId(sandboxId);
  return new implClass(client, sandboxId, params);
}

/**
 * Select the implementation that creates the Sandbox described by `params`.
 *
 * @internal
 * @hidden
 */
export function implClassForCreateParams(
  client: ModalClient,
  params: SandboxCreateParams,
): typeof SandboxImplV1 | typeof SandboxImplV2 {
  // GPUs are not supported on V2, so those calls use the V1 backend
  // regardless of the setting.
  if (params.gpu) {
    return SandboxImplV1;
  }
  return client.profile.sandboxV2 ? SandboxImplV2 : SandboxImplV1;
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
  /** The App to list Sandboxes under. Omit to list across the environment (deprecated). */
  appId?: string;
  /** Only return Sandboxes that include all specified tags. */
  tags?: Record<string, string>;
  /** Override environment for the request; defaults to current profile. */
  environment?: string;
};

/** Optional parameters for {@link SandboxService#fromName client.sandboxes.fromName()}. */
export type SandboxFromNameParams = {
  environment?: string;
  /**
   * If `includeTerminated` is set to `true` and no Sandbox with the name is
   * currently running, the most recent terminated Sandbox will be returned instead.
   */
  includeTerminated?: boolean;
};

/** Optional parameters for {@link SandboxService#experimentalFromName client.sandboxes.experimentalFromName()}. */
export type SandboxExperimentalFromNameParams = {
  environment?: string;
  /**
   * If `includeTerminated` is set to `true` and no Sandbox with the name is
   * currently running, the most recent terminated Sandbox will be returned instead.
   */
  includeTerminated?: boolean;
};

/** Optional parameters for {@link SandboxService#experimentalFromSnapshot client.sandboxes.experimentalFromSnapshot()}. */
export type SandboxExperimentalFromSnapshotParams = {
  /**
   * Name for the restored Sandbox. Omit to reuse the original Sandbox's name,
   * pass `null` to leave it unnamed, or pass a string to override it.
   */
  name?: string | null;
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

/** Optional parameters for Sandbox.experimentalGetExitSnapshot(). */
export type SandboxExperimentalGetExitSnapshotParams = {
  /**
   * Total time to wait in milliseconds, spread across repeated long polls of
   * at most `EXIT_SNAPSHOT_LONG_POLL_TIMEOUT` seconds each. `undefined` (the
   * default) waits until the snapshot reaches a terminal state. `0` performs
   * an immediate check without waiting.
   */
  timeoutMs?: number;
};

/** Optional parameters for {@link Sandbox#reloadVolumes Sandbox.reloadVolumes()}. */
export type SandboxReloadVolumesParams = {
  /** Overall budget in milliseconds. Defaults to 55000. */
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
    secretSources: secretSources(params?.secrets ?? [], params?.env),
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
  containerId = "",
): TaskMountDirectoryRequest {
  return TaskMountDirectoryRequest.create({
    taskId,
    path: new TextEncoder().encode(path),
    imageId,
    customerSuppliedEncryptionKey: validateExperimentalEncryptionKey(
      params?.experimentalEncryptionKey,
    ),
    containerId,
  });
}

/** @ignore */
export function resolveMountImageId(image?: Image): string {
  if (image && !image.imageId) {
    throw new Error(
      "Image must be built before mounting. Call `image.build(app)` first.",
    );
  }
  return image?.imageId ?? "";
}

/** @ignore */
export function buildTaskSnapshotDirectoryRequestProto(
  taskId: string,
  path: string,
  snapshotId: string,
  ttlSeconds: number,
  params?: SandboxSnapshotDirectoryParams,
  containerId = "",
): TaskSnapshotDirectoryRequest {
  return TaskSnapshotDirectoryRequest.create({
    taskId,
    path: new TextEncoder().encode(path),
    snapshotId,
    ttlSeconds,
    customerSuppliedEncryptionKey: validateExperimentalEncryptionKey(
      params?.experimentalEncryptionKey,
    ),
    containerId,
  });
}

/** Sandboxes are secure, isolated containers in Modal that boot in seconds. */
export class Sandbox {
  readonly #client: ModalClient;
  readonly #impl: SandboxImpl;
  readonly sandboxId: string;

  #stdin?: ModalWriteStream<string>;
  #stdout?: ModalReadStream<string>;
  #stderr?: ModalReadStream<string>;
  #attached: boolean = true;
  #filesystem?: SandboxFilesystem;
  #sidecars?: SidecarService;

  /**
   * Wraps a backend implementation; use the {@link SandboxService} factory
   * methods (`client.sandboxes.create()`, `client.sandboxes.fromId()`, ...)
   * to obtain Sandboxes.
   *
   * @ignore
   */
  constructor(impl: SandboxImpl) {
    this.#client = impl.client;
    this.#impl = impl;
    this.sandboxId = impl.sandboxId;
  }

  get stdin(): ModalWriteStream<string> {
    if (!this.#stdin) {
      this.#stdin = this.#impl.stdin();
    }
    return this.#stdin;
  }

  get stdout(): ModalReadStream<string> {
    if (!this.#stdout) {
      this.#stdout = this.#outputStream(FileDescriptor.FILE_DESCRIPTOR_STDOUT);
    }
    return this.#stdout;
  }

  get stderr(): ModalReadStream<string> {
    if (!this.#stderr) {
      this.#stderr = this.#outputStream(FileDescriptor.FILE_DESCRIPTOR_STDERR);
    }
    return this.#stderr;
  }

  /**
   * Access persisted entrypoint logs emitted by this Sandbox.
   *
   * Use {@link SandboxLogsManager#fetch fetch()} to read logs from a UTC time
   * range and {@link SandboxLogsManager#tail tail()} to read the most recent
   * logs.
   *
   * This is separate from {@link Sandbox#stdout stdout} and
   * {@link Sandbox#stderr stderr}, which expose the Sandbox's live output
   * streams.
   */
  get logs(): SandboxLogsManager {
    return new SandboxLogsManager(
      this.#client,
      this.sandboxId,
      () => this.#impl.getAppId(),
      () => this.#impl.resolveTaskIdForLogs(),
    );
  }

  #outputStream(fileDescriptor: FileDescriptor): ModalReadStream<string> {
    return this.#impl.outputStream(fileDescriptor, () =>
      this.#ensureAttached(),
    );
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
        ensureAttached: () => this.#ensureAttached(),
        exec: (command, params, containerId) =>
          this.#execInternal(command, params, containerId),
        commandRouter: () => this.#getCommandRouter(),
        reloadVolumes: (containerId, params) =>
          this.#impl.reloadVolumes(containerId, params),
        snapshotFilesystem: (containerId, params) =>
          this.#impl.snapshotFilesystem(containerId, params),
        validateCreate: (options) => this.#impl.validateSidecarCreate(options),
        create: (request) => this.#impl.createSidecar(request),
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
    await this.#impl.setTags(tags);
  }

  /** Get tags (key-value pairs) currently attached to this Sandbox from the server. */
  async getTags(): Promise<Record<string, string>> {
    this.#ensureAttached();
    return await this.#impl.getTags();
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
    await this.#impl.setName(name);
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
    return await this.#impl.exec(command, params, containerId);
  }

  async #getCommandRouter(
    signal?: AbortSignal,
  ): Promise<[string, TaskCommandRouterClientImpl]> {
    this.#ensureAttached();
    return await this.#impl.getCommandRouter(signal);
  }

  #ensureAttached(): void {
    if (!this.#attached) {
      throw new ClientClosedError();
    }
  }

  /**
   * Blocks until the Sandbox has a task. If the wait fails, the Sandbox is
   * terminated so a queued one cannot start later with no caller. A timeout
   * surfaces as a ResourceExhaustedError.
   *
   * @internal
   * @hidden
   */
  async _waitForScheduling(timeoutMs: number): Promise<void> {
    await this.#impl.waitForScheduling(timeoutMs);
  }

  /**
   * Create a token for making HTTP connections to the Sandbox.
   */
  async createConnectToken(
    params?: SandboxCreateConnectTokenParams,
  ): Promise<SandboxCreateConnectCredentials> {
    this.#ensureAttached();
    const port = params?.port ?? 8080;
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      throw new InvalidError(
        `createConnectToken() expects \`port\` in [1, 65535], got ${port}`,
      );
    }
    return await this.#impl.createConnectToken(port, params?.userMetadata);
  }

  async terminate(): Promise<void>;
  async terminate(params: { wait: true }): Promise<number>;
  async terminate(params?: SandboxTerminateParams): Promise<number | void> {
    this.#ensureAttached();
    await this.#impl.terminate();

    let exitCode: number | undefined;
    if (params?.wait) {
      exitCode = await this.wait();
    }

    // The Sandbox stays attached, in case further interactions like reading
    // stdout/stderr are needed.
    return exitCode;
  }

  /**
   * Disconnect from the Sandbox, cleaning up local resources.
   * The Sandbox continues running on Modal's infrastructure.
   * After calling detach(), most operations on this Sandbox object will throw.
   *
   * This does not block on or interrupt ongoing reads or calls. Connection
   * resources are closed promptly once those operations finish.
   */
  detach(): void {
    this.#attached = false;
    this.#impl.close();
  }

  async wait(): Promise<number> {
    return await this.#impl.wait();
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
    await this.#impl.waitUntilReady(timeoutMs);
  }

  /** Get {@link Tunnel} metadata for the Sandbox.
   *
   * Raises {@link SandboxTimeoutError} if the tunnels are not available after the timeout.
   *
   * @returns A dictionary of {@link Tunnel} objects which are keyed by the container port.
   */
  async tunnels(timeoutMs = 50000): Promise<Record<number, Tunnel>> {
    this.#ensureAttached();
    return await this.#impl.tunnels(timeoutMs);
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
    return await this.#impl.snapshotFilesystem("", params);
  }

  /**
   * Get the exit filesystem snapshot image.
   *
   * An exit snapshot captures the Sandbox filesystem when the Sandbox exits,
   * whether its entrypoint finishes gracefully, abruptly, or it is stopped with
   * {@link Sandbox#terminate}. The resulting {@link Image} can be passed to
   * {@link SandboxService#create client.sandboxes.create()} to start a new
   * Sandbox from the saved filesystem.
   *
   * Exit snapshots are opt-in: the Sandbox must have been created with
   * `experimentalOptions: { enable_exit_snapshot: true }`. Calling this on a
   * Sandbox created without that option throws an {@link InvalidError}.
   *
   * @example
   * ```ts
   * const sb = await modal.sandboxes.create(app, image, {
   *   experimentalOptions: { enable_exit_snapshot: true },
   * });
   * // ... use the Sandbox ...
   * await sb.terminate();
   * const image = await sb.experimentalGetExitSnapshot();
   * const sb2 = await modal.sandboxes.create(app, image);
   * ```
   *
   * EXPERIMENTAL: the API is subject to change.
   *
   * @param params - Optional parameters; see SandboxExperimentalGetExitSnapshotParams.
   * @returns The exit snapshot Image.
   * @throws {InvalidError} If `timeoutMs` is negative, or if exit snapshots
   *   were not enabled when the Sandbox was created.
   * @throws {TimeoutError} If `timeoutMs` elapses before the snapshot
   *   reaches a terminal state. This includes `timeoutMs = 0` when the
   *   snapshot is still pending.
   * @throws {SnapshotCreationError} Snapshot operation is done and failed.
   *   Polling again will not produce an Image; filesystem state is gone.
   * @throws {NotFoundError} If the Sandbox does not exist.
   */
  async experimentalGetExitSnapshot(
    params?: SandboxExperimentalGetExitSnapshotParams,
  ): Promise<Image> {
    return await this.#impl.getExitSnapshot(params);
  }

  /**
   * Snapshot the filesystem and memory of the Sandbox.
   *
   * Returns a {@link SandboxSnapshot} which can be restored into a new Sandbox
   * with {@link SandboxService#experimentalFromSnapshot client.sandboxes.experimentalFromSnapshot()}.
   *
   * The Sandbox must have been created with `experimentalEnableSnapshot: true`.
   *
   * EXPERIMENTAL: the API is subject to change.
   */
  async experimentalSnapshot(): Promise<SandboxSnapshot> {
    this.#ensureAttached();
    return await this.#impl.snapshotMemory();
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
    await this.#impl.mountImage(path, image, params);
  }

  /**
   * Unmounts an {@link Image} previously mounted at a path in the Sandbox filesystem.
   *
   * @param path - The mount path to unmount
   */
  async unmountImage(path: string): Promise<void> {
    this.#ensureAttached();
    await this.#impl.unmountImage(path);
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
    await this.#impl.updateNetworkPolicy(params);
  }

  /**
   * Replace the outbound policy of a running Sandbox.
   *
   * EXPERIMENTAL: the API is subject to change.
   *
   * The new policy replaces all existing policy configuration on the
   * Sandbox; build a policy including any existing rules you want to keep.
   *
   * Only Sandboxes created with an `experimentalOutboundPolicy` can be updated this
   * way; for Sandboxes created without one this fails, since header
   * replacement is only set up at creation time.
   */
  async experimentalUpdateOutboundPolicy(
    outboundPolicy: ExperimentalOutboundPolicy,
  ): Promise<void> {
    this.#ensureAttached();
    outboundPolicy._validate();
    await this.#impl.updateOutboundPolicy(outboundPolicy);
  }

  /**
   * Reload all Volumes mounted in the Sandbox.
   *
   * Blocks until the reload completes, or throws a `TimeoutError` on timeout
   * (the reload may still complete in the background).
   *
   * @param params - Optional parameters; see {@link SandboxReloadVolumesParams}.
   */
  async reloadVolumes(params?: SandboxReloadVolumesParams): Promise<void> {
    this.#ensureAttached();
    await this.#impl.reloadVolumes("", params);
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
    return await this.#impl.snapshotDirectory(path, params);
  }

  /**
   * Check if the Sandbox has finished running.
   *
   * Returns `null` if the Sandbox is still running, else returns the exit code.
   */
  async poll(): Promise<number | null> {
    this.#ensureAttached();
    return await this.#impl.poll();
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

    // Nothing here reads: the first read is what opens the request. Decoding
    // must stay in the iterable, above the stream - a transform pumps its
    // source on its own schedule rather than on demand, and for these sources
    // a read is what opens the request.
    const output = (
      behavior: StdioBehavior,
      fileDescriptor: FileDescriptor,
    ) => {
      if (behavior === "ignore") {
        return ReadableStream.from([]) as ReadableStream<R>;
      }
      const abort = new AbortController();
      const bytes = outputStreamCp(
        commandRouterClient,
        taskId,
        execId,
        fileDescriptor,
        this.#deadline,
        abort.signal,
      );
      const stream =
        mode === "text"
          ? streamConsumingIter(decodeTextIter(bytes), () => abort.abort())
          : streamConsumingIter(bytes, () => abort.abort());
      return stream as ReadableStream<R>;
    };

    this.stdout = toModalReadStream(
      output(stdout, FileDescriptor.FILE_DESCRIPTOR_STDOUT),
    );
    this.stderr = toModalReadStream(
      output(stderr, FileDescriptor.FILE_DESCRIPTOR_STDERR),
    );
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

  /**
   * @internal
   * @hidden
   * Stream `source` into the process's stdin and close it (EOF) on success.
   *
   * Returns the total bytes streamed.
   */
  async _stdinWriteStream(source: StdinSource): Promise<number> {
    return await this.#commandRouterClient.execStdinWriteStream(
      this.#taskId,
      this.#execId,
      source,
    );
  }

  /**
   * @internal
   * @hidden
   * Check if the container process has finished running.
   *
   * Returns `null` if the process is still running, else returns the exit
   * code.
   */
  async _poll(): Promise<number | null> {
    const resp = await this.#commandRouterClient.execPoll(
      this.#taskId,
      this.#execId,
      this.#deadline,
    );
    if (resp.code !== undefined) {
      return resp.code;
    } else if (resp.signal !== undefined) {
      return 128 + resp.signal;
    }
    return null;
  }

  /** Wait for process completion and return the exit code. */
  async wait(): Promise<number> {
    return await this._wait();
  }

  /**
   * @internal
   * @hidden
   * Like `wait()`, but stops waiting when `signal` is aborted.
   */
  async _wait(signal?: AbortSignal): Promise<number> {
    const resp = await this.#commandRouterClient.execWait(
      this.#taskId,
      this.#execId,
      this.#deadline,
      signal,
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
async function* outputStreamCp(
  commandRouterClient: TaskCommandRouterClientImpl,
  taskId: string,
  execId: string,
  fileDescriptor: FileDescriptor,
  deadline: number | null,
  signal: AbortSignal,
): AsyncIterable<Uint8Array> {
  for await (const batch of commandRouterClient.execStdioRead(
    taskId,
    execId,
    fileDescriptor,
    deadline,
    signal,
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

/** @internal @hidden */
export function encodeIfString(chunk: Uint8Array | string): Uint8Array {
  return typeof chunk === "string" ? new TextEncoder().encode(chunk) : chunk;
}
