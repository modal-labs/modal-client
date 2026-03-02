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
  NetworkAccess,
  NetworkAccess_NetworkAccessType,
  VolumeMount,
  CloudBucketMount as CloudBucketMountProto,
  SchedulerPlacement,
  TunnelType,
  PortSpec,
  Resources,
  PortSpecs,
} from "../proto/modal_proto/api";
import {
  TaskExecStartRequest,
  TaskExecStdoutConfig,
  TaskExecStderrConfig,
  TaskMountDirectoryRequest,
  TaskSnapshotDirectoryRequest,
} from "../proto/modal_proto/task_command_router";
import { TaskCommandRouterClientImpl } from "./task_command_router_client";
import { v4 as uuidv4 } from "uuid";
import {
  getDefaultClient,
  type ModalClient,
  isRetryableGrpc,
  ModalGrpcClient,
} from "./client";
import {
  runFilesystemExec,
  SandboxFile,
  SandboxFileMode,
} from "./sandbox_filesystem";
import {
  type ModalReadStream,
  type ModalWriteStream,
  streamConsumingIter,
  toModalReadStream,
  toModalWriteStream,
} from "./streams";
import { type Secret, mergeEnvIntoSecrets } from "./secret";
import {
  InvalidError,
  NotFoundError,
  SandboxTimeoutError,
  AlreadyExistsError,
  ClientClosedError,
} from "./errors";
import { Image } from "./image";
import type { Volume } from "./volume";
import type { Proxy } from "./proxy";
import type { CloudBucketMount } from "./cloud_bucket_mount";
import type { App } from "./app";
import { parseGpuConfig } from "./app";
import { checkForRenamedParams } from "./validation";

// Backoff configuration for SandboxGetLogs retry behavior.
const SB_LOGS_INITIAL_DELAY_MS = 10;
const SB_LOGS_DELAY_FACTOR = 2;
const SB_LOGS_MAX_RETRIES = 10;

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

  /** Enable a PTY for the Sandbox. */
  pty?: boolean;

  /** List of ports to tunnel into the Sandbox. Encrypted ports are tunneled with TLS. */
  encryptedPorts?: number[];

  /** List of encrypted ports to tunnel into the Sandbox, using HTTP/2. */
  h2Ports?: number[];

  /** List of ports to tunnel into the Sandbox without encryption. */
  unencryptedPorts?: number[];

  /** Whether to block all network access from the Sandbox. */
  blockNetwork?: boolean;

  /** List of CIDRs the Sandbox is allowed to access. If None, all CIDRs are allowed. Cannot be used with blockNetwork. */
  cidrAllowlist?: string[];

  /** Cloud provider to run the Sandbox on. */
  cloud?: string;

  /** Region(s) to run the Sandbox on. */
  regions?: string[];

  /** Enable verbose logging. */
  verbose?: boolean;

  /** Reference to a Modal {@link Proxy} to use in front of this Sandbox. */
  proxy?: Proxy;

  /** Optional name for the Sandbox. Unique within an App. */
  name?: string;

  /** Optional experimental options. */
  experimentalOptions?: Record<string, any>;

  /** If set, connections to this Sandbox will be subdomains of this domain rather than the default.
   * This requires prior manual setup by Modal and is only available for Enterprise customers.
   */
  customDomain?: string;
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
    ? Object.entries(params.volumes).map(([mountPath, volume]) => ({
        volumeId: volume.volumeId,
        mountPath,
        allowBackgroundCommits: true,
        readOnly: volume.isReadOnly,
      }))
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
    if (params.cidrAllowlist) {
      throw new Error(
        "cidrAllowlist cannot be used when blockNetwork is enabled",
      );
    }
    networkAccess = {
      networkAccessType: NetworkAccess_NetworkAccessType.BLOCKED,
      allowedCidrs: [],
    };
  } else if (params.cidrAllowlist) {
    networkAccess = {
      networkAccessType: NetworkAccess_NetworkAccessType.ALLOWLIST,
      allowedCidrs: params.cidrAllowlist,
    };
  } else {
    networkAccess = {
      networkAccessType: NetworkAccess_NetworkAccessType.OPEN,
      allowedCidrs: [],
    };
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
  // option type in the future. Currently, the proto only supports Record<string, boolean> so we validate
  // the input here.
  const protoExperimentalOptions: Record<string, boolean> =
    params.experimentalOptions
      ? Object.entries(params.experimentalOptions).reduce(
          (acc, [name, value]) => {
            if (typeof value !== "boolean") {
              throw new Error(
                `experimental option '${name}' must be a boolean, got ${value}`,
              );
            }
            acc[name] = Boolean(value);
            return acc;
          },
          {} as Record<string, boolean>,
        )
      : {};

  return SandboxCreateRequest.create({
    appId,
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
      name: params.name,
      experimentalOptions: protoExperimentalOptions,
      customDomain: params.customDomain,
    },
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

  /** Returns a running {@link Sandbox} object from an ID.
   *
   * @returns Sandbox with ID
   */
  async fromId(sandboxId: string): Promise<Sandbox> {
    try {
      await this.#client.cpClient.sandboxWait({
        sandboxId,
        timeout: 0,
      });
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(`Sandbox with id: '${sandboxId}' not found`);
      throw err;
    }

    return new Sandbox(this.#client, sandboxId);
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

/** Optional parameters for {@link SandboxService#fromName client.sandboxes.fromName()}. */
export type SandboxFromNameParams = {
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
  /** Enable a PTY for the command. */
  pty?: boolean;
};

/** Optional parameters for {@link Sandbox#createConnectToken Sandbox.createConnectToken()}. */
export type SandboxTerminateParams = {
  /** If true, wait for the Sandbox to finish and return the exit code. */
  wait?: boolean;
};

export type SandboxCreateConnectTokenParams = {
  /** Optional user-provided metadata string that will be added to the headers by the proxy when forwarding requests to the Sandbox. */
  userMetadata?: string;
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

export function buildTaskExecStartRequestProto(
  taskId: string,
  execId: string,
  command: string[],
  params?: SandboxExecParams,
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
    ptyInfo,
    runtimeDebug: false,
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
  #commandRouterClient: TaskCommandRouterClientImpl | undefined;
  #commandRouterClientPromise: Promise<TaskCommandRouterClientImpl> | undefined;
  #attached: boolean = true;

  /** @ignore */
  constructor(client: ModalClient, sandboxId: string) {
    this.#client = client;
    this.sandboxId = sandboxId;
  }

  get stdin(): ModalWriteStream<string> {
    if (!this.#stdin) {
      this.#stdin = toModalWriteStream(
        inputStreamSb(this.#client.cpClient, this.sandboxId),
      );
    }
    return this.#stdin;
  }

  get stdout(): ModalReadStream<string> {
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

  /** Set tags (key-value pairs) on the Sandbox. Tags can be used to filter results in {@link SandboxService#list Sandbox.list}. */
  async setTags(tags: Record<string, string>): Promise<void> {
    this.#ensureAttached();
    const tagsList = Object.entries(tags).map(([tagName, tagValue]) => ({
      tagName,
      tagValue,
    }));
    try {
      await this.#client.cpClient.sandboxTagsSet({
        environmentName: this.#client.environmentName(),
        sandboxId: this.sandboxId,
        tags: tagsList,
      });
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
      resp = await this.#client.cpClient.sandboxTagsGet({
        sandboxId: this.sandboxId,
      });
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
   * @deprecated Use {@link SandboxService#fromId client.sandboxes.fromId()} instead.
   */
  static async fromId(sandboxId: string): Promise<Sandbox> {
    return getDefaultClient().sandboxes.fromId(sandboxId);
  }

  /**
   * @deprecated Use {@link SandboxService#fromName client.sandboxes.fromName()} instead.
   */
  static async fromName(
    appName: string,
    name: string,
    environment?: string,
  ): Promise<Sandbox> {
    return getDefaultClient().sandboxes.fromName(appName, name, {
      environment,
    });
  }

  /**
   * Open a file in the Sandbox filesystem.
   * @param path - Path to the file to open
   * @param mode - File open mode (r, w, a, r+, w+, a+)
   * @returns Promise that resolves to a {@link SandboxFile}
   */
  async open(path: string, mode: SandboxFileMode = "r"): Promise<SandboxFile> {
    this.#ensureAttached();
    const taskId = await this.#getTaskId();
    const resp = await runFilesystemExec(this.#client.cpClient, {
      fileOpenRequest: {
        path,
        mode,
      },
      taskId,
    });
    // For Open request, the file descriptor is always set
    const fileDescriptor = resp.response.fileDescriptor as string;
    return new SandboxFile(this.#client, fileDescriptor, taskId);
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
    this.#ensureAttached();
    validateExecArgs(command);
    const taskId = await this.#getTaskId();

    const mergedSecrets = await mergeEnvIntoSecrets(
      this.#client,
      params?.env,
      params?.secrets,
    );
    const mergedParams = {
      ...params,
      secrets: mergedSecrets,
      env: undefined, // setting env to undefined just to clarify it's not needed anymore
    };

    const commandRouterClient =
      await this.#getOrCreateCommandRouterClient(taskId);

    const execId = uuidv4();
    const request = buildTaskExecStartRequestProto(
      taskId,
      execId,
      command,
      mergedParams,
    );

    await commandRouterClient.execStart(request);

    this.#client.logger.debug(
      "Created ContainerProcess",
      "exec_id",
      execId,
      "sandbox_id",
      this.sandboxId,
      "command",
      command,
    );

    const deadline = mergedParams?.timeoutMs
      ? Date.now() + mergedParams.timeoutMs
      : null;

    return new ContainerProcess(
      taskId,
      execId,
      commandRouterClient,
      mergedParams,
      deadline,
    );
  }

  #ensureAttached(): void {
    if (!this.#attached) {
      throw new ClientClosedError();
    }
  }

  static readonly #maxGetTaskIdAttempts = 600; // 5 minutes at 500ms intervals

  async #getTaskId(): Promise<string> {
    if (this.#taskId !== undefined) {
      return this.#taskId;
    }
    for (let i = 0; i < Sandbox.#maxGetTaskIdAttempts; i++) {
      const resp = await this.#client.cpClient.sandboxGetTaskId({
        sandboxId: this.sandboxId,
      });
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
    const resp = await this.#client.cpClient.sandboxCreateConnectToken({
      sandboxId: this.sandboxId,
      userMetadata: params?.userMetadata,
    });
    return { url: resp.url, token: resp.token };
  }

  async terminate(): Promise<void>;
  async terminate(params: { wait: true }): Promise<number>;
  async terminate(params?: SandboxTerminateParams): Promise<number | void> {
    this.#ensureAttached();
    await this.#client.cpClient.sandboxTerminate({ sandboxId: this.sandboxId });

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
      const resp = await this.#client.cpClient.sandboxWait({
        sandboxId: this.sandboxId,
        timeout: 10,
      });
      if (resp.result) {
        const returnCode = Sandbox.#getReturnCode(resp.result)!;
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

    const resp = await this.#client.cpClient.sandboxGetTunnels({
      sandboxId: this.sandboxId,
      timeout: timeoutMs / 1000,
    });

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
   * @param timeoutMs - Timeout for the snapshot operation in milliseconds
   * @returns Promise that resolves to an {@link Image}
   */
  async snapshotFilesystem(timeoutMs = 55000): Promise<Image> {
    this.#ensureAttached();
    const resp = await this.#client.cpClient.sandboxSnapshotFs({
      sandboxId: this.sandboxId,
      timeout: timeoutMs / 1000,
    });

    if (
      resp.result?.status !== GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS
    ) {
      throw new Error(
        `Sandbox snapshot failed: ${resp.result?.exception || "Unknown error"}`,
      );
    }

    if (!resp.imageId) {
      throw new Error("Sandbox snapshot response missing `imageId`");
    }

    return new Image(this.#client, resp.imageId, "");
  }

  /**
   * Mount an {@link Image} at a path in the Sandbox filesystem.
   *
   * @param path - The path where the directory should be mounted
   * @param image - Optional {@link Image} to mount. If undefined, mounts an empty directory.
   */
  async mountImage(path: string, image?: Image): Promise<void> {
    this.#ensureAttached();
    const taskId = await this.#getTaskId();
    const commandRouterClient =
      await this.#getOrCreateCommandRouterClient(taskId);

    if (image && !image.imageId) {
      throw new Error(
        "Image must be built before mounting. Call `image.build(app)` first.",
      );
    }

    const pathBytes = new TextEncoder().encode(path);
    const imageId = image?.imageId ?? "";
    const request = TaskMountDirectoryRequest.create({
      taskId,
      path: pathBytes,
      imageId,
    });
    await commandRouterClient.mountDirectory(request);
  }

  /**
   * Snapshots and creates a new {@link Image} from a directory in the running sandbox.
   *
   * @param path - The path of the directory to snapshot
   * @returns Promise that resolves to an {@link Image}
   */
  async snapshotDirectory(path: string): Promise<Image> {
    this.#ensureAttached();
    const taskId = await this.#getTaskId();
    const commandRouterClient =
      await this.#getOrCreateCommandRouterClient(taskId);

    const pathBytes = new TextEncoder().encode(path);
    const request = TaskSnapshotDirectoryRequest.create({
      taskId,
      path: pathBytes,
    });
    const response = await commandRouterClient.snapshotDirectory(request);

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
    const resp = await this.#client.cpClient.sandboxWait({
      sandboxId: this.sandboxId,
      timeout: 0,
    });

    return Sandbox.#getReturnCode(resp.result);
  }

  /**
   * @deprecated Use {@link SandboxService#list client.sandboxes.list()} instead.
   */
  static async *list(
    params: SandboxListParams = {},
  ): AsyncGenerator<Sandbox, void, unknown> {
    yield* getDefaultClient().sandboxes.list(params);
  }

  static #getReturnCode(result: GenericResult | undefined): number | null {
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
