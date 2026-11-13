import { ClientError, Status, type CallOptions } from "nice-grpc";
import { setTimeout } from "timers/promises";
import { v4 as uuidv4 } from "uuid";
import {
  type CloudBucketMount as CloudBucketMountProto,
  FileDescriptor,
  GenericResult_GenericStatus,
  NetworkAccess,
  NetworkAccess_NetworkAccessType,
  type PTYInfo,
  type SandboxGetExitSnapshotResponse,
  SandboxGetExitSnapshotResponse_ErrorCode,
  type SandboxGetTaskIdResponse,
  type SandboxGetTunnelsResponse,
  type TunnelData,
  type SandboxWaitResponse,
  type SecretSource,
  type VolumeMount,
} from "../proto/modal_proto/api";
import {
  TaskContainerCreateRequest,
  TaskReloadVolumesRequest,
  TaskSetExperimentalOutboundPolicyRequest,
  TaskSetNetworkAccessRequest,
  TaskSnapshotFilesystemRequest,
  TaskUnmountDirectoryRequest,
} from "../proto/modal_proto/task_command_router";
import {
  type ModalClient,
  type RetryOptions,
  type TimeoutOptions,
  getServerRetryPolicy,
  isRetryableGrpc,
} from "./client";
import {
  ClientClosedError,
  ConflictError,
  ExecutionError,
  InvalidError,
  NotFoundError,
  ResourceExhaustedError,
  SandboxTimeoutError,
  SnapshotCreationError,
  TimeoutError,
} from "./errors";
import { Image } from "./image";
import type { ExperimentalOutboundPolicy } from "./outbound_policy";
import {
  ContainerProcess,
  type SandboxCreateConnectCredentials,
  type SandboxExecParams,
  type SandboxExperimentalGetExitSnapshotParams,
  type SandboxMountImageParams,
  type SandboxReloadVolumesParams,
  type SandboxSnapshotDirectoryParams,
  type SandboxSnapshotFilesystemParams,
  type SandboxUpdateNetworkPolicyParams,
  Tunnel,
  buildTaskExecStartRequestProto,
  getReturnCode,
  buildTaskMountDirectoryRequestProto,
  buildTaskSnapshotDirectoryRequestProto,
  resolveMountImageId,
  resolveTtlSeconds,
  validateExecArgs,
} from "./sandbox";
import type { SandboxSnapshot } from "./sandbox_snapshot";
import {
  hydrateSecrets,
  resolvableSecrets,
  validateEnvVarKeys,
} from "./secret";
import type { ModalReadStream, ModalWriteStream } from "./streams";
import {
  TaskCommandRouterClientImpl,
  type CommandRouterAccess,
} from "./task_command_router_client";

/**
 * How long operations that need a running task wait for one.
 *
 * @internal
 * @hidden
 */
export const GET_TASK_ID_TIMEOUT_MS = 5 * 60 * 1000;
/**
 * How long a task ID lookup made to read logs waits for the task.
 *
 * @internal
 * @hidden
 */
export const SB_LOGS_TASK_ID_TIMEOUT_SECONDS = 0.5;

// Exit snapshot polling holds a server-side long poll for at most EXIT_SNAPSHOT_LONG_POLL_TIMEOUT
// seconds and then re-polls from the client, so a request is never held open indefinitely however
// long the caller waits.
const EXIT_SNAPSHOT_LONG_POLL_TIMEOUT = 10;
// Margin added to each poll's network deadline over the server-side hold, covering the
// response's return trip so an answer sent just before the hold expires isn't lost to the
// client-side deadline.
const EXIT_SNAPSHOT_POLL_DEADLINE_MARGIN = 5;
// The fetch loop absorbs transient poll failures and re-polls. It gives up once this many
// polls fail in a row, so an extended outage still surfaces to the caller.
const EXIT_SNAPSHOT_MAX_CONSECUTIVE_POLL_FAILURES = 3;
// Pause between failed polls, so failures that reject instantly (e.g. a refused connection)
// don't burn through the failure budget in milliseconds.
const EXIT_SNAPSHOT_POLL_FAILURE_BACKOFF = 1;

/**
 * Error message for operations that only persistent Sandboxes support.
 *
 * @internal
 * @hidden
 */
export function persistentOnlyMessage(methodName: string): string {
  return `Sandbox.${methodName} requires a persistent Sandbox, created with \`experimentalPersistent: true\`.`;
}

function isTransientTaskIDLookupError(err: unknown): boolean {
  if (!(err instanceof ClientError)) {
    return false;
  }
  return (
    err.code === Status.UNAVAILABLE ||
    err.code === Status.DEADLINE_EXCEEDED ||
    err.code === Status.INTERNAL
  );
}

/**
 * Sidecar options whose support depends on the Sandbox backend; checked by
 * {@link SandboxImpl.validateSidecarCreate} before a sidecar is created.
 *
 * @internal
 * @hidden
 */
export type SidecarCreateOptions = {
  hasCloudBucketMounts: boolean;
  hasProxy: boolean;
  includeOidcIdentityToken: boolean;
};

/**
 * Backend-independent description of a sidecar container to create.
 *
 * @internal
 * @hidden
 */
export type SidecarCreateRequest = {
  containerName: string;
  imageId: string;
  command: string[];
  workdir?: string;
  secretIds: string[];
  secretSources: SecretSource[];
  volumeMounts: VolumeMount[];
  cloudBucketMounts: CloudBucketMountProto[];
  networkAccess: NetworkAccess;
  includeOidcIdentityToken: boolean;
  proxyId?: string;
  ptyInfo?: PTYInfo;
  memoryReserveConsumeMiB?: number;
};

/**
 * Result of {@link SandboxImpl.createSidecar}.
 *
 * @internal
 * @hidden
 */
export type SidecarCreateResult = {
  containerId: string;
  containerName: string;
};

/**
 * Backend-specific operations behind a {@link Sandbox}.
 *
 * A `Sandbox` is a thin facade that owns object identity and attachment state;
 * everything that talks to a backend goes through one of these. Each backend
 * (V1, V2) is a separate class, so the facade has no per-backend branching.
 *
 * @internal
 * @hidden
 */
export interface SandboxImpl {
  readonly client: ModalClient;
  readonly sandboxId: string;

  stdin(): ModalWriteStream<string>;
  outputStream(
    fileDescriptor: FileDescriptor,
    ensureAttached: () => void,
  ): ModalReadStream<string>;

  getTags(): Promise<Record<string, string>>;
  setTags(tags: Record<string, string>): Promise<void>;
  setName(name: string): Promise<void>;
  createConnectToken(
    port: number,
    userMetadata: string | undefined,
  ): Promise<SandboxCreateConnectCredentials>;

  terminate(): Promise<void>;
  /** Stop the current session of a persistent Sandbox, keeping its state. */
  pause(): Promise<void>;
  /** Resume a paused persistent Sandbox from its saved state. */
  resume(): Promise<void>;
  /** Release local resources (e.g. connections); the Sandbox keeps running. */
  close(): void;
  wait(): Promise<number>;
  poll(): Promise<number | null>;
  /**
   * Blocks until the Sandbox has a task. If the wait fails, the Sandbox is
   * terminated so a queued one cannot start later with no caller. A timeout
   * surfaces as a ResourceExhaustedError.
   */
  waitForScheduling(timeoutMs: number): Promise<void>;

  tunnels(timeoutMs: number): Promise<Record<number, Tunnel>>;
  getAppId(): Promise<string>;
  resolveTaskIdForLogs(): Promise<string>;
  getExitSnapshot(
    params: SandboxExperimentalGetExitSnapshotParams | undefined,
  ): Promise<Image>;
  snapshotMemory(): Promise<SandboxSnapshot>;

  getCommandRouter(
    signal?: AbortSignal,
  ): Promise<[string, TaskCommandRouterClientImpl]>;
  waitUntilReady(timeoutMs: number): Promise<void>;
  updateNetworkPolicy(params: SandboxUpdateNetworkPolicyParams): Promise<void>;
  updateOutboundPolicy(
    outboundPolicy: ExperimentalOutboundPolicy,
  ): Promise<void>;
  snapshotFilesystem(
    containerId: string,
    params: SandboxSnapshotFilesystemParams | undefined,
  ): Promise<Image>;
  snapshotDirectory(
    path: string,
    params: SandboxSnapshotDirectoryParams | undefined,
  ): Promise<Image>;
  mountImage(
    path: string,
    image: Image | undefined,
    params: SandboxMountImageParams | undefined,
  ): Promise<void>;
  unmountImage(path: string): Promise<void>;
  reloadVolumes(
    containerId: string,
    params: SandboxReloadVolumesParams | undefined,
  ): Promise<void>;
  exec(
    command: string[],
    params: SandboxExecParams | undefined,
    containerId?: string,
  ): Promise<ContainerProcess>;

  /** Reject sidecar options that `createSidecar` cannot provide on this backend. */
  validateSidecarCreate(options: SidecarCreateOptions): void;
  createSidecar(request: SidecarCreateRequest): Promise<SidecarCreateResult>;
}

/**
 * Optional state known when a Sandbox implementation is constructed, e.g.
 * from a create or restore response.
 *
 * @internal
 * @hidden
 */
export type SandboxImplParams = {
  taskId?: string;
  tunnels?: Record<number, Tunnel>;
  /** Only good for the first command router connection; later ones fetch fresh access. */
  initCommandRouterAccess?: CommandRouterAccess;
};

/**
 * A connection to a task command router that is opened on first use and then
 * shared.
 *
 * Concurrent first callers share a single open attempt, and its client is
 * reused until `close`. After `close`, `get` throws `ClientClosedError`,
 * including for handles obtained earlier that resolve the connection lazily;
 * a connection that finishes opening after `close` is closed instead of
 * handed out.
 *
 * @internal
 * @hidden
 */
export class LazyCommandRouterClient {
  readonly #open: (taskId: string) => Promise<TaskCommandRouterClientImpl>;
  #client: TaskCommandRouterClientImpl | undefined;
  #pending: Promise<TaskCommandRouterClientImpl> | undefined;
  #closed = false;

  constructor(open: (taskId: string) => Promise<TaskCommandRouterClientImpl>) {
    this.#open = open;
  }

  /** Throws `ClientClosedError` once `close` has been called. */
  ensureOpen(): void {
    if (this.#closed) {
      throw new ClientClosedError();
    }
  }

  async get(taskId: string): Promise<TaskCommandRouterClientImpl> {
    this.ensureOpen();
    if (this.#client !== undefined) {
      return this.#client;
    }
    if (this.#pending !== undefined) {
      return this.#pending;
    }

    const pending = (async () => {
      const client = await this.#open(taskId);
      // Opening yields control, so `close` may have been called in the meantime.
      if (this.#closed) {
        client.close();
        throw new ClientClosedError();
      }
      this.#client = client;
      return client;
    })();
    this.#pending = pending;

    try {
      return await pending;
    } catch (err) {
      // clear the Promise so subsequent calls can retry
      if (this.#pending === pending) {
        this.#pending = undefined;
      }
      throw err;
    }
  }

  /** Close permanently, releasing the connection once in-flight operations complete. */
  close(): void {
    this.#closed = true;
    this.#client?.closeWhenIdle();
    this.#client = undefined;
    this.#pending = undefined;
  }
}

/**
 * Open a command router connection to the Sandbox's task. `isV2` selects how
 * the connection authenticates; `access` is used for this connection if given,
 * otherwise access is fetched from the server.
 *
 * @internal
 * @hidden
 */
export async function connectCommandRouter(
  client: ModalClient,
  taskId: string,
  sandboxId: string,
  isV2: boolean,
  access: CommandRouterAccess | undefined,
): Promise<TaskCommandRouterClientImpl> {
  const routerClient = await TaskCommandRouterClientImpl.tryInit(
    client.cpClient,
    taskId,
    sandboxId,
    isV2,
    access,
    client.logger,
    client.profile,
  );
  if (!routerClient) {
    throw new Error("Command router access is not available for this sandbox");
  }
  return routerClient;
}

/**
 * Polls `rpcGetTaskId` until the Sandbox has been assigned a task. Throws a
 * ConflictError if the Sandbox finished before that happened and a
 * TimeoutError once timeoutMs elapses. With retryTransient, lookups that fail
 * with a transient error keep polling until the deadline instead of failing.
 *
 * @internal
 * @hidden
 */
export async function pollTaskId(
  sandboxId: string,
  rpcGetTaskId: (options: CallOptions) => Promise<SandboxGetTaskIdResponse>,
  timeoutMs: number,
  retryTransient: boolean,
  signal?: AbortSignal,
): Promise<string> {
  const deadline = Date.now() + timeoutMs;
  const timedOut = () =>
    new TimeoutError(`Timed out waiting for task ID for Sandbox ${sandboxId}`);
  const pollIntervalMs = 500;
  const noTimeLeft = () => deadline - Date.now() <= pollIntervalMs;
  while (true) {
    let resp;
    try {
      resp = await rpcGetTaskId({ signal });
    } catch (err) {
      if (signal?.aborted || !isTransientTaskIDLookupError(err)) {
        throw err;
      }
      if (!retryTransient && Date.now() < deadline) {
        throw err;
      }
      // A transient failure at the deadline says nothing about scheduling,
      // so report the wait as timed out rather than the last hiccup. A
      // server error is not evidence of missing capacity, so surface it.
      if (noTimeLeft()) {
        if (err instanceof ClientError && err.code === Status.INTERNAL) {
          throw err;
        }
        throw timedOut();
      }
      await setTimeout(pollIntervalMs, undefined, { signal });
      continue;
    }
    if (resp.taskId) {
      return resp.taskId;
    }
    if (resp.taskResult) {
      throw new ConflictError(
        resp.taskResult.exception ||
          `Sandbox ${sandboxId} has already finished`,
      );
    }
    if (noTimeLeft()) {
      throw timedOut();
    }
    await setTimeout(pollIntervalMs, undefined, { signal });
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
export async function waitForScheduling(
  impl: SandboxImpl,
  waitForTaskId: (timeoutMs: number) => Promise<string>,
  timeoutMs: number,
): Promise<void> {
  try {
    await waitForTaskId(timeoutMs);
  } catch (err) {
    if (err instanceof ConflictError) {
      throw err;
    }
    try {
      await impl.terminate();
    } catch (terminateErr) {
      impl.client.logger.debug(
        "Failed to terminate unscheduled Sandbox",
        "sandbox_id",
        impl.sandboxId,
        "error",
        terminateErr,
      );
    }
    if (err instanceof TimeoutError) {
      throw new ResourceExhaustedError(
        "Insufficient capacity to create sandbox.",
      );
    }
    throw err;
  }
}

/**
 * The task ID from a `SandboxGetTaskId` lookup made with a short
 * {@link SB_LOGS_TASK_ID_TIMEOUT_SECONDS} timeout, for reading logs. Fails
 * rather than waiting when the Sandbox has no task yet.
 *
 * @internal
 * @hidden
 */
export async function taskIdForLogs(
  rpcGetTaskId: () => Promise<SandboxGetTaskIdResponse>,
): Promise<string> {
  let resp;
  try {
    resp = await rpcGetTaskId();
  } catch (error) {
    if (
      error instanceof ClientError &&
      error.code === Status.DEADLINE_EXCEEDED
    ) {
      throw new ExecutionError("Sandbox task ID is not available yet.");
    }
    throw error;
  }

  if (!resp.taskId) {
    throw new ExecutionError("Sandbox task ID is not available.");
  }
  return resp.taskId;
}

/**
 * Polls `rpcWait` until the Sandbox has finished and returns its exit code.
 *
 * @internal
 * @hidden
 */
export async function waitForReturnCode(
  impl: SandboxImpl,
  rpcWait: (timeoutSecs: number) => Promise<SandboxWaitResponse>,
): Promise<number> {
  while (true) {
    const resp = await rpcWait(10);
    if (resp.result) {
      const returnCode = getReturnCode(resp.result)!;
      impl.client.logger.debug(
        "Sandbox wait completed",
        "sandbox_id",
        impl.sandboxId,
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
 * The App ID from a `SandboxWait` response.
 *
 * @internal
 * @hidden
 */
export function appIdFromWaitResponse(resp: SandboxWaitResponse): string {
  const appId = resp.metadata?.appId;
  if (!appId) {
    throw new ExecutionError(
      "Sandbox app ID should have been set during hydration",
    );
  }
  return appId;
}

/**
 * The Sandbox's tunnels from a `SandboxGetTunnels` response.
 *
 * @internal
 * @hidden
 */
export function tunnelsFromResponse(
  resp: SandboxGetTunnelsResponse,
): Record<number, Tunnel> {
  if (
    resp.result?.status === GenericResult_GenericStatus.GENERIC_STATUS_TIMEOUT
  ) {
    throw new SandboxTimeoutError();
  }
  return tunnelsFromProtos(resp.tunnels);
}

/**
 * Convert tunnel protos into Tunnels keyed by container port.
 *
 * @internal
 * @hidden
 */
export function tunnelsFromProtos(
  protos: TunnelData[],
): Record<number, Tunnel> {
  const tunnels: Record<number, Tunnel> = {};
  for (const t of protos) {
    tunnels[t.containerPort] = new Tunnel(
      t.host,
      t.port,
      t.unencryptedHost,
      t.unencryptedPort,
    );
  }
  return tunnels;
}

/**
 * Polls `rpcGetExitSnapshot` until the Sandbox's exit snapshot is available
 * and returns it as an Image.
 *
 * @internal
 * @hidden
 */
export async function pollExitSnapshot(
  impl: SandboxImpl,
  params: SandboxExperimentalGetExitSnapshotParams | undefined,
  rpcGetExitSnapshot: (
    timeoutSecs: number,
    options: CallOptions & TimeoutOptions & RetryOptions,
  ) => Promise<SandboxGetExitSnapshotResponse>,
): Promise<Image> {
  const timeoutMs = params?.timeoutMs;
  if (timeoutMs !== undefined && timeoutMs < 0) {
    throw new InvalidError(`timeoutMs must be non-negative, got ${timeoutMs}`);
  }

  const deadline = timeoutMs === undefined ? undefined : Date.now() + timeoutMs;
  const timeoutMessage = `timed out waiting for exit snapshot for Sandbox ${impl.sandboxId}`;
  let consecutivePollFailures = 0;
  while (true) {
    const requestTimeoutSecs =
      deadline === undefined
        ? EXIT_SNAPSHOT_LONG_POLL_TIMEOUT
        : Math.min(
            EXIT_SNAPSHOT_LONG_POLL_TIMEOUT,
            Math.max(0, (deadline - Date.now()) / 1000),
          );

    let statusDetails: Uint8Array | undefined;
    let resp;
    try {
      resp = await rpcGetExitSnapshot(requestTimeoutSecs, {
        timeoutMs:
          (requestTimeoutSecs + EXIT_SNAPSHOT_POLL_DEADLINE_MARGIN) * 1000,
        retries: 0,
        onTrailer: (trailer) => {
          const raw = trailer.get("grpc-status-details-bin");
          if (raw != null) statusDetails = raw as Uint8Array;
        },
      });
    } catch (err) {
      const retryPolicy = getServerRetryPolicy(statusDetails);
      if (retryPolicy) {
        if (deadline !== undefined && Date.now() >= deadline) {
          throw new TimeoutError(timeoutMessage);
        }
        consecutivePollFailures = 0;
        let throttleDelayMs = Math.max(retryPolicy.retryAfterSecs * 1000, 100);
        if (deadline !== undefined) {
          throttleDelayMs = Math.min(
            throttleDelayMs,
            Math.max(0, deadline - Date.now()),
          );
        }
        await setTimeout(throttleDelayMs);
        continue;
      }
      if (err instanceof ClientError && err.code === Status.INVALID_ARGUMENT)
        throw new InvalidError(err.details);
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(err.details);
      if (isRetryableGrpc(err)) {
        if (deadline !== undefined && Date.now() >= deadline) {
          throw new TimeoutError(timeoutMessage);
        }
        consecutivePollFailures += 1;
        if (
          consecutivePollFailures >= EXIT_SNAPSHOT_MAX_CONSECUTIVE_POLL_FAILURES
        ) {
          throw err;
        }
        let backoffMs = EXIT_SNAPSHOT_POLL_FAILURE_BACKOFF * 1000;
        if (deadline !== undefined) {
          backoffMs = Math.min(backoffMs, Math.max(0, deadline - Date.now()));
        }
        await setTimeout(backoffMs);
        continue;
      }
      throw err;
    }
    consecutivePollFailures = 0;

    if (resp.success) {
      if (!resp.success.imageId) {
        throw new ExecutionError("Exit snapshot result is missing image ID");
      }
      return new Image(impl.client, resp.success.imageId, "");
    }
    if (resp.error) {
      if (
        resp.error.errorCode ===
          SandboxGetExitSnapshotResponse_ErrorCode.ERROR_CODE_TIMEOUT ||
        resp.error.errorCode ===
          SandboxGetExitSnapshotResponse_ErrorCode.ERROR_CODE_FILESYSTEM_INCONSISTENT
      ) {
        throw new SnapshotCreationError(
          resp.error.message || "No exit snapshot image will be produced",
        );
      }
      throw new ExecutionError(
        resp.error.message || "Exit snapshot state is malformed",
      );
    }
    if (!resp.pending) {
      throw new ExecutionError("Exit snapshot response is missing an outcome");
    }
    if (deadline !== undefined && Date.now() >= deadline) {
      throw new TimeoutError(timeoutMessage);
    }
  }
}

/** @internal @hidden */
export async function waitUntilReadyViaTcr(
  impl: SandboxImpl,
  timeoutMs: number,
): Promise<void> {
  const [taskId, commandRouterClient] = await impl.getCommandRouter();
  await commandRouterClient.sandboxWaitUntilReady(taskId, timeoutMs);
}

/** @internal @hidden */
export async function updateNetworkPolicyViaTcr(
  impl: SandboxImpl,
  params: SandboxUpdateNetworkPolicyParams,
): Promise<void> {
  const [taskId, commandRouterClient] = await impl.getCommandRouter();

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

/** @internal @hidden */
export async function updateOutboundPolicyViaTcr(
  impl: SandboxImpl,
  outboundPolicy: ExperimentalOutboundPolicy,
): Promise<void> {
  const [, [taskId, commandRouterClient]] = await Promise.all([
    hydrateSecrets(impl.client, outboundPolicy._secrets()),
    impl.getCommandRouter(),
  ]);

  const request = TaskSetExperimentalOutboundPolicyRequest.create({
    taskId,
    experimentalOutboundPolicy: outboundPolicy._toProto(),
  });
  await commandRouterClient.setOutboundPolicy(request);
}

/** @internal @hidden */
export async function snapshotFilesystemViaTcr(
  impl: SandboxImpl,
  containerId: string,
  params: SandboxSnapshotFilesystemParams | undefined,
): Promise<Image> {
  const wireTtlSeconds = resolveTtlSeconds(params?.ttlMs);
  // Treat both undefined and 0 as "use default", matching the Go
  // `Timeout time.Duration` zero-value convention on SandboxSnapshotFilesystemParams.
  const timeoutMs = params?.timeoutMs || 55000;
  const [taskId, commandRouterClient] = await impl.getCommandRouter();

  const request = TaskSnapshotFilesystemRequest.create({
    taskId,
    snapshotId: uuidv4(),
    ttlSeconds: wireTtlSeconds,
    containerId,
  });

  const response = await commandRouterClient.snapshotFilesystem(request, {
    timeoutMs,
  });

  if (!response.imageId) {
    throw new Error("Sandbox snapshot filesystem response missing `imageId`");
  }

  return new Image(impl.client, response.imageId, "");
}

/** @internal @hidden */
export async function snapshotDirectoryViaTcr(
  impl: SandboxImpl,
  path: string,
  params: SandboxSnapshotDirectoryParams | undefined,
): Promise<Image> {
  const wireTtlSeconds = resolveTtlSeconds(params?.ttlMs);
  // Treat both undefined and 0 as "use default", matching the Go
  // `Timeout time.Duration` zero-value convention on SandboxSnapshotDirectoryParams.
  const timeoutMs = params?.timeoutMs || 55000;
  const [taskId, commandRouterClient] = await impl.getCommandRouter();

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

  return new Image(impl.client, response.imageId, "");
}

/** @internal @hidden */
export async function mountImageViaTcr(
  impl: SandboxImpl,
  path: string,
  image: Image | undefined,
  params: SandboxMountImageParams | undefined,
): Promise<void> {
  const [taskId, commandRouterClient] = await impl.getCommandRouter();

  const imageId = resolveMountImageId(image);
  const request = buildTaskMountDirectoryRequestProto(
    taskId,
    path,
    imageId,
    params,
  );
  await commandRouterClient.mountDirectory(request);
}

/** @internal @hidden */
export async function unmountImageViaTcr(
  impl: SandboxImpl,
  path: string,
): Promise<void> {
  const [taskId, commandRouterClient] = await impl.getCommandRouter();

  const pathBytes = new TextEncoder().encode(path);
  const request = TaskUnmountDirectoryRequest.create({
    taskId,
    path: pathBytes,
  });
  await commandRouterClient.unmountDirectory(request);
}

/**
 * Reload one container's Volumes; empty `containerId` targets the main container.
 *
 * @internal
 * @hidden
 */
export async function reloadVolumesViaTcr(
  impl: SandboxImpl,
  containerId: string,
  params: SandboxReloadVolumesParams | undefined,
): Promise<void> {
  if (params?.timeoutMs !== undefined && params.timeoutMs < 0) {
    throw new InvalidError("`timeoutMs` must not be negative");
  }
  // Both undefined and 0 fall back to the default, matching the Go
  // `Timeout time.Duration` zero-value convention.
  const timeoutMs = params?.timeoutMs || 55000;
  const [taskId, commandRouterClient] = await impl.getCommandRouter();
  await commandRouterClient.reloadVolumes(
    TaskReloadVolumesRequest.create({ taskId, containerId }),
    { timeoutMs },
  );
}

/** @internal @hidden */
export async function execViaTcr(
  impl: SandboxImpl,
  command: string[],
  params: SandboxExecParams | undefined,
  containerId?: string,
): Promise<ContainerProcess> {
  validateExecArgs(command);

  // Locally-created Secrets (fromObject) are inlined into the request's
  // secret sources, so only the remaining Secrets need hydrating. This avoids
  // a SecretGetOrCreate round-trip for env-dict Secrets.
  validateEnvVarKeys(params?.env ?? {});
  await hydrateSecrets(impl.client, resolvableSecrets(params?.secrets ?? []));

  const [taskId, commandRouterClient] = await impl.getCommandRouter();

  const execId = uuidv4();
  const request = buildTaskExecStartRequestProto(
    taskId,
    execId,
    command,
    params,
    containerId,
  );

  await commandRouterClient.execStart(request);

  impl.client.logger.debug(
    "Created ContainerProcess",
    "exec_id",
    execId,
    "sandbox_id",
    impl.sandboxId,
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

/**
 * Create a sidecar container over the Sandbox's task command router connection.
 *
 * @internal
 * @hidden
 */
export async function createSidecarViaTcr(
  impl: SandboxImpl,
  request: SidecarCreateRequest,
): Promise<SidecarCreateResult> {
  const [taskId, client] = await impl.getCommandRouter();
  const resp = await client.containerCreate(
    TaskContainerCreateRequest.create({
      taskId,
      containerName: request.containerName,
      imageId: request.imageId,
      args: request.command,
      workdir: request.workdir ?? "",
      secretSources: request.secretSources,
      volumeMounts: request.volumeMounts,
      networkAccess: request.networkAccess,
      ptyInfo: request.ptyInfo,
      memoryReserveConsumeMib: request.memoryReserveConsumeMiB,
    }),
  );
  return { containerId: resp.containerId, containerName: resp.containerName };
}
