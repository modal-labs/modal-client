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

// How long operations that need a running task wait for one.
const GET_TASK_ID_TIMEOUT_MS = 5 * 60 * 1000;
const SB_LOGS_TASK_ID_TIMEOUT_SECONDS = 0.5;

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
 * Shared machinery for backends whose Sandbox is a single task reached through
 * the task command router: task ID lookup, command router connection, and the
 * operations that only differ between V1 and V2 in which control plane RPC
 * they call.
 *
 * @internal
 * @hidden
 */
export abstract class SandboxImplBase implements SandboxImpl {
  readonly client: ModalClient;
  readonly sandboxId: string;

  protected appId: string | undefined;
  protected taskId: string | undefined;
  protected cachedTunnels: Record<number, Tunnel> | undefined;
  #commandRouterClient: TaskCommandRouterClientImpl | undefined;
  #commandRouterClientPromise: Promise<TaskCommandRouterClientImpl> | undefined;
  #initCommandRouterAccess: CommandRouterAccess | undefined;
  #closed = false;

  constructor(
    client: ModalClient,
    sandboxId: string,
    params: {
      taskId?: string;
      tunnels?: Record<number, Tunnel>;
      initCommandRouterAccess?: CommandRouterAccess;
    } = {},
  ) {
    this.client = client;
    this.sandboxId = sandboxId;
    this.taskId = params.taskId || undefined;
    this.cachedTunnels = params.tunnels;
    this.#initCommandRouterAccess = params.initCommandRouterAccess;
  }

  /** Whether command router connections authenticate as a V2 Sandbox. */
  protected abstract readonly isV2: boolean;

  protected abstract rpcWait(timeoutSecs: number): Promise<SandboxWaitResponse>;
  protected abstract rpcGetTaskId(
    timeoutSecs: number | undefined,
    options?: CallOptions,
  ): Promise<SandboxGetTaskIdResponse>;
  protected abstract rpcGetTunnels(
    timeoutMs: number,
  ): Promise<SandboxGetTunnelsResponse>;
  protected abstract rpcGetExitSnapshot(
    timeoutSecs: number,
    options?: CallOptions & TimeoutOptions & RetryOptions,
  ): Promise<SandboxGetExitSnapshotResponse>;
  protected abstract rpcTerminate(): Promise<void>;

  abstract stdin(): ModalWriteStream<string>;
  abstract outputStream(
    fileDescriptor: FileDescriptor,
    ensureAttached: () => void,
  ): ModalReadStream<string>;
  abstract getTags(): Promise<Record<string, string>>;
  abstract setTags(tags: Record<string, string>): Promise<void>;
  abstract setName(name: string): Promise<void>;
  abstract createConnectToken(
    port: number,
    userMetadata: string | undefined,
  ): Promise<SandboxCreateConnectCredentials>;
  abstract snapshotMemory(): Promise<SandboxSnapshot>;
  abstract validateSidecarCreate(options: SidecarCreateOptions): void;
  abstract createSidecar(
    request: SidecarCreateRequest,
  ): Promise<SidecarCreateResult>;

  async terminate(): Promise<void> {
    await this.rpcTerminate();
  }

  close(): void {
    this.#closed = true;
    this.#commandRouterClient?.closeWhenIdle();
    this.#commandRouterClient = undefined;
    this.#commandRouterClientPromise = undefined;
  }

  async wait(): Promise<number> {
    while (true) {
      const resp = await this.rpcWait(10);
      if (resp.result) {
        const returnCode = getReturnCode(resp.result)!;
        this.client.logger.debug(
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

  async poll(): Promise<number | null> {
    const resp = await this.rpcWait(0);
    return getReturnCode(resp.result);
  }

  async waitForScheduling(timeoutMs: number): Promise<void> {
    try {
      await this.waitForTaskId(timeoutMs, true);
    } catch (err) {
      if (err instanceof ConflictError) {
        throw err;
      }
      try {
        await this.rpcTerminate();
      } catch (terminateErr) {
        this.client.logger.debug(
          "Failed to terminate unscheduled Sandbox",
          "sandbox_id",
          this.sandboxId,
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

  async tunnels(timeoutMs: number): Promise<Record<number, Tunnel>> {
    if (this.cachedTunnels) {
      return this.cachedTunnels;
    }

    const resp = await this.rpcGetTunnels(timeoutMs);

    if (
      resp.result?.status === GenericResult_GenericStatus.GENERIC_STATUS_TIMEOUT
    ) {
      throw new SandboxTimeoutError();
    }

    const tunnels: Record<number, Tunnel> = {};
    for (const t of resp.tunnels) {
      tunnels[t.containerPort] = new Tunnel(
        t.host,
        t.port,
        t.unencryptedHost,
        t.unencryptedPort,
      );
    }
    this.cachedTunnels = tunnels;
    return tunnels;
  }

  async getAppId(): Promise<string> {
    if (this.appId !== undefined) {
      return this.appId;
    }

    const resp = await this.rpcWait(0);
    const appId = resp.metadata?.appId;
    if (!appId) {
      throw new ExecutionError(
        "Sandbox app ID should have been set during hydration",
      );
    }
    this.appId = appId;
    return appId;
  }

  async resolveTaskIdForLogs(): Promise<string> {
    if (this.taskId !== undefined) {
      return this.taskId;
    }

    let resp;
    try {
      resp = await this.rpcGetTaskId(SB_LOGS_TASK_ID_TIMEOUT_SECONDS);
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
    this.taskId = resp.taskId;
    return this.taskId;
  }

  async getExitSnapshot(
    params: SandboxExperimentalGetExitSnapshotParams | undefined,
  ): Promise<Image> {
    const timeoutMs = params?.timeoutMs;
    if (timeoutMs !== undefined && timeoutMs < 0) {
      throw new InvalidError(
        `timeoutMs must be non-negative, got ${timeoutMs}`,
      );
    }

    const deadline =
      timeoutMs === undefined ? undefined : Date.now() + timeoutMs;
    const timeoutMessage = `timed out waiting for exit snapshot for Sandbox ${this.sandboxId}`;
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
        resp = await this.rpcGetExitSnapshot(requestTimeoutSecs, {
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
          let throttleDelayMs = Math.max(
            retryPolicy.retryAfterSecs * 1000,
            100,
          );
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
            consecutivePollFailures >=
            EXIT_SNAPSHOT_MAX_CONSECUTIVE_POLL_FAILURES
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
        return new Image(this.client, resp.success.imageId, "");
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
        throw new ExecutionError(
          "Exit snapshot response is missing an outcome",
        );
      }
      if (deadline !== undefined && Date.now() >= deadline) {
        throw new TimeoutError(timeoutMessage);
      }
    }
  }

  async getCommandRouter(
    signal?: AbortSignal,
  ): Promise<[string, TaskCommandRouterClientImpl]> {
    if (this.#closed) {
      throw new ClientClosedError();
    }
    const taskId = await this.getTaskId(signal);
    const client = await this.#getOrCreateCommandRouterClient(taskId);
    return [taskId, client];
  }

  protected async getTaskId(signal?: AbortSignal): Promise<string> {
    return await this.waitForTaskId(GET_TASK_ID_TIMEOUT_MS, false, signal);
  }

  /**
   * Polls until the Sandbox has been assigned a task. Throws a ConflictError
   * if the Sandbox finished before that happened and a TimeoutError once
   * timeoutMs elapses. With retryTransient, lookups that fail with a
   * transient error keep polling until the deadline instead of failing.
   */
  protected async waitForTaskId(
    timeoutMs: number,
    retryTransient: boolean,
    signal?: AbortSignal,
  ): Promise<string> {
    if (this.taskId !== undefined) {
      return this.taskId;
    }
    const deadline = Date.now() + timeoutMs;
    const timedOut = () =>
      new TimeoutError(
        `Timed out waiting for task ID for Sandbox ${this.sandboxId}`,
      );
    const pollIntervalMs = 500;
    const noTimeLeft = () => deadline - Date.now() <= pollIntervalMs;
    while (true) {
      let resp;
      try {
        resp = await this.rpcGetTaskId(undefined, { signal });
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
        this.taskId = resp.taskId;
        return this.taskId;
      }
      if (resp.taskResult) {
        throw new ConflictError(
          resp.taskResult.exception ||
            `Sandbox ${this.sandboxId} has already finished`,
        );
      }
      if (noTimeLeft()) {
        throw timedOut();
      }
      await setTimeout(pollIntervalMs, undefined, { signal });
    }
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

    // Consume the seeded access, so if connecting with it fails the retry below
    // falls back to the authoritative RPC instead of re-trying the same
    // credentials forever.
    const access = this.#initCommandRouterAccess;
    this.#initCommandRouterAccess = undefined;

    const promise = (async () => {
      const client = await TaskCommandRouterClientImpl.tryInit(
        this.client.cpClient,
        taskId,
        this.sandboxId,
        this.isV2,
        access,
        this.client.logger,
        this.client.profile,
      );
      if (!client) {
        throw new Error(
          "Command router access is not available for this sandbox",
        );
      }
      if (this.#closed) {
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

  async waitUntilReady(timeoutMs: number): Promise<void> {
    const [taskId, commandRouterClient] = await this.getCommandRouter();
    await commandRouterClient.sandboxWaitUntilReady(taskId, timeoutMs);
  }

  async updateNetworkPolicy(
    params: SandboxUpdateNetworkPolicyParams,
  ): Promise<void> {
    const [taskId, commandRouterClient] = await this.getCommandRouter();

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

  async updateOutboundPolicy(
    outboundPolicy: ExperimentalOutboundPolicy,
  ): Promise<void> {
    const [, [taskId, commandRouterClient]] = await Promise.all([
      hydrateSecrets(this.client, outboundPolicy._secrets()),
      this.getCommandRouter(),
    ]);

    const request = TaskSetExperimentalOutboundPolicyRequest.create({
      taskId,
      experimentalOutboundPolicy: outboundPolicy._toProto(),
    });
    await commandRouterClient.setOutboundPolicy(request);
  }

  async snapshotFilesystem(
    containerId: string,
    params: SandboxSnapshotFilesystemParams | undefined,
  ): Promise<Image> {
    const wireTtlSeconds = resolveTtlSeconds(params?.ttlMs);
    // Treat both undefined and 0 as "use default", matching the Go
    // `Timeout time.Duration` zero-value convention on SandboxSnapshotFilesystemParams.
    const timeoutMs = params?.timeoutMs || 55000;
    const [taskId, commandRouterClient] = await this.getCommandRouter();

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

    return new Image(this.client, response.imageId, "");
  }

  async snapshotDirectory(
    path: string,
    params: SandboxSnapshotDirectoryParams | undefined,
  ): Promise<Image> {
    const wireTtlSeconds = resolveTtlSeconds(params?.ttlMs);
    // Treat both undefined and 0 as "use default", matching the Go
    // `Timeout time.Duration` zero-value convention on SandboxSnapshotDirectoryParams.
    const timeoutMs = params?.timeoutMs || 55000;
    const [taskId, commandRouterClient] = await this.getCommandRouter();

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

    return new Image(this.client, response.imageId, "");
  }

  async mountImage(
    path: string,
    image: Image | undefined,
    params: SandboxMountImageParams | undefined,
  ): Promise<void> {
    const [taskId, commandRouterClient] = await this.getCommandRouter();

    const imageId = resolveMountImageId(image);
    const request = buildTaskMountDirectoryRequestProto(
      taskId,
      path,
      imageId,
      params,
    );
    await commandRouterClient.mountDirectory(request);
  }

  async unmountImage(path: string): Promise<void> {
    const [taskId, commandRouterClient] = await this.getCommandRouter();

    const pathBytes = new TextEncoder().encode(path);
    const request = TaskUnmountDirectoryRequest.create({
      taskId,
      path: pathBytes,
    });
    await commandRouterClient.unmountDirectory(request);
  }

  // Reload one container's Volumes; empty `containerId` targets the main container.
  async reloadVolumes(
    containerId: string,
    params: SandboxReloadVolumesParams | undefined,
  ): Promise<void> {
    if (params?.timeoutMs !== undefined && params.timeoutMs < 0) {
      throw new InvalidError("`timeoutMs` must not be negative");
    }
    // Both undefined and 0 fall back to the default, matching the Go
    // `Timeout time.Duration` zero-value convention.
    const timeoutMs = params?.timeoutMs || 55000;
    const [taskId, commandRouterClient] = await this.getCommandRouter();
    await commandRouterClient.reloadVolumes(
      TaskReloadVolumesRequest.create({ taskId, containerId }),
      { timeoutMs },
    );
  }

  async exec(
    command: string[],
    params: SandboxExecParams | undefined,
    containerId?: string,
  ): Promise<ContainerProcess> {
    validateExecArgs(command);

    // Locally-created Secrets (fromObject) are inlined into the request's
    // secret sources, so only the remaining Secrets need hydrating. This avoids
    // a SecretGetOrCreate round-trip for env-dict Secrets.
    validateEnvVarKeys(params?.env ?? {});
    await hydrateSecrets(this.client, resolvableSecrets(params?.secrets ?? []));

    const [taskId, commandRouterClient] = await this.getCommandRouter();

    const execId = uuidv4();
    const request = buildTaskExecStartRequestProto(
      taskId,
      execId,
      command,
      params,
      containerId,
    );

    await commandRouterClient.execStart(request);

    this.client.logger.debug(
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

  /** Create a sidecar container over the Sandbox's task command router connection. */
  protected async createSidecarViaCommandRouter(
    request: SidecarCreateRequest,
  ): Promise<SidecarCreateResult> {
    const [taskId, client] = await this.getCommandRouter();
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
}
