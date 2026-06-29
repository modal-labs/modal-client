import { setTimeout } from "timers/promises";
import {
  CallOptions,
  Client,
  ClientError,
  ChannelCredentials,
  createChannel,
  createClientFactory,
  Metadata,
  Status,
} from "nice-grpc";
import {
  TaskCommandRouterDefinition,
  TaskContainerCreateRequest,
  TaskContainerCreateResponse,
  TaskContainerGetRequest,
  TaskContainerGetResponse,
  TaskContainerListRequest,
  TaskContainerListResponse,
  TaskContainerTerminateRequest,
  TaskContainerWaitRequest,
  TaskContainerWaitResponse,
  TaskExecPollRequest,
  TaskExecPollResponse,
  TaskExecStartRequest,
  TaskExecStartResponse,
  TaskExecStdinWriteRequest,
  TaskExecStdinWriteResponse,
  TaskExecStdioFileDescriptor,
  TaskExecStdioReadRequest,
  TaskExecStdioReadResponse,
  TaskExecWaitRequest,
  TaskExecWaitResponse,
  TaskMountDirectoryRequest,
  TaskSnapshotDirectoryRequest,
  TaskSnapshotDirectoryResponse,
  TaskSnapshotFilesystemRequest,
  TaskSnapshotFilesystemResponse,
  TaskUnmountDirectoryRequest,
  TaskSetNetworkAccessRequest,
  SandboxWaitUntilReadyTcrRequest,
  SandboxWaitUntilReadyTcrResponse,
} from "../proto/modal_proto/task_command_router";
import {
  TaskGetCommandRouterAccessRequest,
  FileDescriptor,
  SandboxGetCommandRouterAccessRequest,
} from "../proto/modal_proto/api";
import type { ModalGrpcClient } from "./client";
import { timeoutMiddleware, type TimeoutOptions } from "./client";
import type { Logger } from "./logger";
import type { Profile } from "./config";
import { isLocalhost } from "./config";
import { ClientClosedError, TimeoutError } from "./errors";

type TaskCommandRouterClient = Client<typeof TaskCommandRouterDefinition>;

type CommandRouterAccess = {
  url: string;
  jwt: string;
};

export function parseJwtExpiration(
  jwtToken: string,
  logger: Logger,
): number | null {
  try {
    const parts = jwtToken.split(".");
    if (parts.length !== 3) {
      return null;
    }
    const payloadB64 = parts[1];
    const padding = "=".repeat((4 - (payloadB64.length % 4)) % 4);
    const payloadJson = Buffer.from(payloadB64 + padding, "base64").toString(
      "utf8",
    );
    const payload = JSON.parse(payloadJson);
    const exp = payload.exp;
    if (typeof exp === "number") {
      return exp;
    }
  } catch (e) {
    // Avoid raising on malformed tokens; fall back to server-driven refresh logic.
    logger.warn("Failed to parse JWT expiration", "error", e);
  }
  return null;
}

class RetryDeadlineExceededError extends Error {
  constructor() {
    super("Deadline exceeded");
  }
}

export async function callWithRetriesOnTransientErrors<T>(
  func: () => Promise<T>,
  baseDelayMs: number = 10,
  delayFactor: number = 2,
  maxRetries: number | null = 10,
  deadlineMs: number | null = null,
  isClosed?: () => boolean,
  /**
   * gRPC status codes to exclude from retry logic even if they would
   * otherwise be retryable. Use this to let errors like DEADLINE_EXCEEDED
   * propagate immediately when the caller specified their own timeout.
   */
  excludeStatusCodes: Status[] = [],
): Promise<T> {
  let delayMs = baseDelayMs;
  let numRetries = 0;

  const retryableStatusCodes = new Set([
    Status.DEADLINE_EXCEEDED,
    Status.UNAVAILABLE,
    Status.CANCELLED,
    Status.INTERNAL,
    Status.UNKNOWN,
  ]);
  const excluded = new Set(excludeStatusCodes);

  while (true) {
    if (deadlineMs !== null && Date.now() >= deadlineMs) {
      throw new RetryDeadlineExceededError();
    }

    try {
      return await func();
    } catch (err) {
      if (
        err instanceof ClientError &&
        err.code === Status.CANCELLED &&
        isClosed?.()
      ) {
        throw new ClientClosedError();
      }
      if (
        err instanceof ClientError &&
        retryableStatusCodes.has(err.code) &&
        !excluded.has(err.code) &&
        (maxRetries === null || numRetries < maxRetries)
      ) {
        // Clamp the backoff to the remaining deadline budget so we don't
        // sleep past it. If the budget is already exhausted, the next
        // iteration's top-of-loop check throws RetryDeadlineExceededError
        // with Date.now() actually past the deadline — letting callers
        // translate consistently against the wall clock.
        let sleepMs = delayMs;
        if (deadlineMs !== null) {
          sleepMs = Math.min(sleepMs, deadlineMs - Date.now());
        }
        if (sleepMs < 0) sleepMs = 0;
        await setTimeout(sleepMs);
        delayMs *= delayFactor;
        numRetries++;
      } else {
        throw err;
      }
    }
  }
}

/** @ignore */
export class TaskCommandRouterClientImpl {
  private stub: TaskCommandRouterClient;
  private channel: ReturnType<typeof createChannel>;
  private serverClient: ModalGrpcClient;
  private taskId: string;
  private sandboxId: string;
  private isV2: boolean;
  private serverUrl: string;
  private jwt: string;
  private jwtExp: number | null;
  private jwtRefreshLock: Promise<void> = Promise.resolve();
  private logger: Logger;
  private closed: boolean = false;

  static async tryInit(
    serverClient: ModalGrpcClient,
    taskId: string,
    sandboxId: string,
    isV2: boolean,
    logger: Logger,
    profile: Profile,
  ): Promise<TaskCommandRouterClientImpl | null> {
    let resp: CommandRouterAccess;
    try {
      resp = await getCommandRouterAccess(
        serverClient,
        taskId,
        sandboxId,
        isV2,
      );
    } catch (err) {
      if (
        err instanceof ClientError &&
        err.code === Status.FAILED_PRECONDITION
      ) {
        logger.debug(
          "Command router access is not enabled for task",
          "task_id",
          taskId,
        );
        return null;
      }
      throw err;
    }

    logger.debug(
      "Using command router access for task",
      "task_id",
      taskId,
      "url",
      resp.url,
    );

    const url = new URL(resp.url);
    if (url.protocol !== "https:") {
      throw new Error(`Task router URL must be https, got: ${resp.url}`);
    }

    const host = url.hostname;
    const port = url.port ? parseInt(url.port) : 443;
    const serverUrl = `${host}:${port}`;
    const channelConfig = {
      "grpc.max_receive_message_length": 100 * 1024 * 1024,
      "grpc.max_send_message_length": 100 * 1024 * 1024,
      "grpc-node.flow_control_window": 64 * 1024 * 1024,
      "grpc.keepalive_time_ms": 30000,
      "grpc.keepalive_timeout_ms": 10000,
      "grpc.keepalive_permit_without_calls": 1,
    };

    let channel;
    if (isLocalhost(profile)) {
      logger.warn(
        "Using insecure TLS (skip certificate verification) for task command router",
      );
      channel = createChannel(
        serverUrl,
        ChannelCredentials.createInsecure(),
        channelConfig,
      );
    } else {
      channel = createChannel(
        serverUrl,
        ChannelCredentials.createSsl(),
        channelConfig,
      );
    }

    const client = new TaskCommandRouterClientImpl(
      serverClient,
      taskId,
      sandboxId,
      isV2,
      resp.url,
      resp.jwt,
      channel,
      logger,
    );

    logger.debug(
      "Successfully initialized command router client",
      "task_id",
      taskId,
    );

    return client;
  }

  private constructor(
    serverClient: ModalGrpcClient,
    taskId: string,
    sandboxId: string,
    isV2: boolean,
    serverUrl: string,
    jwt: string,
    channel: ReturnType<typeof createChannel>,
    logger: Logger,
  ) {
    this.serverClient = serverClient;
    this.taskId = taskId;
    this.sandboxId = sandboxId;
    this.isV2 = isV2;
    this.serverUrl = serverUrl;
    this.jwt = jwt;
    this.jwtExp = parseJwtExpiration(jwt, logger);
    this.logger = logger;
    this.channel = channel;

    // Capture 'this' so the auth middleware can access the current JWT after refreshes.
    // We need to alias 'this' because generator functions cannot be arrow functions.
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;

    const factory = createClientFactory()
      .use(timeoutMiddleware)
      .use(async function* authMiddleware(call, options: CallOptions) {
        options.metadata ??= new Metadata();
        options.metadata.set("authorization", `Bearer ${self.jwt}`);
        return yield* call.next(call.request, options);
      });

    this.stub = factory.create(TaskCommandRouterDefinition, channel);
  }

  close(): void {
    if (this.closed) {
      return;
    }

    this.closed = true;
    this.channel.close();
  }

  /** Run a unary RPC against the command router with the default retry policy. */
  private async callUnary<T>(fn: () => Promise<T>): Promise<T> {
    return await callWithRetriesOnTransientErrors(
      () => this.callWithAuthRetry(fn),
      10, // baseDelayMs
      2, // delayFactor
      10, // maxRetries
      null, // no overall deadline
      () => this.closed,
    );
  }

  async execStart(
    request: TaskExecStartRequest,
  ): Promise<TaskExecStartResponse> {
    return await this.callUnary(() => this.stub.taskExecStart(request));
  }

  async containerCreate(
    request: TaskContainerCreateRequest,
  ): Promise<TaskContainerCreateResponse> {
    return await this.callUnary(() => this.stub.taskContainerCreate(request));
  }

  async containerGet(
    request: TaskContainerGetRequest,
  ): Promise<TaskContainerGetResponse> {
    return await this.callUnary(() => this.stub.taskContainerGet(request));
  }

  async containerList(
    request: TaskContainerListRequest,
  ): Promise<TaskContainerListResponse> {
    return await this.callUnary(() => this.stub.taskContainerList(request));
  }

  async containerTerminate(
    request: TaskContainerTerminateRequest,
  ): Promise<void> {
    await this.callUnary(() => this.stub.taskContainerTerminate(request));
  }

  async containerWait(
    request: TaskContainerWaitRequest,
  ): Promise<TaskContainerWaitResponse> {
    return await this.callUnary(() => this.stub.taskContainerWait(request));
  }

  async *execStdioRead(
    taskId: string,
    execId: string,
    fileDescriptor: FileDescriptor,
    deadline: number | null = null,
  ): AsyncGenerator<TaskExecStdioReadResponse> {
    let srFd: TaskExecStdioFileDescriptor;
    if (fileDescriptor === FileDescriptor.FILE_DESCRIPTOR_STDOUT) {
      srFd = TaskExecStdioFileDescriptor.TASK_EXEC_STDIO_FILE_DESCRIPTOR_STDOUT;
    } else if (fileDescriptor === FileDescriptor.FILE_DESCRIPTOR_STDERR) {
      srFd = TaskExecStdioFileDescriptor.TASK_EXEC_STDIO_FILE_DESCRIPTOR_STDERR;
    } else if (
      fileDescriptor === FileDescriptor.FILE_DESCRIPTOR_INFO ||
      fileDescriptor === FileDescriptor.FILE_DESCRIPTOR_UNSPECIFIED
    ) {
      throw new Error(`Unsupported file descriptor: ${fileDescriptor}`);
    } else {
      throw new Error(`Invalid file descriptor: ${fileDescriptor}`);
    }

    yield* this.streamStdio(taskId, execId, srFd, deadline);
  }

  async execStdinWrite(
    taskId: string,
    execId: string,
    offset: number,
    data: Uint8Array,
    eof: boolean,
  ): Promise<TaskExecStdinWriteResponse> {
    const request = TaskExecStdinWriteRequest.create({
      taskId,
      execId,
      offset,
      data,
      eof,
    });
    return await this.callUnary(() => this.stub.taskExecStdinWrite(request));
  }

  async execPoll(
    taskId: string,
    execId: string,
    deadline: number | null = null,
  ): Promise<TaskExecPollResponse> {
    const request = TaskExecPollRequest.create({ taskId, execId });

    // The timeout here is really a backstop in the event of a hang contacting
    // the command router. Poll should usually be instantaneous.
    if (deadline && deadline <= Date.now()) {
      throw new Error(`Deadline exceeded while polling for exec ${execId}`);
    }

    try {
      return await callWithRetriesOnTransientErrors(
        () => this.callWithAuthRetry(() => this.stub.taskExecPoll(request)),
        10, // baseDelayMs
        2, // delayFactor
        10, // maxRetries
        deadline, // Enforce overall deadline.
        () => this.closed,
      );
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.DEADLINE_EXCEEDED) {
        throw new Error(`Deadline exceeded while polling for exec ${execId}`);
      }
      throw err;
    }
  }

  async execWait(
    taskId: string,
    execId: string,
    deadline: number | null = null,
  ): Promise<TaskExecWaitResponse> {
    const request = TaskExecWaitRequest.create({ taskId, execId });

    if (deadline && deadline <= Date.now()) {
      throw new Error(`Deadline exceeded while waiting for exec ${execId}`);
    }

    try {
      return await callWithRetriesOnTransientErrors(
        () =>
          this.callWithAuthRetry(() =>
            this.stub.taskExecWait(request, {
              timeoutMs: 60_000,
            } as CallOptions & TimeoutOptions),
          ),
        1000, // Retry after 1s since total time is expected to be long.
        1, // Fixed delay.
        null, // Retry forever.
        deadline, // Enforce overall deadline.
        () => this.closed,
      );
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.DEADLINE_EXCEEDED) {
        throw new Error(`Deadline exceeded while waiting for exec ${execId}`);
      }
      throw err;
    }
  }

  async mountDirectory(request: TaskMountDirectoryRequest): Promise<void> {
    await this.callUnary(() => this.stub.taskMountDirectory(request));
  }

  async snapshotDirectory(
    request: TaskSnapshotDirectoryRequest,
    options?: TimeoutOptions,
  ): Promise<TaskSnapshotDirectoryResponse> {
    // Mirrors snapshotFilesystem's deadline handling. `timeoutMs` is the
    // overall budget across all retry attempts; any error observed at or
    // after the deadline is translated into a TimeoutError. Errors
    // observed *before* the deadline (including caller-driven aborts)
    // propagate unchanged.
    const overallDeadlineMs =
      options?.timeoutMs !== undefined ? Date.now() + options.timeoutMs : null;
    try {
      return await callWithRetriesOnTransientErrors(
        () =>
          this.callWithAuthRetry(() => {
            const remainingMs =
              overallDeadlineMs !== null
                ? Math.max(1, overallDeadlineMs - Date.now())
                : options?.timeoutMs;
            return this.stub.taskSnapshotDirectory(request, {
              ...options,
              timeoutMs: remainingMs,
            } as CallOptions & TimeoutOptions);
          }),
        10,
        2,
        10,
        overallDeadlineMs,
        () => this.closed,
        [Status.DEADLINE_EXCEEDED, Status.CANCELLED],
      );
    } catch (err) {
      if (overallDeadlineMs !== null && Date.now() >= overallDeadlineMs) {
        throw new TimeoutError("Timeout expired");
      }
      throw err;
    }
  }

  async snapshotFilesystem(
    request: TaskSnapshotFilesystemRequest,
    options?: TimeoutOptions,
  ): Promise<TaskSnapshotFilesystemResponse> {
    // TaskSnapshotFilesystem has a caller-controllable timeout. We treat
    // it as the overall budget across all retry attempts: each attempt
    // receives the *remaining* budget as its per-call gRPC deadline, and
    // retries are aborted once the deadline elapses — otherwise a
    // transient retryable error would grant another fresh full window and
    // the caller's intent would be violated. DEADLINE_EXCEEDED / CANCELLED
    // are excluded from the retry set so another attempt cannot reset the
    // deadline.
    //
    // Any error observed at or after the deadline is translated into a
    // TimeoutError. Errors observed *before* the deadline are propagated
    // unchanged — including a caller-driven AbortSignal cancellation
    // (which nice-grpc surfaces as Status.CANCELLED), so callers see
    // their cancel rather than a misleading timeout.
    const overallDeadlineMs =
      options?.timeoutMs !== undefined ? Date.now() + options.timeoutMs : null;
    try {
      return await callWithRetriesOnTransientErrors(
        () =>
          this.callWithAuthRetry(() => {
            // At least 1ms so the timeoutMiddleware's `!options.timeoutMs`
            // truthy check doesn't skip the deadline entirely; if the
            // budget really is exhausted the outer retry loop's pre-check
            // will short-circuit on the next iteration.
            const remainingMs =
              overallDeadlineMs !== null
                ? Math.max(1, overallDeadlineMs - Date.now())
                : options?.timeoutMs;
            return this.stub.taskSnapshotFilesystem(request, {
              ...options,
              timeoutMs: remainingMs,
            } as CallOptions & TimeoutOptions);
          }),
        10,
        2,
        10,
        overallDeadlineMs,
        () => this.closed,
        [Status.DEADLINE_EXCEEDED, Status.CANCELLED],
      );
    } catch (err) {
      if (overallDeadlineMs !== null && Date.now() >= overallDeadlineMs) {
        throw new TimeoutError("Timeout expired");
      }
      throw err;
    }
  }

  async unmountDirectory(request: TaskUnmountDirectoryRequest): Promise<void> {
    await this.callUnary(() => this.stub.taskUnmountDirectory(request));
  }

  async setNetworkAccess(request: TaskSetNetworkAccessRequest): Promise<void> {
    await this.callUnary(() => this.stub.taskSetNetworkAccess(request));
  }

  async sandboxWaitUntilReady(
    taskId: string,
    timeoutMs: number,
  ): Promise<SandboxWaitUntilReadyTcrResponse> {
    const deadlineMs = Date.now() + timeoutMs;
    try {
      return await callWithRetriesOnTransientErrors(
        () =>
          this.callWithAuthRetry(() => {
            const remainingMs = Math.max(1, deadlineMs - Date.now());
            const request = SandboxWaitUntilReadyTcrRequest.create({
              taskId,
              timeout: remainingMs / 1000,
            });
            return this.stub.sandboxWaitUntilReady(request, {
              timeoutMs: remainingMs,
            } as CallOptions & TimeoutOptions);
          }),
        10,
        2,
        10,
        deadlineMs,
        () => this.closed,
      );
    } catch (err) {
      if (err instanceof RetryDeadlineExceededError) {
        throw new TimeoutError("Timeout expired");
      }
      throw err;
    }
  }

  private async refreshJwt(): Promise<void> {
    let error: unknown;

    this.jwtRefreshLock = this.jwtRefreshLock.then(async () => {
      if (this.closed) {
        return;
      }

      // If the current JWT expiration is already far enough in the future, don't refresh.
      if (this.jwtExp !== null && this.jwtExp - Date.now() / 1000 > 30) {
        // This can happen if multiple concurrent requests to the task command router
        // get UNAUTHENTICATED errors and all refresh at the same time - one of them
        // will win and the others will not refresh.
        this.logger.debug(
          "Skipping JWT refresh because expiration is far enough in the future",
          "task_id",
          this.taskId,
        );
        return;
      }

      try {
        const resp = await getCommandRouterAccess(
          this.serverClient,
          this.taskId,
          this.sandboxId,
          this.isV2,
        );

        if (resp.url !== this.serverUrl) {
          this.logger.warn("Task router URL changed during session");
        }

        this.jwt = resp.jwt;
        this.jwtExp = parseJwtExpiration(resp.jwt, this.logger);
      } catch (err) {
        // Capture the error but don't reject the promise chain.
        // This ensures the chain remains usable for future refresh attempts.
        error = err;
      }
    });

    await this.jwtRefreshLock;

    if (error) {
      throw error;
    }
  }

  private async callWithAuthRetry<T>(func: () => Promise<T>): Promise<T> {
    try {
      return await func();
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.UNAUTHENTICATED) {
        await this.refreshJwt();
        return await func();
      }
      throw err;
    }
  }

  private async *streamStdio(
    taskId: string,
    execId: string,
    fileDescriptor: TaskExecStdioFileDescriptor,
    deadline: number | null,
  ): AsyncGenerator<TaskExecStdioReadResponse> {
    let offset = 0;
    let delayMs = 10;
    const delayFactor = 2;
    let numRetriesRemaining = 10;
    // Flag to prevent infinite auth retries in the event that the JWT
    // refresh yields an invalid JWT somehow or that the JWT is otherwise invalid.
    let didAuthRetry = false;

    const retryableStatusCodes = new Set([
      Status.DEADLINE_EXCEEDED,
      Status.UNAVAILABLE,
      Status.CANCELLED,
      Status.INTERNAL,
      Status.UNKNOWN,
    ]);

    while (true) {
      try {
        const timeoutMs =
          deadline !== null ? Math.max(0, deadline - Date.now()) : undefined;

        const request = TaskExecStdioReadRequest.create({
          taskId,
          execId,
          offset,
          fileDescriptor,
        });

        const stream = this.stub.taskExecStdioRead(request, {
          timeoutMs,
        } as CallOptions & TimeoutOptions);

        try {
          for await (const item of stream) {
            // We successfully authenticated after a JWT refresh, reset the auth retry flag.
            if (didAuthRetry) {
              didAuthRetry = false;
            }
            delayMs = 10;
            offset += item.data.length;
            yield item;
          }
          return;
        } catch (err) {
          if (
            err instanceof ClientError &&
            err.code === Status.UNAUTHENTICATED &&
            !didAuthRetry
          ) {
            await this.refreshJwt();
            // Mark that we've retried authentication for this streaming attempt, to
            // prevent subsequent retries.
            didAuthRetry = true;
            continue;
          }
          throw err;
        }
      } catch (err) {
        if (
          err instanceof ClientError &&
          err.code === Status.CANCELLED &&
          this.closed
        ) {
          throw new ClientClosedError();
        }
        if (
          err instanceof ClientError &&
          retryableStatusCodes.has(err.code) &&
          numRetriesRemaining > 0
        ) {
          if (deadline && deadline - Date.now() <= delayMs) {
            throw new Error(
              `Deadline exceeded while streaming stdio for exec ${execId}`,
            );
          }

          this.logger.debug(
            "Retrying stdio read with delay",
            "delay_ms",
            delayMs,
            "error",
            err,
          );
          await setTimeout(delayMs);
          delayMs *= delayFactor;
          numRetriesRemaining--;
        } else {
          throw err;
        }
      }
    }
  }
}

async function getCommandRouterAccess(
  serverClient: ModalGrpcClient,
  taskId: string,
  sandboxId: string,
  isV2: boolean,
): Promise<CommandRouterAccess> {
  if (isV2) {
    const resp = await serverClient.sandboxGetCommandRouterAccess(
      SandboxGetCommandRouterAccessRequest.create({ sandboxId }),
    );
    return { url: resp.url, jwt: resp.jwt };
  }
  const resp = await serverClient.taskGetCommandRouterAccess(
    TaskGetCommandRouterAccessRequest.create({ taskId }),
  );
  return { url: resp.url, jwt: resp.jwt };
}
