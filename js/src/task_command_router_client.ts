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
  TaskExecStdinStatusRequest,
  TaskExecStdinStatusResponse,
  TaskExecStdinWriteRequest,
  TaskExecStdinWriteResponse,
  TaskExecStdinWriteStreamRequest,
  TaskExecStdioFileDescriptor,
  TaskExecStdioReadRequest,
  TaskExecStdioReadResponse,
  TaskExecWaitRequest,
  TaskExecWaitResponse,
  TaskMountDirectoryRequest,
  TaskReloadVolumesRequest,
  TaskSnapshotDirectoryRequest,
  TaskSnapshotDirectoryResponse,
  TaskSnapshotFilesystemRequest,
  TaskSnapshotFilesystemResponse,
  TaskSnapshotMemoryRequest,
  TaskSnapshotMemoryResponse,
  TaskUnmountDirectoryRequest,
  TaskSetNetworkAccessRequest,
  SandboxStdinWriteV2Request,
  SandboxStdinWriteV2Response,
  SandboxStdioFileDescriptor,
  SandboxStdioReadV2Request,
  SandboxStdioReadV2Response,
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
import { DEFAULT_SANDBOX_CHANNEL_IDLE_TIMEOUT_MS, isLocalhost } from "./config";
import { ClientClosedError, TimeoutError } from "./errors";

type TaskCommandRouterClient = Client<typeof TaskCommandRouterDefinition>;

/**
 * @internal
 * @hidden
 */
export type CommandRouterAccess = {
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

/**
 * Bytes per outbound message on a streaming stdin upload.
 *
 * @internal
 * @hidden
 */
export const STREAMING_STDIN_CHUNK_SIZE = 256 * 1024;

/**
 * Seekable byte source for streaming stdin uploads.
 *
 * @internal
 * @hidden
 */
export interface StdinSource {
  /**
   * Return a fresh iterable over the source's bytes starting at `offset`,
   * so an upload can resume mid-stream after a transient failure.
   */
  readFrom(offset: number): AsyncIterable<Uint8Array>;
}

/**
 * What one stdio stream supplies beyond what they share: which RPC to open, and
 * how a message maps onto bytes.
 *
 * @internal
 * @hidden
 */
type StdioStreamSpec<T> = {
  open: (offset: number, signal: AbortSignal) => AsyncIterable<T>;
  /** Where the next read resumes, so a reopened stream picks up where the last
   * one left off rather than repeating output the caller has seen. */
  nextOffset: (item: T, offset: number, firstOfAttempt: boolean) => number;
  /** Set only when the exec has a deadline, in which case running past it is a
   * timeout rather than an ordinary cancellation. */
  deadline: number | null;
  label: string;
  /**
   * Ends the call in flight when the caller stops reading.
   */
  signal?: AbortSignal;
};

/** gRPC status codes eligible for transient-error retries. */
const RETRYABLE_GRPC_STATUS_CODES = new Set([
  Status.DEADLINE_EXCEEDED,
  Status.UNAVAILABLE,
  Status.CANCELLED,
  Status.INTERNAL,
  Status.UNKNOWN,
]);

/**
 * Whether an error from a streaming stdin upload attempt is resumable via
 * `execStdinStatus` + a new attempt. Mirrors the transient-retry set used for
 * unary calls, plus UNAUTHENTICATED (handled with a JWT refresh before the
 * next attempt). Connection-level failures are covered by the status check:
 * grpc-js surfaces them as UNAVAILABLE/INTERNAL/CANCELLED.
 */
function isResumableStreamingStdinError(err: unknown): boolean {
  return (
    err instanceof ClientError &&
    (RETRYABLE_GRPC_STATUS_CODES.has(err.code) ||
      err.code === Status.UNAUTHENTICATED)
  );
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
        RETRYABLE_GRPC_STATUS_CODES.has(err.code) &&
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
  /**
   * The connection, absent while it is released. It is dialled again by the
   * next operation, so nothing may hold `stub` across one.
   */
  private channel: ReturnType<typeof createChannel> | undefined;
  private dial: () => ReturnType<typeof createChannel>;
  private factory!: ReturnType<typeof createClientFactory>;
  /**
   * How many operations are using the connection. A count rather than a flag
   * because operations overlap: only the last one out starts the idle clock.
   */
  private inFlight = 0;
  private idleTimer: ReturnType<typeof globalThis.setTimeout> | undefined;
  /**
   * Which timer the pending callback belongs to. A callback that finds a
   * different one has been superseded and stands down.
   */
  private idleTimerSeq = 0;
  /**
   * Bumped every time a connection is released. A stream opened before the last
   * bump is stale: the connection under it has since been given up, so its
   * failure is expected rather than a fault.
   */
  private generation = 0;
  /**
   * Streams open right now, tracked so they can be ended on release.
   *
   * grpc-js does not cancel a call in flight when its channel is closed, and a
   * subchannel with a live call on it is never collected - so closing alone
   * leaves the socket up. Aborting the calls first is what actually frees it.
   */
  private liveStreams = new Set<AbortController>();
  /**
   * How long the client may go unused before its connection is given up. Zero
   * keeps it up until the client is closed. Overridable in tests.
   */
  idleTimeoutMs: number = DEFAULT_SANDBOX_CHANNEL_IDLE_TIMEOUT_MS;
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

  /**
   * `access` is the router access returned by SandboxCreateV2, which lets a
   * freshly created sandbox connect without a round-trip. Pass undefined to look
   * it up — a re-attached sandbox, or the server could not mint a token.
   */
  static async tryInit(
    serverClient: ModalGrpcClient,
    taskId: string,
    sandboxId: string,
    isV2: boolean,
    access: CommandRouterAccess | undefined,
    logger: Logger,
    profile: Profile,
  ): Promise<TaskCommandRouterClientImpl | null> {
    let resp: CommandRouterAccess;
    if (access !== undefined) {
      resp = access;
    } else {
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

    const insecure = isLocalhost(profile);
    if (insecure) {
      logger.warn(
        "Using insecure TLS (skip certificate verification) for task command router",
      );
    }
    const dial = () =>
      createChannel(
        serverUrl,
        insecure
          ? ChannelCredentials.createInsecure()
          : ChannelCredentials.createSsl(),
        channelConfig,
      );

    const client = new TaskCommandRouterClientImpl(
      serverClient,
      taskId,
      sandboxId,
      isV2,
      resp.url,
      resp.jwt,
      dial,
      logger,
      profile.sandboxChannelIdleTimeoutMs,
    );

    logger.debug(
      "Successfully initialized command router client",
      "task_id",
      taskId,
    );

    return client;
  }

  /**
   * Builds a client around an already-resolved connection. Public so a test can
   * construct one without the control-plane round trip `tryInit` makes; the
   * class is not exported from the package, so this is not API surface.
   */
  constructor(
    serverClient: ModalGrpcClient,
    taskId: string,
    sandboxId: string,
    isV2: boolean,
    serverUrl: string,
    jwt: string,
    dial: () => ReturnType<typeof createChannel>,
    logger: Logger,
    idleTimeoutMs: number,
  ) {
    this.dial = dial;
    this.idleTimeoutMs = idleTimeoutMs;
    this.serverClient = serverClient;
    this.taskId = taskId;
    this.sandboxId = sandboxId;
    this.isV2 = isV2;
    this.serverUrl = serverUrl;
    this.jwt = jwt;
    this.jwtExp = parseJwtExpiration(jwt, logger);
    this.logger = logger;
    this.channel = dial();

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

    this.factory = factory;
    this.stub = factory.create(TaskCommandRouterDefinition, this.channel);

    // The connection is live from here on, so start the countdown now rather
    // than when the first operation finishes: an operation may never come, and
    // nothing else would give the connection back.
    this.armIdleTimer();
  }

  /**
   * Says the client is about to be used: holds off the idle timer, and dials
   * again if the connection was already given up. Pair every call with endOp.
   */
  private beginOp(): void {
    if (this.closed) {
      throw new ClientClosedError();
    }
    this.idleTimerSeq++;
    if (this.idleTimer !== undefined) {
      globalThis.clearTimeout(this.idleTimer);
      this.idleTimer = undefined;
    }
    if (this.channel === undefined) {
      this.channel = this.dial();
      this.stub = this.factory.create(
        TaskCommandRouterDefinition,
        this.channel,
      );
      this.logger.debug(
        "Reconnected to the command router after an idle release",
        "task_id",
        this.taskId,
      );
    }
    this.inFlight = (this.inFlight ?? 0) + 1;
  }

  /** Says the caller is done. The last one out starts the clock. */
  private endOp(): void {
    this.inFlight--;
    if (this.inFlight > 0) {
      return;
    }
    this.armIdleTimer();
  }

  /**
   * Starts the countdown to giving the connection back. Call it with nothing
   * in flight.
   */
  private armIdleTimer(): void {
    if (this.idleTimeoutMs <= 0 || this.closed) {
      return;
    }
    this.idleTimerSeq++;
    const seq = this.idleTimerSeq;
    this.idleTimer = globalThis.setTimeout(() => {
      if (seq !== this.idleTimerSeq) {
        return;
      }
      this.idleTimer = undefined;
      if (this.inFlight > 0 || this.channel === undefined) {
        return;
      }
      this.logger.debug(
        "Releasing the command router connection to an idle Sandbox",
        "task_id",
        this.taskId,
      );
      // Bumped before the streams are ended, so a reader that wakes to a
      // failure already sees that the stamp it took has moved on.
      this.generation++;
      for (const stream of this.liveStreams) {
        stream.abort();
      }
      this.liveStreams.clear();
      this.channel.close();
      this.channel = undefined;
    }, this.idleTimeoutMs);
    // A pending release must not hold the process open on its own.
    this.idleTimer.unref?.();
  }

  close(): void {
    if (this.closed) {
      return;
    }

    this.closed = true;
    this.idleTimerSeq++;
    if (this.idleTimer !== undefined) {
      globalThis.clearTimeout(this.idleTimer);
      this.idleTimer = undefined;
    }
    // Closing the channel does not end a call in flight, so a reader still
    // holding a stream would keep the socket up after a detach.
    for (const stream of this.liveStreams) {
      stream.abort();
    }
    this.liveStreams.clear();
    this.channel?.close();
    this.channel = undefined;
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
    signal?: AbortSignal,
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

    yield* this.streamExecStdio(taskId, execId, srFd, deadline, signal);
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

  /**
   * Read the current stdin write status for an exec'd command.
   *
   * Used by streaming clients to find the resume offset after a stream
   * failure. Evicts any in-flight stdin stream for the exec.
   */
  async execStdinStatus(
    taskId: string,
    execId: string,
  ): Promise<TaskExecStdinStatusResponse> {
    const request = TaskExecStdinStatusRequest.create({ taskId, execId });
    return await this.callUnary(() => this.stub.taskExecStdinStatus(request));
  }

  /**
   * Stream `source` into the exec's stdin, with bounded resume on transient
   * failures.
   *
   * Streams the full contents of `source` in one client-streaming RPC and
   * closes stdin (EOF) on success. On a resumable error, queries
   * `execStdinStatus` for the server's offset and reopens the
   * stream from that point. Returns the total bytes streamed.
   */
  async execStdinWriteStream(
    taskId: string,
    execId: string,
    source: StdinSource,
    chunkSize: number = STREAMING_STDIN_CHUNK_SIZE,
    maxResumeAttempts: number = 9,
  ): Promise<number> {
    let offset = 0;
    let attempt = 0;
    while (true) {
      let bytesRead = offset;
      let sourceExhausted = false;
      // A local source error must fail the upload immediately.
      let sourceError: unknown;

      const requests =
        async function* (): AsyncIterable<TaskExecStdinWriteStreamRequest> {
          yield TaskExecStdinWriteStreamRequest.create({
            start: { taskId, execId, offset },
          });
          const chunks = (async function* () {
            try {
              yield* source.readFrom(offset);
            } catch (err) {
              sourceError = err;
              throw err;
            }
          })();
          for await (const chunk of chunks) {
            for (let i = 0; i < chunk.length; i += chunkSize) {
              const data = chunk.subarray(i, i + chunkSize);
              if (data.length === 0) continue;
              bytesRead += data.length;
              yield TaskExecStdinWriteStreamRequest.create({ data });
            }
          }
          sourceExhausted = true;
          // The server closes stdin only on this explicit End message. A
          // stream that breaks before it leaves stdin open for resume.
          yield TaskExecStdinWriteStreamRequest.create({ end: {} });
        };

      // Registered so a release or a detach can end the upload; without it the
      // call would keep the connection alive after either.
      const abort = new AbortController();
      this.beginOp();
      this.liveStreams.add(abort);
      try {
        await this.stub.taskExecStdinWriteStream(requests(), {
          signal: abort.signal,
        } as CallOptions);
        return bytesRead;
      } catch (err) {
        if (sourceError !== undefined) {
          throw sourceError;
        }
        if (
          err instanceof ClientError &&
          err.code === Status.CANCELLED &&
          this.closed
        ) {
          throw new ClientClosedError();
        }
        if (!isResumableStreamingStdinError(err)) {
          throw err;
        }
        attempt++;
        if (attempt > maxResumeAttempts) {
          throw err;
        }
        if (err instanceof ClientError && err.code === Status.UNAUTHENTICATED) {
          // One refresh per attempt; the attempt counter above bounds the
          // total number of refreshes.
          await this.refreshJwt();
        }
        const status = await this.execStdinStatus(taskId, execId);
        if (status.closed) {
          // stdin only closes on our explicit End message; if the server
          // accepted everything we read and the source is exhausted, the
          // upload completed but the response was lost.
          if (sourceExhausted && status.numBytesWritten === bytesRead) {
            this.logger.debug(
              "execStdinWriteStream completed but response was lost",
              "error",
              err,
            );
            return bytesRead;
          }
          throw err;
        }
        offset = status.numBytesWritten;
        this.logger.debug(
          "execStdinWriteStream resuming after error",
          "offset",
          offset,
          "error",
          err,
        );
      } finally {
        this.endOp();
        if (this.liveStreams.delete(abort)) {
          abort.abort();
        }
      }
    }
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

  async snapshotMemory(
    request: TaskSnapshotMemoryRequest,
    options?: TimeoutOptions,
  ): Promise<TaskSnapshotMemoryResponse> {
    // Mirrors snapshotFilesystem's deadline handling.
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
            return this.stub.taskSnapshotMemory(request, {
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

  /**
   * Reload all Volumes mounted in the task to reflect their latest committed state.
   *
   * `timeoutMs` is the client-side deadline. If the reload does not complete
   * within this window, the call is cancelled and a TimeoutError is thrown.
   */
  async reloadVolumes(
    request: TaskReloadVolumesRequest,
    options?: TimeoutOptions,
  ): Promise<void> {
    const overallDeadlineMs =
      options?.timeoutMs !== undefined ? Date.now() + options.timeoutMs : null;
    try {
      await callWithRetriesOnTransientErrors(
        () =>
          this.callWithAuthRetry(() => {
            const remainingMs =
              overallDeadlineMs !== null
                ? Math.max(1, overallDeadlineMs - Date.now())
                : options?.timeoutMs;
            return this.stub.taskReloadVolumes(request, {
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

  async *sandboxStdioReadV2(
    taskId: string,
    fileDescriptor: FileDescriptor,
    signal?: AbortSignal,
  ): AsyncGenerator<SandboxStdioReadV2Response> {
    let srFd: SandboxStdioFileDescriptor;
    if (fileDescriptor === FileDescriptor.FILE_DESCRIPTOR_STDOUT) {
      srFd = SandboxStdioFileDescriptor.SANDBOX_STDIO_FILE_DESCRIPTOR_STDOUT;
    } else if (fileDescriptor === FileDescriptor.FILE_DESCRIPTOR_STDERR) {
      srFd = SandboxStdioFileDescriptor.SANDBOX_STDIO_FILE_DESCRIPTOR_STDERR;
    } else if (
      fileDescriptor === FileDescriptor.FILE_DESCRIPTOR_INFO ||
      fileDescriptor === FileDescriptor.FILE_DESCRIPTOR_UNSPECIFIED
    ) {
      throw new Error(`Unsupported file descriptor: ${fileDescriptor}`);
    } else {
      throw new Error(`Invalid file descriptor: ${fileDescriptor}`);
    }

    yield* this.streamSandboxStdio(taskId, srFd, signal);
  }

  async sandboxStdinWriteV2(
    taskId: string,
    offset: number,
    data: Uint8Array,
    eof: boolean,
  ): Promise<SandboxStdinWriteV2Response> {
    const request = SandboxStdinWriteV2Request.create({
      taskId,
      offset,
      data,
      eof,
    });
    return await this.callUnary(() => this.stub.sandboxStdinWriteV2(request));
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

  /**
   * Runs one attempt, refreshing the JWT and retrying once if it was rejected.
   *
   * Every unary call reaches this, so the lease is taken here: a method that
   * takes its own route to the stub cannot then reach a connection that has
   * been given up.
   */
  private async callWithAuthRetry<T>(func: () => Promise<T>): Promise<T> {
    this.beginOp();
    try {
      return await func();
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.UNAUTHENTICATED) {
        await this.refreshJwt();
        return await func();
      }
      throw err;
    } finally {
      this.endOp();
    }
  }

  /**
   * Yields the next chunk of stdio, opening or reopening the stream as it needs
   * to, and determining whether or not to retry a failed open or read. It waits
   * out the backoff itself.
   */
  private async *streamStdioWithRetries<T>(
    spec: StdioStreamSpec<T>,
  ): AsyncGenerator<T> {
    const baseDelayMs = 10;
    const delayFactor = 2;
    const maxRetries = 10;

    let offset = 0;
    let delayMs = baseDelayMs;
    let numRetriesRemaining = maxRetries;
    let didAuthRetry = false;

    while (true) {
      // Pulling from the stream is what keeps the client in use; while this
      // generator is suspended at a yield it is not, so the connection may be
      // given up under it. The stream then goes with it, and the loop below
      // reopens at the offset the consumer reached.
      spec.signal?.throwIfAborted();

      let stale = false;
      // Registered so a release can end this stream; a suspended generator
      // holds no lease, so the release may land while it is open.
      const abort = new AbortController();
      const abortWithCaller = () => abort.abort();
      spec.signal?.addEventListener("abort", abortWithCaller, { once: true });

      try {
        const generation = this.generation;
        this.beginOp();
        this.liveStreams.add(abort);
        let stream;
        try {
          stream = spec.open(offset, abort.signal);
        } finally {
          this.endOp();
        }

        try {
          const items = stream[Symbol.asyncIterator]();
          let firstOfAttempt = true;
          while (true) {
            // Held only while waiting on the wire, not while the consumer has
            // the chunk.
            this.beginOp();
            let next;
            try {
              next = await items.next();
            } finally {
              this.endOp();
            }
            if (next.done) {
              return;
            }
            const item = next.value;

            // We successfully authenticated after a JWT refresh, reset the auth retry flag.
            if (didAuthRetry) {
              didAuthRetry = false;
            }
            delayMs = baseDelayMs;
            numRetriesRemaining = maxRetries;
            offset = spec.nextOffset(item, offset, firstOfAttempt);
            firstOfAttempt = false;

            yield item;
          }
        } catch (err) {
          if (this.generation !== generation) {
            // The connection this stream was on was given up for idleness.
            // Nothing went wrong, so reopen without spending a retry.
            stale = true;
            continue;
          }
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
        if (stale) {
          continue;
        }
        if (spec.signal?.aborted) {
          throw err;
        }
        if (
          err instanceof ClientError &&
          err.code === Status.CANCELLED &&
          this.closed
        ) {
          throw new ClientClosedError();
        }
        if (
          err instanceof ClientError &&
          RETRYABLE_GRPC_STATUS_CODES.has(err.code) &&
          numRetriesRemaining > 0
        ) {
          if (spec.deadline && spec.deadline - Date.now() <= delayMs) {
            throw new Error(
              `Deadline exceeded while streaming stdio for ${spec.label}`,
            );
          }

          this.logger.debug(
            "Retrying stdio read with delay",
            "delay_ms",
            delayMs,
            "error",
            err,
          );
          await setTimeout(delayMs, undefined, { signal: spec.signal });
          delayMs *= delayFactor;
          numRetriesRemaining--;
        } else {
          throw err;
        }
      } finally {
        spec.signal?.removeEventListener("abort", abortWithCaller);
        // A consumer can abandon this generator while it is suspended at a
        // yield, which reaches here and nowhere else. Ending the stream is what
        // frees the socket under it.
        if (this.liveStreams.delete(abort)) {
          abort.abort();
        }
      }
    }
  }

  private streamExecStdio(
    taskId: string,
    execId: string,
    fileDescriptor: TaskExecStdioFileDescriptor,
    deadline: number | null,
    signal?: AbortSignal,
  ): AsyncGenerator<TaskExecStdioReadResponse> {
    return this.streamStdioWithRetries({
      open: (offset, signal) =>
        this.stub.taskExecStdioRead(
          TaskExecStdioReadRequest.create({
            taskId,
            execId,
            offset,
            fileDescriptor,
          }),
          {
            timeoutMs:
              deadline !== null
                ? Math.max(0, deadline - Date.now())
                : undefined,
            signal,
          } as CallOptions & TimeoutOptions,
        ),
      nextOffset: (item, offset) => offset + item.data.length,
      deadline,
      label: `exec ${execId}`,
      signal,
    });
  }

  private streamSandboxStdio(
    taskId: string,
    fileDescriptor: SandboxStdioFileDescriptor,
    signal?: AbortSignal,
  ): AsyncGenerator<SandboxStdioReadV2Response> {
    return this.streamStdioWithRetries({
      open: (offset, signal) =>
        this.stub.sandboxStdioReadV2(
          SandboxStdioReadV2Request.create({ taskId, offset, fileDescriptor }),
          { signal } as CallOptions,
        ),
      nextOffset: (item, offset, firstOfAttempt) =>
        (firstOfAttempt ? item.startingOffset : offset) + item.data.length,
      deadline: null,
      label: `Sandbox task ${taskId}`,
      signal,
    });
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
