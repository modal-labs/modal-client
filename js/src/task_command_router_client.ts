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
} from "../proto/modal_proto/task_command_router";
import {
  TaskGetCommandRouterAccessRequest,
  FileDescriptor,
  TaskGetCommandRouterAccessResponse,
} from "../proto/modal_proto/api";
import type { ModalGrpcClient } from "./client";
import { timeoutMiddleware, type TimeoutOptions } from "./client";
import type { Logger } from "./logger";
import type { Profile } from "./config";
import { isLocalhost } from "./config";
import { ClientClosedError } from "./errors";

type TaskCommandRouterClient = Client<typeof TaskCommandRouterDefinition>;

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

export async function callWithRetriesOnTransientErrors<T>(
  func: () => Promise<T>,
  baseDelayMs: number = 10,
  delayFactor: number = 2,
  maxRetries: number | null = 10,
  deadlineMs: number | null = null,
  isClosed?: () => boolean,
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

  while (true) {
    if (deadlineMs !== null && Date.now() >= deadlineMs) {
      throw new Error("Deadline exceeded");
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
        (maxRetries === null || numRetries < maxRetries)
      ) {
        if (deadlineMs !== null && Date.now() + delayMs >= deadlineMs) {
          throw new Error("Deadline exceeded");
        }

        await setTimeout(delayMs);
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
  private serverUrl: string;
  private jwt: string;
  private jwtExp: number | null;
  private jwtRefreshLock: Promise<void> = Promise.resolve();
  private logger: Logger;
  private closed: boolean = false;

  static async tryInit(
    serverClient: ModalGrpcClient,
    taskId: string,
    logger: Logger,
    profile: Profile,
  ): Promise<TaskCommandRouterClientImpl | null> {
    let resp: TaskGetCommandRouterAccessResponse;
    try {
      resp = await serverClient.taskGetCommandRouterAccess(
        TaskGetCommandRouterAccessRequest.create({ taskId }),
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
    serverUrl: string,
    jwt: string,
    channel: ReturnType<typeof createChannel>,
    logger: Logger,
  ) {
    this.serverClient = serverClient;
    this.taskId = taskId;
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

  async execStart(
    request: TaskExecStartRequest,
  ): Promise<TaskExecStartResponse> {
    return await callWithRetriesOnTransientErrors(
      () => this.callWithAuthRetry(() => this.stub.taskExecStart(request)),
      10,
      2,
      10,
      null,
      () => this.closed,
    );
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
    return await callWithRetriesOnTransientErrors(
      () => this.callWithAuthRetry(() => this.stub.taskExecStdinWrite(request)),
      10,
      2,
      10,
      null,
      () => this.closed,
    );
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
    await callWithRetriesOnTransientErrors(
      () => this.callWithAuthRetry(() => this.stub.taskMountDirectory(request)),
      10,
      2,
      10,
      null,
      () => this.closed,
    );
  }

  async snapshotDirectory(
    request: TaskSnapshotDirectoryRequest,
  ): Promise<TaskSnapshotDirectoryResponse> {
    return await callWithRetriesOnTransientErrors(
      () =>
        this.callWithAuthRetry(() => this.stub.taskSnapshotDirectory(request)),
      10,
      2,
      10,
      null,
      () => this.closed,
    );
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
        const resp = await this.serverClient.taskGetCommandRouterAccess(
          TaskGetCommandRouterAccessRequest.create({ taskId: this.taskId }),
        );

        if (resp.url !== this.serverUrl) {
          throw new Error("Task router URL changed during session");
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
