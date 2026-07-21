import { expect, test, vi } from "vitest";
import {
  parseJwtExpiration,
  callWithRetriesOnTransientErrors,
  TaskCommandRouterClientImpl,
  type StdinSource,
} from "../src/task_command_router_client";
import { ClientError, Status } from "nice-grpc";
import {
  TaskExecStdinStatusResponse,
  TaskExecStdinWriteStreamRequest,
  TaskSnapshotFilesystemRequest,
} from "../proto/modal_proto/task_command_router";
import { TimeoutError } from "../src/errors";

const mockLogger = {
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
};

function mockJwt(exp: number | string | null): string {
  const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  const payload =
    exp !== null ? btoa(JSON.stringify({ exp })) : btoa(JSON.stringify({}));
  const signature = "fake-signature";
  return `${header}.${payload}.${signature}`;
}

test("parseJwtExpiration with valid JWT", () => {
  const exp = Math.floor(Date.now() / 1000) + 3600;
  const jwt = mockJwt(exp);
  const result = parseJwtExpiration(jwt, mockLogger);
  expect(result).toBe(exp);
});

test("parseJwtExpiration without exp claim", () => {
  const jwt = mockJwt(null);
  const result = parseJwtExpiration(jwt, mockLogger);
  expect(result).toBeNull();
});

test("parseJwtExpiration with malformed JWT (wrong number of parts)", () => {
  const jwt = "only.two";
  const result = parseJwtExpiration(jwt, mockLogger);
  expect(result).toBeNull();
});

test("parseJwtExpiration with invalid base64", () => {
  const jwt = "invalid.!!!invalid!!!.signature";
  const result = parseJwtExpiration(jwt, mockLogger);
  expect(result).toBeNull();
  expect(mockLogger.warn).toHaveBeenCalled();
});

test("parseJwtExpiration with non-numeric exp", () => {
  const jwt = mockJwt("not-a-number");
  const result = parseJwtExpiration(jwt, mockLogger);
  expect(result).toBeNull();
});

test("callWithRetriesOnTransientErrors success on first attempt", async () => {
  const func = vi.fn().mockResolvedValue("success");
  const result = await callWithRetriesOnTransientErrors(func);
  expect(result).toBe("success");
  expect(func).toHaveBeenCalledTimes(1);
});

test.each([
  [Status.DEADLINE_EXCEEDED, "timeout"],
  [Status.UNAVAILABLE, "unavailable"],
  [Status.CANCELLED, "cancelled"],
  [Status.INTERNAL, "internal error"],
  [Status.UNKNOWN, "unknown error"],
])(
  "callWithRetriesOnTransientErrors retries on %s",
  async (status, message) => {
    const func = vi
      .fn()
      .mockRejectedValueOnce(new ClientError("/test", status, message))
      .mockResolvedValue("success");
    const result = await callWithRetriesOnTransientErrors(func, 10);
    expect(result).toBe("success");
    expect(func).toHaveBeenCalledTimes(2);
  },
);

test("callWithRetriesOnTransientErrors non-retryable error", async () => {
  const error = new ClientError("/test", Status.INVALID_ARGUMENT, "invalid");
  const func = vi.fn().mockRejectedValue(error);
  await expect(callWithRetriesOnTransientErrors(func, 10)).rejects.toThrow(
    error,
  );
  expect(func).toHaveBeenCalledTimes(1);
});

test("callWithRetriesOnTransientErrors max retries exceeded", async () => {
  const error = new ClientError("/test", Status.UNAVAILABLE, "unavailable");
  const func = vi.fn().mockRejectedValue(error);
  const maxRetries = 3;
  await expect(
    callWithRetriesOnTransientErrors(func, 10, 2, maxRetries),
  ).rejects.toThrow(error);
  expect(func).toHaveBeenCalledTimes(maxRetries + 1);
});

test.each([Status.DEADLINE_EXCEEDED, Status.CANCELLED])(
  "callWithRetriesOnTransientErrors does not retry excluded status %s",
  async (excludedStatus) => {
    const error = new ClientError("/test", excludedStatus, "excluded");
    const func = vi.fn().mockRejectedValue(error);
    await expect(
      callWithRetriesOnTransientErrors(func, 10, 2, 10, null, undefined, [
        Status.DEADLINE_EXCEEDED,
        Status.CANCELLED,
      ]),
    ).rejects.toThrow(error);
    // Excluded codes are not retried, even though they're in the
    // general retryable set.
    expect(func).toHaveBeenCalledTimes(1);
  },
);

test("callWithRetriesOnTransientErrors exclude codes does not affect other retryable codes", async () => {
  const transient = new ClientError("/test", Status.UNAVAILABLE, "unavailable");
  const func = vi.fn().mockRejectedValueOnce(transient).mockResolvedValue("ok");
  const result = await callWithRetriesOnTransientErrors(
    func,
    10,
    2,
    10,
    null,
    undefined,
    [Status.DEADLINE_EXCEEDED, Status.CANCELLED],
  );
  expect(result).toBe("ok");
  expect(func).toHaveBeenCalledTimes(2);
});

test("callWithRetriesOnTransientErrors deadline exceeded", async () => {
  const error = new ClientError("/test", Status.UNAVAILABLE, "unavailable");
  const func = vi.fn().mockRejectedValue(error);
  const deadline = Date.now() + 50;
  await expect(
    callWithRetriesOnTransientErrors(func, 100, 2, null, deadline),
  ).rejects.toThrow();
});

// Regression test for a preemptive-deadline error-translation bug.
//
// `callWithRetriesOnTransientErrors` throws `RetryDeadlineExceededError`
// as soon as the *next* backoff sleep would overshoot the deadline — at
// that moment `Date.now()` is still strictly before the deadline.
// `snapshotFilesystem`'s outer translation only converts to TimeoutError
// when `Date.now() >= overallDeadlineMs`, so the internal sentinel leaks
// through to the caller instead of TimeoutError.
test("snapshotFilesystem preemptive deadline returns TimeoutError", async () => {
  const mockStub = {
    taskSnapshotFilesystem: vi
      .fn()
      .mockRejectedValue(
        new ClientError("/test", Status.UNAVAILABLE, "transient"),
      ),
  };

  const client = Object.create(TaskCommandRouterClientImpl.prototype) as any;
  client.stub = mockStub;
  client.jwt = "fake-jwt";
  client.closed = false;

  // With baseDelay=10ms doubling each retry, a 100ms timeout will reach
  // a point where Date.now()+nextDelay >= deadline before Date.now()
  // itself crosses it, triggering the preemptive throw.
  await expect(
    client.snapshotFilesystem(
      TaskSnapshotFilesystemRequest.create({ taskId: "t" }),
      { timeoutMs: 100 },
    ),
  ).rejects.toBeInstanceOf(TimeoutError);
});

// ---------------------------------------------------------------------------
// execStdinWriteStream
// ---------------------------------------------------------------------------

/** One scripted failure for a single TaskExecStdinWriteStream call. */
interface ScriptedFailure {
  /**
   * Payload bytes to accept before throwing `error`. Use `Infinity` with
   * `afterEnd` to consume the whole stream (including End) and then fail,
   * simulating a lost response.
   */
  acceptBytes: number;
  error: Error;
  afterEnd?: boolean;
}

/**
 * In-memory fake of the server side of TaskExecStdinWriteStream and
 * TaskExecStdinStatus, with scripted per-call failures.
 */
class FakeStdinStreamServer {
  buffer: number[] = [];
  closed = false;
  writeStreamCalls = 0;
  statusCalls = 0;
  /** Start offset observed on each TaskExecStdinWriteStream call. */
  startOffsets: number[] = [];
  /** Data message sizes observed on each TaskExecStdinWriteStream call. */
  dataSizes: number[][] = [];
  /** Failures applied to successive calls, in order. */
  failures: ScriptedFailure[];

  constructor(failures: ScriptedFailure[] = []) {
    this.failures = failures;
  }

  stub() {
    return {
      taskExecStdinWriteStream: async (
        requests: AsyncIterable<TaskExecStdinWriteStreamRequest>,
      ) => {
        this.writeStreamCalls++;
        const failure = this.failures.shift();
        let accepted = 0;
        const sizes: number[] = [];
        this.dataSizes.push(sizes);
        for await (const req of requests) {
          if (req.start !== undefined) {
            this.startOffsets.push(req.start.offset);
            this.buffer = this.buffer.slice(0, req.start.offset);
          } else if (req.data !== undefined) {
            if (
              failure !== undefined &&
              !failure.afterEnd &&
              accepted >= failure.acceptBytes
            ) {
              throw failure.error;
            }
            sizes.push(req.data.length);
            this.buffer.push(...req.data);
            accepted += req.data.length;
          } else if (req.end !== undefined) {
            this.closed = true;
          }
        }
        if (failure !== undefined) {
          throw failure.error;
        }
        return {};
      },
      taskExecStdinStatus: async () => {
        this.statusCalls++;
        return TaskExecStdinStatusResponse.create({
          numBytesWritten: this.buffer.length,
          closed: this.closed,
        });
      },
    };
  }
}

function makeStdinStreamClient(server: FakeStdinStreamServer): any {
  const client = Object.create(TaskCommandRouterClientImpl.prototype) as any;
  client.stub = server.stub();
  client.logger = mockLogger;
  client.closed = false;
  return client;
}

function bytesSource(bytes: Uint8Array): StdinSource {
  return {
    readFrom(offset: number): AsyncIterable<Uint8Array> {
      return (async function* () {
        yield bytes.subarray(offset);
      })();
    },
  };
}

const unavailable = () =>
  new ClientError("/test", Status.UNAVAILABLE, "unavailable");

test("execStdinWriteStream streams start, data chunks, and end", async () => {
  const server = new FakeStdinStreamServer();
  const client = makeStdinStreamClient(server);
  const data = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

  const total = await client.execStdinWriteStream(
    "ta-1",
    "ex-1",
    bytesSource(data),
    4, // chunkSize
  );

  expect(total).toBe(10);
  expect(server.startOffsets).toEqual([0]);
  expect(server.dataSizes).toEqual([[4, 4, 2]]);
  expect(new Uint8Array(server.buffer)).toEqual(data);
  expect(server.closed).toBe(true);
  expect(server.statusCalls).toBe(0);
});

test("execStdinWriteStream empty source sends start and end only", async () => {
  const server = new FakeStdinStreamServer();
  const client = makeStdinStreamClient(server);

  const total = await client.execStdinWriteStream(
    "ta-1",
    "ex-1",
    bytesSource(new Uint8Array(0)),
    4,
  );

  expect(total).toBe(0);
  expect(server.startOffsets).toEqual([0]);
  expect(server.dataSizes).toEqual([[]]);
  expect(server.buffer).toEqual([]);
  expect(server.closed).toBe(true);
});

test("execStdinWriteStream resumes from reported offset after midstream failure", async () => {
  const server = new FakeStdinStreamServer([
    { acceptBytes: 8, error: unavailable() },
  ]);
  const client = makeStdinStreamClient(server);
  const data = new Uint8Array(12).map((_, i) => i);

  const total = await client.execStdinWriteStream(
    "ta-1",
    "ex-1",
    bytesSource(data),
    4,
  );

  expect(total).toBe(12);
  expect(server.writeStreamCalls).toBe(2);
  expect(server.statusCalls).toBe(1);
  // The second attempt resumed from the server's canonical offset.
  expect(server.startOffsets).toEqual([0, 8]);
  expect(new Uint8Array(server.buffer)).toEqual(data);
  expect(server.closed).toBe(true);
});

test("execStdinWriteStream throws after exhausting resume attempts", async () => {
  // 10 total attempts: the initial one plus 9 resumes.
  const failures = Array.from({ length: 10 }, () => ({
    acceptBytes: 0,
    error: unavailable(),
  }));
  const server = new FakeStdinStreamServer(failures);
  const client = makeStdinStreamClient(server);

  await expect(
    client.execStdinWriteStream(
      "ta-1",
      "ex-1",
      bytesSource(new Uint8Array([1, 2, 3])),
      4,
    ),
  ).rejects.toThrow("unavailable");

  expect(server.writeStreamCalls).toBe(10);
  expect(server.statusCalls).toBe(9);
  expect(server.closed).toBe(false);
});

test("execStdinWriteStream treats closed stream with all bytes written as success", async () => {
  // The server consumes the whole stream (including End) but the response is
  // lost to a transient error.
  const server = new FakeStdinStreamServer([
    { acceptBytes: Infinity, error: unavailable(), afterEnd: true },
  ]);
  const client = makeStdinStreamClient(server);
  const data = new Uint8Array([9, 8, 7, 6, 5]);

  const total = await client.execStdinWriteStream(
    "ta-1",
    "ex-1",
    bytesSource(data),
    4,
  );

  expect(total).toBe(5);
  expect(server.writeStreamCalls).toBe(1);
  expect(server.statusCalls).toBe(1);
  expect(new Uint8Array(server.buffer)).toEqual(data);
  expect(server.closed).toBe(true);
});

test("execStdinWriteStream does not resume on a local source error", async () => {
  const server = new FakeStdinStreamServer();
  const client = makeStdinStreamClient(server);
  // ENOENT-style Node system errors carry a string `code`, which must not be
  // mistaken for a resumable connection error.
  const sourceError = Object.assign(new Error("boom"), { code: "ENOENT" });
  const failingSource = {
    // eslint-disable-next-line require-yield
    async *readFrom(): AsyncIterable<Uint8Array> {
      throw sourceError;
    },
  };

  await expect(
    client.execStdinWriteStream("ta-1", "ex-1", failingSource, 4),
  ).rejects.toThrow("boom");

  expect(server.statusCalls).toBe(0);
});

test("execStdinWriteStream does not resume on FAILED_PRECONDITION", async () => {
  const server = new FakeStdinStreamServer([
    {
      acceptBytes: 0,
      error: new ClientError("/test", Status.FAILED_PRECONDITION, "dropped"),
    },
  ]);
  const client = makeStdinStreamClient(server);

  await expect(
    client.execStdinWriteStream(
      "ta-1",
      "ex-1",
      bytesSource(new Uint8Array([1, 2, 3])),
      4,
    ),
  ).rejects.toThrow("dropped");

  expect(server.writeStreamCalls).toBe(1);
  expect(server.statusCalls).toBe(0);
});

test("refreshJwt recovers after transient failure", async () => {
  let callCount = 0;
  const mockServerClient = {
    taskGetCommandRouterAccess: vi.fn().mockImplementation(async () => {
      callCount++;
      if (callCount === 1) {
        throw new Error("Transient network error");
      }
      return {
        url: "https://example.com",
        jwt: mockJwt(Math.floor(Date.now() / 1000) + 3600),
      };
    }),
  };

  const client = Object.create(TaskCommandRouterClientImpl.prototype) as any;
  client.serverClient = mockServerClient;
  client.taskId = "test-task";
  client.serverUrl = "https://example.com";
  client.jwt = mockJwt(0); // Expired JWT
  client.jwtExp = 0; // Expired, so refresh will attempt
  client.jwtRefreshLock = Promise.resolve();
  client.logger = mockLogger;
  client.closed = false;

  const refreshJwt = client.refreshJwt.bind(client);

  await expect(refreshJwt()).rejects.toThrow("Transient network error");
  expect(callCount).toBe(1);

  await expect(refreshJwt()).resolves.not.toThrow();
  expect(callCount).toBe(2);
});
