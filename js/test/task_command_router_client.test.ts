import { expect, test, vi } from "vitest";
import {
  parseJwtExpiration,
  callWithRetriesOnTransientErrors,
  TaskCommandRouterClientImpl,
  type StdinSource,
} from "../src/task_command_router_client";
import { ClientError, Status } from "nice-grpc";
import {
  SandboxStdinWriteV2Request,
  SandboxStdioFileDescriptor,
  SandboxStdioReadV2Request,
  SandboxStdioReadV2Response,
  TaskExecStdinStatusResponse,
  TaskExecStdinWriteStreamRequest,
  TaskSnapshotFilesystemRequest,
} from "../proto/modal_proto/task_command_router";
import { FileDescriptor } from "../proto/modal_proto/api";
import { TimeoutError } from "../src/errors";

const mockLogger = {
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
};

/**
 * A client built the way the real one is, with its stub swapped for a fake.
 *
 * Constructing rather than assembling matters: the field initialisers only run
 * as part of construction, and a client missing one fails in ways that do not
 * look like a missing field.
 */
function makeTestClient(stub: unknown): any {
  const client: any = new TaskCommandRouterClientImpl(
    undefined as any, // serverClient, set by the tests that need it
    "ta-1",
    "sb-1",
    true,
    "https://example.com",
    "fake-jwt",
    () => ({ close() {} }) as any, // never dialled: the stub below replaces it
    mockLogger as any,
    0, // idle release off; these tests are about the calls, not the connection
  );
  client.stub = stub;
  return client;
}

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

  const client = makeTestClient(mockStub);

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
  return makeTestClient(server.stub());
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

class FakeSandboxStdioServer {
  readOffsets: number[] = [];
  readFds: SandboxStdioFileDescriptor[] = [];
  stdinBuffer: number[] = [];
  stdinClosed = false;

  constructor(
    private readonly output: Partial<
      Record<SandboxStdioFileDescriptor, Uint8Array>
    > = {},
    private readonly opts: {
      chunkSize?: number;
      droppedPrefix?: number;
      failures?: number;
    } = {},
  ) {}

  stub() {
    let failuresRemaining = this.opts.failures ?? 0;
    return {
      sandboxStdioReadV2: (req: SandboxStdioReadV2Request) => {
        this.readOffsets.push(req.offset);
        this.readFds.push(req.fileDescriptor);
        const fail = failuresRemaining > 0;
        if (fail) failuresRemaining--;
        const full = this.output[req.fileDescriptor] ?? new Uint8Array(0);
        const chunkSize = this.opts.chunkSize || full.length;
        let offset = Math.max(req.offset, this.opts.droppedPrefix ?? 0);
        return (async function* () {
          while (offset < full.length) {
            const end = Math.min(offset + chunkSize, full.length);
            yield SandboxStdioReadV2Response.create({
              data: full.subarray(offset, end),
              startingOffset: offset,
            });
            offset = end;
            if (fail) throw unavailable();
          }
        })();
      },
      sandboxStdinWriteV2: async (req: SandboxStdinWriteV2Request) => {
        this.stdinBuffer = this.stdinBuffer.slice(0, req.offset);
        this.stdinBuffer.push(...req.data);
        if (req.eof) this.stdinClosed = true;
        return {};
      },
    };
  }
}

function makeSandboxStdioClient(server: FakeSandboxStdioServer): any {
  const client = Object.create(TaskCommandRouterClientImpl.prototype) as any;
  client.stub = server.stub();
  client.logger = mockLogger;
  client.closed = false;
  // The stdio streams take part in the idle-release bookkeeping. A channel that
  // is already there keeps beginOp from dialling, and a zero timeout keeps
  // endOp from scheduling a release.
  client.channel = {};
  client.liveStreams = new Set();
  client.generation = 0;
  client.inFlight = 0;
  client.idleTimerSeq = 0;
  client.idleTimeoutMs = 0;
  return client;
}

function deterministicBytes(n: number): Uint8Array {
  return new Uint8Array(n).map((_, i) => i % 251);
}

async function collectStdio(
  stream: AsyncIterable<SandboxStdioReadV2Response>,
): Promise<Uint8Array> {
  const chunks: number[] = [];
  for await (const item of stream) {
    chunks.push(...item.data);
  }
  return new Uint8Array(chunks);
}

test("sandboxStdioReadV2 does not reopen when the caller aborts mid-backoff", async () => {
  let opens = 0;
  const client = Object.create(TaskCommandRouterClientImpl.prototype) as any;
  client.logger = mockLogger;
  client.closed = false;
  client.channel = {};
  client.liveStreams = new Set();
  client.generation = 0;
  client.inFlight = 0;
  client.idleTimerSeq = 0;
  client.idleTimeoutMs = 0;
  client.stub = {
    sandboxStdioReadV2: () => {
      opens++;
      return (async function* () {
        throw new ClientError("/test", Status.UNAVAILABLE, "unavailable");
        yield SandboxStdioReadV2Response.create({});
      })();
    },
  };

  const caller = new AbortController();
  const stream = client.sandboxStdioReadV2(
    "ta-1",
    FileDescriptor.FILE_DESCRIPTOR_STDOUT,
    caller.signal,
  );
  let chunks = 0;
  const drained = (async () => {
    for await (const _ of stream) {
      chunks++;
    }
  })().catch(() => undefined);

  await new Promise((resolve) => globalThis.setTimeout(resolve, 5));
  caller.abort();

  const started = Date.now();
  await drained;
  expect(chunks).toBe(0);
  expect(Date.now() - started).toBeLessThan(1000);
  expect(opens).toBe(1);
});

test("sandboxStdioReadV2 does not retry a call the caller aborted", async () => {
  let opens = 0;
  const client = Object.create(TaskCommandRouterClientImpl.prototype) as any;
  client.logger = mockLogger;
  client.closed = false;
  client.channel = {};
  client.liveStreams = new Set();
  client.generation = 0;
  client.inFlight = 0;
  client.idleTimerSeq = 0;
  client.idleTimeoutMs = 0;
  client.stub = {
    sandboxStdioReadV2: (_req: unknown, opts: { signal?: AbortSignal }) => {
      opens++;
      return (async function* () {
        await new Promise((_resolve, reject) => {
          opts.signal?.addEventListener(
            "abort",
            () => reject(new ClientError("/test", Status.CANCELLED, "cancel")),
            { once: true },
          );
        });
        yield SandboxStdioReadV2Response.create({});
      })();
    },
  };

  const caller = new AbortController();
  const stream = client.sandboxStdioReadV2(
    "ta-1",
    FileDescriptor.FILE_DESCRIPTOR_STDOUT,
    caller.signal,
  );
  let chunks = 0;
  const drained = (async () => {
    for await (const _ of stream) {
      chunks++;
    }
  })().catch(() => undefined);

  await new Promise((resolve) => globalThis.setTimeout(resolve, 50));
  expect(opens).toBe(1);

  const started = Date.now();
  caller.abort();
  await drained;
  expect(chunks).toBe(0);
  expect(Date.now() - started).toBeLessThan(1000);
  expect(opens).toBe(1);
});

test("sandboxStdioReadV2 streams the whole buffer in one call", async () => {
  const stdout = deterministicBytes(5000);
  const server = new FakeSandboxStdioServer(
    {
      [SandboxStdioFileDescriptor.SANDBOX_STDIO_FILE_DESCRIPTOR_STDOUT]: stdout,
    },
    { chunkSize: 1000 },
  );
  const client = makeSandboxStdioClient(server);

  const got = await collectStdio(
    client.sandboxStdioReadV2("ta-1", FileDescriptor.FILE_DESCRIPTOR_STDOUT),
  );

  expect(got).toEqual(stdout);
  expect(server.readOffsets).toEqual([0]);
});

test("sandboxStdioReadV2 resumes from the next byte after a transient error", async () => {
  const stdout = deterministicBytes(5000);
  const server = new FakeSandboxStdioServer(
    {
      [SandboxStdioFileDescriptor.SANDBOX_STDIO_FILE_DESCRIPTOR_STDOUT]: stdout,
    },
    { chunkSize: 1000, failures: 1 },
  );
  const client = makeSandboxStdioClient(server);

  const got = await collectStdio(
    client.sandboxStdioReadV2("ta-1", FileDescriptor.FILE_DESCRIPTOR_STDOUT),
  );

  expect(got).toEqual(stdout);
  expect(server.readOffsets).toEqual([0, 1000]);
});

test("sandboxStdioReadV2 rebases the resume offset onto the worker's starting offset", async () => {
  const stdout = deterministicBytes(5000);
  const server = new FakeSandboxStdioServer(
    {
      [SandboxStdioFileDescriptor.SANDBOX_STDIO_FILE_DESCRIPTOR_STDOUT]: stdout,
    },
    { chunkSize: 1000, droppedPrefix: 1500, failures: 1 },
  );
  const client = makeSandboxStdioClient(server);

  const got = await collectStdio(
    client.sandboxStdioReadV2("ta-1", FileDescriptor.FILE_DESCRIPTOR_STDOUT),
  );

  expect(got).toEqual(stdout.subarray(1500));
  expect(server.readOffsets).toEqual([0, 2500]);
});

test("sandboxStdioReadV2 maps stderr onto the Sandbox stdio descriptor", async () => {
  const stderr = deterministicBytes(64);
  const server = new FakeSandboxStdioServer({
    [SandboxStdioFileDescriptor.SANDBOX_STDIO_FILE_DESCRIPTOR_STDERR]: stderr,
  });
  const client = makeSandboxStdioClient(server);

  const got = await collectStdio(
    client.sandboxStdioReadV2("ta-1", FileDescriptor.FILE_DESCRIPTOR_STDERR),
  );

  expect(got).toEqual(stderr);
  expect(server.readFds).toEqual([
    SandboxStdioFileDescriptor.SANDBOX_STDIO_FILE_DESCRIPTOR_STDERR,
  ]);
});

test("sandboxStdioReadV2 rejects descriptors without a Sandbox stdio equivalent", async () => {
  const server = new FakeSandboxStdioServer();
  const client = makeSandboxStdioClient(server);

  await expect(
    collectStdio(
      client.sandboxStdioReadV2("ta-1", FileDescriptor.FILE_DESCRIPTOR_INFO),
    ),
  ).rejects.toThrow("Unsupported file descriptor");
});

test("sandboxStdinWriteV2 writes at the given offset and closes on eof", async () => {
  const server = new FakeSandboxStdioServer();
  const client = makeSandboxStdioClient(server);

  await client.sandboxStdinWriteV2("ta-1", 0, new Uint8Array([1, 2, 3]), false);
  await client.sandboxStdinWriteV2("ta-1", 3, new Uint8Array(0), true);

  expect(new Uint8Array(server.stdinBuffer)).toEqual(new Uint8Array([1, 2, 3]));
  expect(server.stdinClosed).toBe(true);
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

  const client = makeTestClient(undefined);
  client.serverClient = mockServerClient;
  // The V1 access path, which is what this mock implements.
  client.isV2 = false;
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
