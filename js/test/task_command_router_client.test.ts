import { expect, test, vi } from "vitest";
import {
  parseJwtExpiration,
  callWithRetriesOnTransientErrors,
  TaskCommandRouterClientImpl,
} from "../src/task_command_router_client";
import { ClientError, Status } from "nice-grpc";
import { TaskSnapshotFilesystemRequest } from "../proto/modal_proto/task_command_router";
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
