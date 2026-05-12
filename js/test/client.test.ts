import {
  ClientMiddlewareCall,
  CallOptions,
  ClientError,
  Metadata,
  Status,
} from "nice-grpc";
import { ModalClient, type Logger } from "modal";
import { RPCRetryPolicy, RPCStatus } from "../proto/modal_proto/api";
import { Any } from "../proto/google/protobuf/any";
import { afterEach, expect, test, vi } from "vitest";

// --- helpers for RPCRetryPolicy tests ---

const noopLogger: Logger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
};

/** Encode a grpc-status-details-bin trailer carrying an RPCRetryPolicy. */
function buildThrottleTrailer(retryAfterSecs: number): Buffer {
  const retryPolicyBytes = RPCRetryPolicy.encode(
    RPCRetryPolicy.fromPartial({ retryAfterSecs }),
  ).finish();
  const statusBytes = RPCStatus.encode(
    RPCStatus.fromPartial({
      code: 8, // RESOURCE_EXHAUSTED
      message: "server throttled",
      details: [
        Any.fromPartial({
          typeUrl: "type.googleapis.com/modal.client.RPCRetryPolicy",
          value: retryPolicyBytes,
        }),
      ],
    }),
  ).finish();
  return Buffer.from(statusBytes);
}

type MockCallResult = {
  /** The call object to pass to the middleware. */
  call: ReturnType<typeof makeMockCall>;
  getCallCount: () => number;
  getLastMetadata: () => Metadata | undefined;
};

/** Build a mock ClientMiddlewareCall whose next() throws throttle errors N times then succeeds. */
function makeThrottlingMockCall(
  retryAfterSecs: number,
  throttleTimes: number,
): MockCallResult {
  let callCount = 0;
  let lastMetadata: Metadata | undefined;
  const trailerBytes = buildThrottleTrailer(retryAfterSecs);

  const call = makeMockCall(async function* (
    _request: unknown,
    options: CallOptions,
  ) {
    callCount++;
    lastMetadata = options.metadata as Metadata;
    if (callCount <= throttleTimes) {
      const trailer = new Metadata({
        "grpc-status-details-bin": trailerBytes,
      });
      options.onTrailer?.(trailer);
      throw new ClientError(
        "/modal.client.ModalClient/AppGetOrCreate",
        Status.RESOURCE_EXHAUSTED,
        "server throttled",
      );
    }
    yield { appId: "ok" };
  });

  return {
    call,
    getCallCount: () => callCount,
    getLastMetadata: () => lastMetadata,
  };
}

function makeMockCall(
  nextFn: (req: unknown, opts: CallOptions) => AsyncGenerator<any>,
): any {
  return {
    method: {
      path: "/modal.client.ModalClient/AppGetOrCreate",
      requestStream: false,
      responseStream: false,
    },
    request: {},
    requestStream: false,
    responseStream: false,
    next: nextFn,
  };
}

// --- RPCRetryPolicy tests ---

test("retryMiddleware: server-driven retries via RPCRetryPolicy succeed", async () => {
  const client = new ModalClient({
    tokenId: "test",
    tokenSecret: "test",
    logger: noopLogger,
  });
  const middleware = (client as any).retryMiddleware();

  const { call, getCallCount, getLastMetadata } = makeThrottlingMockCall(
    0.01,
    3,
  );

  // Drain the async generator (unary call yields exactly one response value).
  for await (const _ of middleware(call, {})) {
    // intentionally empty
  }

  expect(getCallCount()).toBe(4); // 3 throttle failures + 1 success

  const md = getLastMetadata()!;
  // Server-driven retries are tracked separately; client attempt stays at 0.
  expect(md.get("x-throttle-retry-attempt")).toBe("3");
  expect(md.get("x-retry-attempt")).toBe("0");
  // x-throttle-retry-delay is set once throttleRetries > 0.
  expect(md.get("x-throttle-retry-delay")).toBeTruthy();
  // x-retry-delay is only set when attempt > 0.
  expect(md.get("x-retry-delay")).toBeUndefined();
});

test("retryMiddleware: server-driven retries do not count against client retry limit", async () => {
  const client = new ModalClient({
    tokenId: "test",
    tokenSecret: "test",
    logger: noopLogger,
  });
  const middleware = (client as any).retryMiddleware();

  // One throttle retry followed by repeated UNAVAILABLE errors that exhaust the client limit.
  let callCount = 0;
  const trailerBytes = buildThrottleTrailer(0.01);
  // eslint-disable-next-line require-yield
  const call = makeMockCall(async function* (_req: unknown, opts: CallOptions) {
    callCount++;
    if (callCount === 1) {
      const trailer = new Metadata({ "grpc-status-details-bin": trailerBytes });
      opts.onTrailer?.(trailer);
      throw new ClientError(
        "/modal.client.ModalClient/AppGetOrCreate",
        Status.RESOURCE_EXHAUSTED,
        "throttled",
      );
    }
    throw new ClientError(
      "/modal.client.ModalClient/AppGetOrCreate",
      Status.UNAVAILABLE,
      "unavailable",
    );
  });

  await expect(
    (async () => {
      for await (const _ of middleware(call, { retries: 3, baseDelay: 1 })) {
        // intentionally empty
      }
    })(),
  ).rejects.toBeInstanceOf(ClientError);

  // 1 throttle + 4 client attempts (0 through 3) = 5 total calls.
  expect(callCount).toBe(5);
});

test("retryMiddleware: server-driven retry is cancelled when signal aborts", async () => {
  const client = new ModalClient({
    tokenId: "test",
    tokenSecret: "test",
    logger: noopLogger,
  });
  const middleware = (client as any).retryMiddleware();

  const controller = new AbortController();
  let callCount = 0;
  const trailerBytes = buildThrottleTrailer(60); // long delay

  // eslint-disable-next-line require-yield
  const call = makeMockCall(async function* (_req: unknown, opts: CallOptions) {
    callCount++;
    controller.abort(); // abort before the sleep can run
    const trailer = new Metadata({ "grpc-status-details-bin": trailerBytes });
    opts.onTrailer?.(trailer);
    throw new ClientError(
      "/modal.client.ModalClient/AppGetOrCreate",
      Status.RESOURCE_EXHAUSTED,
      "throttled",
    );
  });

  await expect(
    (async () => {
      for await (const _ of middleware(call, { signal: controller.signal })) {
        // intentionally empty
      }
    })(),
  ).rejects.toBeDefined();

  expect(callCount).toBe(1); // aborted during sleep, no second attempt
});

test("retryMiddleware: server-driven retry emits a warning log", async () => {
  const warnSpy = vi.fn();
  const logger: Logger = { ...noopLogger, warn: warnSpy };
  const client = new ModalClient({
    tokenId: "test",
    tokenSecret: "test",
    logger,
  });
  const middleware = (client as any).retryMiddleware();

  const { call } = makeThrottlingMockCall(0.01, 2);

  for await (const _ of middleware(call, {})) {
    // intentionally empty
  }

  expect(warnSpy).toHaveBeenCalledWith(
    "Server requested retry delay. Retrying...",
    "status",
    Status.RESOURCE_EXHAUSTED,
    "message",
    "server throttled",
    "method",
    "/modal.client.ModalClient/AppGetOrCreate",
  );
});

// --- maxThrottleWaitSecs tests ---

afterEach(() => vi.unstubAllEnvs());

test("retryMiddleware: maxThrottleWaitSecs exceeded by server delay throws immediately", async () => {
  const client = new ModalClient({
    tokenId: "test",
    tokenSecret: "test",
    logger: noopLogger,
    maxThrottleWaitSecs: 5,
  });
  const middleware = (client as any).retryMiddleware();

  // Server requests a 60-second delay; maxThrottleWaitSecs is 5 seconds.
  const { call, getCallCount } = makeThrottlingMockCall(60, 3);

  await expect(
    (async () => {
      for await (const _ of middleware(call, {})) {
        // intentionally empty
      }
    })(),
  ).rejects.toBeInstanceOf(ClientError);

  // Should fail on first attempt: 0 elapsed + 60s delay >= 5s limit.
  expect(getCallCount()).toBe(1);
});

test("retryMiddleware: maxThrottleWaitSecs allows retries that stay within limit", async () => {
  const client = new ModalClient({
    tokenId: "test",
    tokenSecret: "test",
    logger: noopLogger,
    maxThrottleWaitSecs: 60,
  });
  const middleware = (client as any).retryMiddleware();

  // Server requests 0.01s delay; maxThrottleWaitSecs is 60 seconds — plenty of room.
  const { call, getCallCount } = makeThrottlingMockCall(0.01, 2);

  for await (const _ of middleware(call, {})) {
    // intentionally empty
  }

  // 2 throttle failures + 1 success = 3 total calls.
  expect(getCallCount()).toBe(3);
});

test("retryMiddleware: maxThrottleWaitSecs=0 env var disables server-directed retries", async () => {
  const client = new ModalClient({
    tokenId: "test",
    tokenSecret: "test",
    logger: noopLogger,
    maxThrottleWaitSecs: 0,
  });
  const middleware = (client as any).retryMiddleware();

  const { call, getCallCount } = makeThrottlingMockCall(0.01, 3);

  await expect(
    (async () => {
      for await (const _ of middleware(call, {})) {
        // intentionally empty
      }
    })(),
  ).rejects.toBeInstanceOf(ClientError);

  expect(getCallCount()).toBe(1);
});

test("retryMiddleware: server-directed retries work by default", async () => {
  const client = new ModalClient({
    tokenId: "test",
    tokenSecret: "test",
    logger: noopLogger,
  });
  const middleware = (client as any).retryMiddleware();

  const { call, getCallCount } = makeThrottlingMockCall(0.01, 2);

  for await (const _ of middleware(call, {})) {
    // intentionally empty
  }

  // 2 throttle failures + 1 success = 3 total calls.
  expect(getCallCount()).toBe(3);
});

test("retryMiddleware: MODAL_MAX_THROTTLE_WAIT='' allows server-directed retries", async () => {
  vi.stubEnv("MODAL_MAX_THROTTLE_WAIT", "");

  const client = new ModalClient({
    tokenId: "test",
    tokenSecret: "test",
    logger: noopLogger,
  });
  const middleware = (client as any).retryMiddleware();

  const { call, getCallCount } = makeThrottlingMockCall(0.01, 2);

  for await (const _ of middleware(call, {})) {
    // intentionally empty
  }

  // 2 throttle failures + 1 success = 3 total calls.
  expect(getCallCount()).toBe(3);
});

test("retryMiddleware: MODAL_MAX_THROTTLE_WAIT=invalid allows server-directed retries", async () => {
  vi.stubEnv("MODAL_MAX_THROTTLE_WAIT", "not-a-number");

  const client = new ModalClient({
    tokenId: "test",
    tokenSecret: "test",
    logger: noopLogger,
  });
  const middleware = (client as any).retryMiddleware();

  const { call, getCallCount } = makeThrottlingMockCall(0.01, 2);

  for await (const _ of middleware(call, {})) {
    // intentionally empty
  }

  // 2 throttle failures + 1 success = 3 total calls.
  expect(getCallCount()).toBe(3);
});

test("ModalClient config has higher precedence", async () => {
  vi.stubEnv("MODAL_MAX_THROTTLE_WAIT", "30");

  const client = new ModalClient({
    tokenId: "test",
    tokenSecret: "test",
    logger: noopLogger,
    maxThrottleWaitSecs: 10,
  });

  expect(client.profile.maxThrottleWaitSecs).toBe(10);
});

// --- existing test ---

test("ModalClient with custom middleware", async () => {
  let firstCalled = false;
  let secondCalled = false;
  let firstMethod = "";
  let secondMethod = "";

  const firstMiddleware = async function* <Request, Response>(
    call: ClientMiddlewareCall<Request, Response>,
    options: CallOptions,
  ) {
    firstCalled = true;
    firstMethod = call.method.path;
    return yield* call.next(call.request, options);
  };

  const secondMiddleware = async function* <Request, Response>(
    call: ClientMiddlewareCall<Request, Response>,
    options: CallOptions,
  ) {
    secondCalled = true;
    secondMethod = call.method.path;
    return yield* call.next(call.request, options);
  };

  const mc = new ModalClient({
    grpcMiddleware: [firstMiddleware, secondMiddleware],
  });

  try {
    await mc.functions.fromName("libmodal-test-support", "non-existent");
  } catch (_err) {
    // Don't care about success here, just need the RPC to be made
  } finally {
    mc.close();
  }

  expect(firstCalled).toBe(true);
  expect(firstMethod).toContain("ModalClient/");
  expect(secondCalled).toBe(true);
  expect(secondMethod).toContain("ModalClient/");
});
