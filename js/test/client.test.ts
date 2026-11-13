import {
  ClientMiddlewareCall,
  CallOptions,
  ClientError,
  createChannel,
  createServer,
  Metadata,
  ServerError,
  Status,
  type CallContext,
} from "nice-grpc";
import { ModalClient, type Logger } from "modal";
import {
  ModalClientDefinition,
  RPCRetryPolicy,
  RPCStatus,
} from "../proto/modal_proto/api";
import { Any } from "../proto/google/protobuf/any";
import { connectivityState } from "@grpc/grpc-js";
import { connectFirst } from "../src/client";
import { generateKeyPairSync } from "node:crypto";
import { once } from "node:events";
import net from "node:net";
import jwt from "jsonwebtoken";
import { afterEach, expect, onTestFinished, test, vi } from "vitest";

// --- helpers for RPCRetryPolicy tests ---

const noopLogger: Logger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
};

const testJwtKeyPair = generateKeyPairSync("rsa", { modulusLength: 2048 });
const testJwtKeyPem = testJwtKeyPair.privateKey
  .export({ type: "pkcs8", format: "pem" })
  .toString();
const testJwtPublicKeyPem = testJwtKeyPair.publicKey.export({
  type: "spki",
  format: "pem",
});

test("authMiddleware sends OAuth credentials without token credentials", async () => {
  const client = new ModalClient({
    oauthRefreshToken: "refresh-token",
    oauthClientId: "oc-client-id",
    oauthClientSecret: "ov-client-secret",
    logger: noopLogger,
    cpClient: {} as any,
  });
  // authMiddleware takes the profile of a single server, not a list of fallbacks.
  const middleware = (client as any).authMiddleware({
    ...client.profile,
    serverUrl: "https://api.modal.com:443",
  });
  let metadata: Metadata | undefined;
  const call = makeMockCall(async function* (
    _request: unknown,
    options: CallOptions,
  ) {
    metadata = options.metadata as Metadata;
    yield {};
  });
  call.method.path = "/modal.client.ModalClient/AuthTokenGet";

  for await (const _ of middleware(call, {})) {
    // Drain the unary response.
  }

  expect(metadata?.get("x-modal-refresh-token")).toBe("refresh-token");
  expect(metadata?.get("x-modal-oauth-client-id")).toBe("oc-client-id");
  expect(metadata?.get("x-modal-oauth-client-secret")).toBe("ov-client-secret");
  expect(metadata?.get("x-modal-token-id")).toBeUndefined();
  expect(metadata?.get("x-modal-token-secret")).toBeUndefined();
});

test("authMiddleware sends a client assertion instead of a client secret", async () => {
  const client = new ModalClient({
    oauthRefreshToken: "refresh-token",
    oauthClientId: "oc-client-id",
    oauthJwtKey: testJwtKeyPem,
    logger: noopLogger,
    cpClient: {} as any,
  });
  // authMiddleware takes the profile of a single server, not a list of fallbacks.
  const middleware = (client as any).authMiddleware({
    ...client.profile,
    serverUrl: "https://api.modal.com:443",
  });
  let metadata: Metadata | undefined;
  const call = makeMockCall(async function* (
    _request: unknown,
    options: CallOptions,
  ) {
    metadata = options.metadata as Metadata;
    yield {};
  });
  call.method.path = "/modal.client.ModalClient/AuthTokenGet";

  for await (const _ of middleware(call, {})) {
    // Drain the unary response.
  }

  expect(metadata?.get("x-modal-refresh-token")).toBe("refresh-token");
  expect(metadata?.get("x-modal-oauth-client-id")).toBe("oc-client-id");
  expect(metadata?.get("x-modal-oauth-client-secret")).toBeUndefined();
  const assertion = metadata?.get("x-modal-oauth-client-assertion");
  expect(assertion).toBeDefined();
  expect(
    jwt.verify(assertion!, testJwtPublicKeyPem, {
      algorithms: ["RS256"],
      audience: "modal-server",
      issuer: "oc-client-id",
    }),
  ).toBeTruthy();
});

test("ModalClient rejects a malformed OAuth JWT key", () => {
  expect(
    () =>
      new ModalClient({
        oauthRefreshToken: "refresh-token",
        oauthClientId: "oc-client-id",
        oauthJwtKey: "not-a-key",
        logger: noopLogger,
        cpClient: {} as any,
      }),
  ).toThrow("must be an unencrypted RSA private key");
});

test.each([
  [{ oauthRefreshToken: "refresh-token" }, "must all be configured"],
  [
    {
      oauthRefreshToken: "refresh-token",
      oauthClientId: "oc-client-id",
      oauthClientSecret: "ov-client-secret",
      oauthJwtKey: testJwtKeyPem,
    },
    "exactly one of client secret or JWT key",
  ],
  [
    {
      oauthRefreshToken: "",
      oauthClientId: "oc-client-id",
      oauthClientSecret: "ov-client-secret",
    },
    "must all be configured",
  ],
  [
    {
      tokenId: "ak-token-id",
      tokenSecret: "as-token-secret",
      oauthRefreshToken: "refresh-token",
      oauthClientId: "oc-client-id",
      oauthClientSecret: "ov-client-secret",
    },
    "cannot both be configured",
  ],
])(
  "ModalClient rejects invalid credential configuration",
  (params, message) => {
    expect(
      () =>
        new ModalClient({ ...params, logger: noopLogger, cpClient: {} as any }),
    ).toThrow(message);
  },
);

test("explicit OAuth credentials do not backfill from the profile", () => {
  vi.stubEnv("MODAL_OAUTH_REFRESH_TOKEN", "profile-refresh-token");
  vi.stubEnv("MODAL_OAUTH_CLIENT_ID", "profile-client-id");
  vi.stubEnv("MODAL_OAUTH_CLIENT_SECRET", "profile-client-secret");

  try {
    expect(
      () =>
        new ModalClient({
          oauthRefreshToken: "explicit-refresh-token",
          logger: noopLogger,
          cpClient: {} as any,
        }),
    ).toThrow("must all be configured");
  } finally {
    vi.unstubAllEnvs();
  }
});

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

test("authMiddleware: injects the endpoint hostname as x-modal-host", async () => {
  const client = new ModalClient({
    tokenId: "test",
    tokenSecret: "test",
    logger: noopLogger,
  });
  const middleware = (client as any).authMiddleware({
    ...client.profile,
    serverUrl: "https://api.modal.com:443",
  });

  let metadata: Metadata | undefined;
  // Use AuthTokenGet so the middleware skips the auth token fetch, which would hit the network.
  const call = makeMockCall(async function* (
    _request: unknown,
    options: CallOptions,
  ) {
    metadata = options.metadata as Metadata;
    yield { token: "" };
  });
  call.method.path = "/modal.client.ModalClient/AuthTokenGet";

  for await (const _ of middleware(call, {})) {
    // intentionally empty
  }
  client.close();

  expect(metadata!.get("x-modal-host")).toBe("api.modal.com");
});

// --- server URL fallback tests ---

/** Head start each server URL gets over the next one, as the client sets it. */
const FALLBACK_STAGGER_MS = 3_000;

/** A URL that refuses connections: nothing listens on port 1. */
const REFUSED_URL = "http://127.0.0.1:1";

/**
 * Serves the Modal API on a local port and records the x-modal-host header of every call. Calls
 * other than AuthTokenGet fail with PERMISSION_DENIED, which the client does not retry.
 */
async function startFakeServer() {
  const hosts: string[] = [];
  const deny = async (_: unknown, { metadata }: CallContext) => {
    hosts.push(metadata.get("x-modal-host") ?? "");
    throw new ServerError(Status.PERMISSION_DENIED, "fake server");
  };
  async function* denyStream(req: unknown, ctx: CallContext) {
    yield await deny(req, ctx);
  }
  const impl: Record<string, unknown> = {
    authTokenGet: async () => ({ token: "token" }),
  };
  for (const [name, m] of Object.entries(ModalClientDefinition.methods)) {
    impl[name] ??= m.responseStream ? denyStream : deny;
  }
  const server = createServer();
  server.add(ModalClientDefinition, impl as any);
  const port = await server.listen("127.0.0.1:0");
  onTestFinished(() => server.forceShutdown());
  return { url: `http://127.0.0.1:${port}`, hosts };
}

/** Returns a URL on a local port that accepts TCP connections but never responds. */
async function blackholeUrl() {
  const sockets: net.Socket[] = [];
  const server = net.createServer((socket) => sockets.push(socket));
  await once(server.listen(0, "127.0.0.1"), "listening");
  onTestFinished(() => {
    sockets.forEach((socket) => socket.destroy());
    server.close();
  });
  return `http://127.0.0.1:${(server.address() as net.AddressInfo).port}`;
}

/** `url`, on 127.0.0.1, under the host name "localhost" instead. */
const localhostUrl = (url: string) => url.replace("127.0.0.1", "localhost");

/** Creates a client for the given server URLs. */
function clientFor(...urls: string[]) {
  vi.stubEnv("MODAL_SERVER_URL", urls.join(","));
  const creds = { tokenId: "ak-test", tokenSecret: "as-test" };
  return new ModalClient({ ...creds, logger: noopLogger });
}

/** Makes a request through `client` that a fake server rejects. */
const request = (client: ModalClient) =>
  expect(client.secrets.fromName("x")).rejects.toThrow("fake server");

const drain = async (stream: AsyncIterable<unknown>) => {
  for await (const _ of stream);
};

const unreachablePrimaries = {
  "connection refused": async () => REFUSED_URL,
  "DNS failure": async () => "http://modal-test-host.invalid:443",
  "TLS to a plaintext server": async () =>
    (await startFakeServer()).url.replace("http:", "https:"),
  blackholed: blackholeUrl,
};

for (const [name, primary] of Object.entries(unreachablePrimaries)) {
  test(`server URL fallback: fails over when primary has ${name}`, async () => {
    const secondary = await startFakeServer();
    const start = Date.now();
    await request(clientFor(await primary(), localhostUrl(secondary.url)));
    // Only a silent primary makes the secondary wait out its stagger.
    const waited = Date.now() - start >= FALLBACK_STAGGER_MS;
    expect(waited).toBe(name === "blackholed");
    expect(new Set(secondary.hosts)).toEqual(new Set(["localhost"]));
  });
}

test("server URL fallback: prefers primary when reachable", async () => {
  const primary = await startFakeServer();
  const secondary = await startFakeServer();
  // Whitespace around the separator is ignored.
  await request(clientFor(primary.url, ` ${localhostUrl(secondary.url)}`));
  expect(new Set(primary.hosts)).toEqual(new Set(["127.0.0.1"]));
  expect(secondary.hosts).toEqual([]);
});

test("server URL fallback: uses the third URL when the first two fail", async () => {
  const third = await startFakeServer();
  await request(clientFor(REFUSED_URL, REFUSED_URL, third.url));
  expect(third.hosts).not.toEqual([]);
});

test("server URL fallback: reuses the winning connection for later requests", async () => {
  const secondary = await startFakeServer();
  const client = clientFor(REFUSED_URL, secondary.url);
  await request(client);
  const start = Date.now();
  await request(client);
  expect(Date.now() - start).toBeLessThan(FALLBACK_STAGGER_MS);
  expect(secondary.hosts).toHaveLength(2);
});

test("server URL fallback: streaming calls go to the winner", async () => {
  const secondary = await startFakeServer();
  const client = clientFor(REFUSED_URL, secondary.url);
  const logs = client.cpClient.appGetLogs({ appId: "ap-test" });
  await expect(drain(logs)).rejects.toThrow("fake server");
  expect(secondary.hosts).toHaveLength(1);
});

test("server URL fallback: timeoutMs bounds the wait for a connection", async () => {
  const blackhole = await blackholeUrl();
  const client = clientFor(blackhole, blackhole);
  const call = client.cpClient.appGetOrCreate({}, { timeoutMs: 200 });
  await expect(call).rejects.toMatchObject({
    code: Status.DEADLINE_EXCEEDED,
    path: "/modal.client.ModalClient/AppGetOrCreate",
    details: "Timed out after 200ms",
  });
});

test("server URL fallback: an aborted signal ends the wait for a connection", async () => {
  const blackhole = await blackholeUrl();
  const client = clientFor(blackhole, blackhole);
  const signal = AbortSignal.timeout(200);
  const call = client.cpClient.appGetOrCreate({}, { signal });
  expect(await call.catch((err) => err)).toBe(signal.reason);
  // A signal that is already aborted ends a streaming call's wait right away.
  const logs = client.cpClient.appGetLogs({}, { signal });
  expect(await drain(logs).catch((err) => err)).toBe(signal.reason);
});

test("server URL fallback: a timed-out call leaves the race running for later calls", async () => {
  const secondary = await startFakeServer();
  const client = clientFor(await blackholeUrl(), secondary.url);
  const call = client.cpClient.appGetOrCreate({}, { timeoutMs: 200 });
  await expect(call).rejects.toThrow("Timed out after 200ms");
  // The second call waits out the rest of the race the first one started.
  await request(client);
  expect(secondary.hosts).toHaveLength(1);
});

test("connectFirst: rejects and closes every channel when none becomes ready", async () => {
  const channels = [createChannel(REFUSED_URL), createChannel(REFUSED_URL)];
  await expect(connectFirst(channels, 500)).rejects.toThrow(AggregateError);
  const states = channels.map((c) => c.getConnectivityState(false));
  expect(states).toEqual([
    connectivityState.SHUTDOWN,
    connectivityState.SHUTDOWN,
  ]);
});
