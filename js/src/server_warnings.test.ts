import { CallOptions, ClientError, Metadata, Status } from "nice-grpc";
import { expect, test, vi } from "vitest";
import type { Logger } from "./logger";
import {
  SERVER_WARNING_REGISTRY_LIMIT,
  logServerWarnings,
  serverWarningMiddleware,
} from "./server_warnings";

function makeLogger() {
  return {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  } satisfies Logger;
}

function makeCall(trailer: Metadata, err?: Error): any {
  return {
    method: {
      path: "/modal.client.ModalClient/AppGetOrCreate",
      requestStream: false,
      responseStream: false,
    },
    request: {},
    requestStream: false,
    responseStream: false,
    async *next(_request: unknown, options: CallOptions) {
      options.onTrailer?.(trailer);
      if (err) throw err;
      yield {};
    },
  };
}

async function drain(gen: AsyncIterable<unknown>) {
  for await (const _ of gen) {
    // Drain the response.
  }
}

test("serverWarningMiddleware logs decoded warnings", async () => {
  const logger = makeLogger();
  const trailer = new Metadata();
  trailer.append("x-modal-warning", "Hello%20world");
  trailer.append("x-modal-warning", "100%25%2C%20done");

  await drain(serverWarningMiddleware(logger)(makeCall(trailer), {}));

  expect(logger.warn.mock.calls).toEqual([["Hello world"], ["100%, done"]]);
});

test("serverWarningMiddleware logs warnings on failed calls", async () => {
  const logger = makeLogger();
  const trailer = new Metadata({ "x-modal-warning": "Something%20happened" });
  const rpcErr = new ClientError("/m", Status.INTERNAL, "boom");

  await expect(
    drain(serverWarningMiddleware(logger)(makeCall(trailer, rpcErr), {})),
  ).rejects.toBe(rpcErr);

  expect(logger.warn.mock.calls).toEqual([["Something happened"]]);
});

test("serverWarningMiddleware ignores other trailers", async () => {
  const logger = makeLogger();
  const trailer = new Metadata({ "x-other": "value" });

  await drain(serverWarningMiddleware(logger)(makeCall(trailer), {}));

  expect(logger.warn).not.toHaveBeenCalled();
});

test("serverWarningMiddleware logs malformed encodings verbatim", async () => {
  const logger = makeLogger();
  const trailer = new Metadata({ "x-modal-warning": "bad%E0%A4%A" });

  await drain(serverWarningMiddleware(logger)(makeCall(trailer), {}));

  expect(logger.warn.mock.calls).toEqual([["bad%E0%A4%A"]]);
});

test("serverWarningMiddleware forwards trailers to the caller", async () => {
  const logger = makeLogger();
  const trailer = new Metadata({ "x-modal-warning": "hi" });
  const onTrailer = vi.fn();

  await drain(
    serverWarningMiddleware(logger)(makeCall(trailer), { onTrailer }),
  );

  expect(onTrailer).toHaveBeenCalledWith(trailer);
});

test("serverWarningMiddleware logs a repeated warning once", async () => {
  const logger = makeLogger();
  const middleware = serverWarningMiddleware(logger);
  const trailer = new Metadata({ "x-modal-warning": "Token%20expires%20soon" });

  await drain(middleware(makeCall(trailer), {}));
  await drain(middleware(makeCall(trailer), {}));

  expect(logger.warn.mock.calls).toEqual([["Token expires soon"]]);
});

test("logServerWarnings keeps the registry bounded but never goes quiet", () => {
  // A server that varies the text of a warning grows the registry, but only up to the limit.
  const logger = makeLogger();
  const seen = new Set<string>();
  for (let i = 0; i < SERVER_WARNING_REGISTRY_LIMIT * 3; i++) {
    const trailer = new Metadata({
      "x-modal-warning": `warning about object ${i}`,
    });
    logServerWarnings(logger, trailer, seen);
  }
  // Overflow drops the entry for this message rather than silencing it.
  logServerWarnings(
    logger,
    new Metadata({ "x-modal-warning": "warning about object 0" }),
    seen,
  );

  expect(seen.size).toBeLessThanOrEqual(SERVER_WARNING_REGISTRY_LIMIT + 2);
  const messages = logger.warn.mock.calls.map(([m]) => m);
  expect(messages).toHaveLength(SERVER_WARNING_REGISTRY_LIMIT * 3 + 1);
  expect(messages.at(-1)).toBe("warning about object 0");
});
