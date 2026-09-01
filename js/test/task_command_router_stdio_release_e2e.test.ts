import { expect, test, vi } from "vitest";
import { createChannel, createClientFactory, createServer } from "nice-grpc";
import {
  TaskCommandRouterDefinition,
  TaskExecStdioReadResponse,
  type TaskExecStdioReadRequest,
} from "../proto/modal_proto/task_command_router";
import { FileDescriptor } from "../proto/modal_proto/api";
import { TaskCommandRouterClientImpl } from "../src/task_command_router_client";
import { streamConsumingIter } from "../src/streams";
import { ContainerProcess } from "../src/sandbox";

/**
 * These run against a real grpc-js server over a real socket, so they can say
 * what happened to the channel rather than only to the RPC. The fake-stub tests
 * alongside them cover the resume bookkeeping.
 *
 * They read the channel's connectivity state, which reaches IDLE when the SDK
 * gives the connection up - some seconds before grpc-js gets round to closing
 * the socket underneath it. For the socket itself, see
 * `sandbox_stdio_release.test.ts`, which counts connections at the worker.
 */

const mockLogger = {
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
};

const OUTPUT = new TextEncoder().encode(
  Array.from({ length: 10 }, (_, i) => `line-${i}\n`).join(""),
);
const CHUNK_SIZE = 7;
/**
 * grpc-js's ConnectivityState.IDLE. Spelled out rather than imported, because
 * grpc-js reaches us through nice-grpc rather than as a direct dependency.
 */
const CONNECTIVITY_STATE_IDLE = 0;
/** Small enough to keep tests quick, above grpc-js's 1s floor on the idle stage. */
const IDLE_BUDGET_MS = 1500;

function mockJwt(exp: number): string {
  const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  return `${header}.${btoa(JSON.stringify({ exp }))}.fake-signature`;
}

/** Serves exec stdio from a byte offset, the way the worker seeks its stdio file. */
function stdioServiceImpl(requestedOffsets: number[]) {
  // The server wants an implementation for every method on the definition, so
  // fill the rest in with stubs that fail loudly if a test reaches them.
  const unimplemented: Record<string, unknown> = {};
  for (const methodName of Object.keys(TaskCommandRouterDefinition.methods)) {
    unimplemented[methodName] = () => {
      throw new Error(`${methodName} is not implemented in this test server`);
    };
  }

  return {
    ...unimplemented,
    async *taskExecStdioRead(
      request: TaskExecStdioReadRequest,
      context: { signal: AbortSignal },
    ) {
      let offset = Number(request.offset);
      requestedOffsets.push(offset);
      while (offset < OUTPUT.length) {
        const end = Math.min(offset + CHUNK_SIZE, OUTPUT.length);
        yield TaskExecStdioReadResponse.create({
          data: OUTPUT.subarray(offset, end),
        });
        offset = end;
      }
      // Hold the stream open until the client gives up on it.
      await new Promise<void>((resolve) => {
        if (context.signal.aborted) {
          resolve();
          return;
        }
        context.signal.addEventListener("abort", () => resolve(), {
          once: true,
        });
      });
    },
  };
}

type Harness = {
  client: any;
  channel: ReturnType<typeof createChannel>;
  requestedOffsets: number[];
  shutdown: () => Promise<void>;
};

async function startHarness(idleBudgetMs: number): Promise<Harness> {
  const requestedOffsets: number[] = [];
  const server = createServer();
  server.add(
    TaskCommandRouterDefinition,
    stdioServiceImpl(requestedOffsets) as any,
  );
  const port = await server.listen("127.0.0.1:0");

  // The stream timeout is a fraction of the connection one, so a test can stall
  // past the first without waiting out the second.
  const handoffMs = Math.floor(idleBudgetMs / 10);
  const channel = createChannel(
    `127.0.0.1:${port}`,
    undefined,
    idleBudgetMs > 0 ? { "grpc.client_idle_timeout_ms": idleBudgetMs } : {},
  );

  const client = Object.create(TaskCommandRouterClientImpl.prototype) as any;
  client.stub = createClientFactory().create(
    TaskCommandRouterDefinition,
    channel,
  );
  client.logger = mockLogger;
  client.closed = false;
  client.jwt = mockJwt(Math.floor(Date.now() / 1000) + 3600);
  client.stdioHandoffTimeoutMs = handoffMs;

  return {
    client,
    channel,
    requestedOffsets,
    shutdown: async () => {
      channel.close();
      // Forced, not graceful: these tests deliberately leave a read open on the
      // server, which a graceful shutdown would wait on for ever.
      server.forceShutdown();
    },
  };
}

function readStdio(client: any) {
  return client.execStdioRead(
    "ta-1",
    "ex-1",
    FileDescriptor.FILE_DESCRIPTOR_STDOUT,
    null,
  );
}

async function waitFor(
  predicate: () => boolean,
  timeoutMs: number,
): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (predicate()) return true;
    await new Promise((resolve) => globalThis.setTimeout(resolve, 10));
  }
  return predicate();
}

function isIdle(channel: ReturnType<typeof createChannel>): boolean {
  return channel.getConnectivityState(false) === CONNECTIVITY_STATE_IDLE;
}

test("a partial read of a then-forgotten Sandbox takes its channel idle", async () => {
  const h = await startHarness(IDLE_BUDGET_MS);
  try {
    const stream = readStdio(h.client);
    const first = await stream.next();
    expect(first.done).toBe(false);
    expect(isIdle(h.channel)).toBe(false);

    // Never ask for another chunk, and keep the iterator referenced throughout,
    // so nothing here can be attributed to it being collected.
    const released = await waitFor(() => isIdle(h.channel), 4 * IDLE_BUDGET_MS);
    expect(stream).toBeDefined();
    expect(released).toBe(true);
  } finally {
    await h.shutdown();
  }
});

test("the channel goes idle one timeout after a partial read", async () => {
  const h = await startHarness(IDLE_BUDGET_MS);
  try {
    const stream = readStdio(h.client);
    await stream.next();

    const start = Date.now();
    const released = await waitFor(() => isIdle(h.channel), 4 * IDLE_BUDGET_MS);
    const elapsed = Date.now() - start;

    expect(released).toBe(true);
    // Giving up on the consumer and dropping the transport happen one after the
    // other, so they share the configured budget rather than each taking it.
    expect(elapsed).toBeGreaterThanOrEqual(IDLE_BUDGET_MS / 2);
    expect(elapsed).toBeLessThan(2 * IDLE_BUDGET_MS);
  } finally {
    await h.shutdown();
  }
});

test("a partial read resumed later continues at the same offset", async () => {
  const h = await startHarness(IDLE_BUDGET_MS);
  try {
    const stream = readStdio(h.client);
    const chunks: Uint8Array[] = [];
    chunks.push((await stream.next()).value.data);

    // Stall until the connection has actually gone, so the resume below is
    // reopening from nothing rather than riding a stream that stayed open.
    // Waiting on the reopen itself would deadlock: it cannot happen until this
    // consumer asks for the next chunk.
    expect(await waitFor(() => isIdle(h.channel), 4 * IDLE_BUDGET_MS)).toBe(
      true,
    );

    let total = chunks.reduce((n, c) => n + c.length, 0);
    while (total < OUTPUT.length) {
      const next = await stream.next();
      if (next.done) break;
      chunks.push(next.value.data);
      total += next.value.data.length;
    }

    const received = new Uint8Array(total);
    let at = 0;
    for (const c of chunks) {
      received.set(c, at);
      at += c.length;
    }
    // Byte-for-byte equality is the real check on the resume offset: reopening
    // anywhere but the exact next byte would duplicate or drop output.
    expect(received).toEqual(OUTPUT);
    expect(h.requestedOffsets.length).toBeGreaterThan(1);
    expect(h.requestedOffsets[0]).toBe(0);
  } finally {
    await h.shutdown();
  }
});

test("many partly read Sandboxes all take their channels idle", async () => {
  const harnesses: Harness[] = [];
  try {
    for (let i = 0; i < 5; i++) {
      harnesses.push(await startHarness(IDLE_BUDGET_MS));
    }
    // Held for the whole test, so nothing below is down to collection.
    const streams = harnesses.map((h) => readStdio(h.client));
    for (const stream of streams) {
      expect((await stream.next()).done).toBe(false);
    }
    expect(harnesses.every((h) => !isIdle(h.channel))).toBe(true);

    const allReleased = await waitFor(
      () => harnesses.every((h) => isIdle(h.channel)),
      4 * IDLE_BUDGET_MS,
    );
    expect(streams).toHaveLength(harnesses.length);
    expect(allReleased).toBe(true);
  } finally {
    for (const h of harnesses) await h.shutdown();
  }
});

// The tests above drive the stdio generator directly. Callers do not: they read
// the ReadableStream that `stdout` hands them, which is what these cover.

/** Builds the reader the way ContainerProcess does, pulling only when read. */
function readStdioStream(client: any): ReadableStream<Uint8Array> {
  async function* bytes() {
    for await (const batch of readStdio(client)) {
      yield batch.data;
    }
  }
  return streamConsumingIter(bytes());
}

test("a partial read of the stdout stream takes the channel idle", async () => {
  const h = await startHarness(IDLE_BUDGET_MS);
  try {
    const reader = readStdioStream(h.client).getReader();
    const first = await reader.read();
    expect(first.done).toBe(false);
    expect(isIdle(h.channel)).toBe(false);

    // Read once, then walk away while keeping the reader referenced.
    const released = await waitFor(() => isIdle(h.channel), 4 * IDLE_BUDGET_MS);
    expect(reader).toBeDefined();
    expect(released).toBe(true);
  } finally {
    await h.shutdown();
  }
});

test("the stdout stream resumes at the same offset after a release", async () => {
  const h = await startHarness(IDLE_BUDGET_MS);
  try {
    const reader = readStdioStream(h.client).getReader();
    const chunks: Uint8Array[] = [];
    chunks.push((await reader.read()).value!);

    expect(await waitFor(() => isIdle(h.channel), 4 * IDLE_BUDGET_MS)).toBe(
      true,
    );

    let total = chunks.reduce((n, c) => n + c.length, 0);
    while (total < OUTPUT.length) {
      const next = await reader.read();
      if (next.done) break;
      chunks.push(next.value);
      total += next.value.length;
    }

    const received = new Uint8Array(total);
    let at = 0;
    for (const c of chunks) {
      received.set(c, at);
      at += c.length;
    }
    expect(received).toEqual(OUTPUT);
    expect(h.requestedOffsets.length).toBeGreaterThan(1);
  } finally {
    await h.shutdown();
  }
});

// And these go through ContainerProcess itself, the object callers are handed.
// Text mode - the default - pipes the byte stream through a TextDecoderStream,
// which is another place read-ahead could hide.

test("a partial read of ContainerProcess.stdout takes the channel idle", async () => {
  const h = await startHarness(IDLE_BUDGET_MS);
  try {
    const process = new ContainerProcess("ta-1", "ex-1", h.client, {}, null);
    const reader = process.stdout.getReader();
    const first = await reader.read();
    expect(first.done).toBe(false);
    expect(typeof first.value).toBe("string");
    expect(isIdle(h.channel)).toBe(false);

    const released = await waitFor(() => isIdle(h.channel), 4 * IDLE_BUDGET_MS);
    expect(reader).toBeDefined();
    expect(released).toBe(true);
  } finally {
    await h.shutdown();
  }
});

test("ContainerProcess.stdout resumes at the same offset after a release", async () => {
  const h = await startHarness(IDLE_BUDGET_MS);
  try {
    const process = new ContainerProcess("ta-1", "ex-1", h.client, {}, null);
    const reader = process.stdout.getReader();
    let received = (await reader.read()).value as string;

    expect(await waitFor(() => isIdle(h.channel), 4 * IDLE_BUDGET_MS)).toBe(
      true,
    );

    const expected = new TextDecoder().decode(OUTPUT);
    while (received.length < expected.length) {
      const next = await reader.read();
      if (next.done) break;
      received += next.value as string;
    }

    expect(received).toEqual(expected);
    expect(h.requestedOffsets.length).toBeGreaterThan(1);
  } finally {
    await h.shutdown();
  }
});
