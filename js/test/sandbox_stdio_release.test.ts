import { expect, test } from "vitest";
import net from "node:net";
import { createServer } from "nice-grpc";
import {
  TaskCommandRouterDefinition,
  TaskExecStartResponse,
  TaskExecStdioReadResponse,
  type TaskExecStdioReadRequest,
} from "../proto/modal_proto/task_command_router";
import { createMockModalClients } from "../test-support/grpc_mock";
// grpc-js reaches us through nice-grpc rather than as a direct dependency, and
// its subchannel pool is the only handle on the cleanup these tests force.
// eslint-disable-next-line import-x/no-extraneous-dependencies
import { getSubchannelPool } from "@grpc/grpc-js/build/src/subchannel-pool";

/**
 * These enter where callers do - sandbox.exec - and run against a local worker
 * over a real socket, with the control plane mocked. They exist because the
 * property under test is about what happens between the caller and the wire,
 * which is exactly what a test starting further down cannot see.
 */

const SANDBOX_ID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV";
const TASK_ID = "ta-01ARZ3NDEKTSV4RRFFQ69G5FAV";
const OUTPUT = new TextEncoder().encode(
  Array.from({ length: 5 }, (_, i) => `line-${i}\n`).join(""),
);
const CHUNK_SIZE = 7;
/** Short enough to keep tests quick, above grpc-js's 1s floor on the idle timeout. */
const IDLE_TIMEOUT_MS = 1500;
/** A caller's grace before their stream stops counting as in use. */
/**
 * grpc-js hands an idled channel's subchannel back to a pool that only sweeps
 * every REF_CHECK_INTERVAL (10s), so a socket outlives the channel going idle by
 * up to that long. Nothing in the channel options changes it -
 * `grpc.use_local_subchannel_pool` sweeps on the same interval.
 *
 * Waiting that out would put 10s on every test here, so the sweep is run by hand
 * instead. What that leaves under test is the part that is ours: whether the SDK
 * gives the connection up when it should. The library's delay in collecting it
 * afterwards is noted in the changelog rather than waited for.
 *
 * Safe to call from a test sharing a process with others: the sweep only
 * collects subchannels that nothing but the pool still references.
 */
function forceSubchannelSweep(): void {
  getSubchannelPool(true).unrefUnusedSubchannels();
}

function mockJwt(): string {
  const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  const exp = Math.floor(Date.now() / 1000) + 3600;
  return `${header}.${btoa(JSON.stringify({ exp }))}.signature`;
}

/**
 * Counts the connections the worker currently holds. Measured at the socket
 * rather than on the channel, because the channel is the SDK's private business
 * once a caller has a Sandbox.
 */
function countingProxy(targetPort: number): Promise<{
  port: number;
  live: () => number;
  close: () => Promise<void>;
}> {
  let live = 0;
  const sockets = new Set<net.Socket>();
  const server = net.createServer((incoming) => {
    live++;
    sockets.add(incoming);
    const upstream = net.connect(targetPort, "127.0.0.1");
    sockets.add(upstream);
    incoming.pipe(upstream);
    upstream.pipe(incoming);
    const done = () => {
      if (sockets.delete(incoming)) live--;
      incoming.destroy();
      upstream.destroy();
    };
    incoming.on("close", done);
    incoming.on("error", done);
    upstream.on("close", done);
    upstream.on("error", done);
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const port = (server.address() as net.AddressInfo).port;
      resolve({
        port,
        live: () => live,
        close: async () => {
          for (const s of sockets) s.destroy();
          await new Promise<void>((r) => server.close(() => r()));
        },
      });
    });
  });
}

/** Serves exec stdio from a byte offset, the way a worker seeks its stdio file. */
function fakeWorker(
  requestedOffsets: number[],
  chunksPerStream: number,
  active: { count: number },
) {
  const unimplemented: Record<string, unknown> = {};
  for (const methodName of Object.keys(TaskCommandRouterDefinition.methods)) {
    unimplemented[methodName] = () => {
      throw new Error(`${methodName} is not implemented in this test worker`);
    };
  }
  return {
    ...unimplemented,
    async taskExecStart() {
      return TaskExecStartResponse.create({});
    },
    async *taskExecStdioRead(
      request: TaskExecStdioReadRequest,
      context: { signal: AbortSignal },
    ) {
      let offset = Number(request.offset);
      requestedOffsets.push(offset);
      active.count++;
      try {
        let sent = 0;
        while (offset < OUTPUT.length) {
          // A chunk limit stops the whole output arriving in one stream, where
          // the transport would hand it over from its buffer and a reader would
          // never need to reopen.
          if (chunksPerStream > 0 && sent === chunksPerStream) break;
          sent++;
          const end = Math.min(offset + CHUNK_SIZE, OUTPUT.length);
          yield TaskExecStdioReadResponse.create({
            data: OUTPUT.subarray(offset, end),
          });
          offset = end;
        }
        // Hold the stream open, as a worker does for an exec still running.
        await new Promise<void>((resolve) => {
          if (context.signal.aborted) return resolve();
          context.signal.addEventListener("abort", () => resolve(), {
            once: true,
          });
        });
      } finally {
        active.count--;
      }
    },
  };
}

async function startFakeWorker(chunksPerStream = 0) {
  const requestedOffsets: number[] = [];
  // How many stdio handlers are running: a stream the client abandoned without
  // ending would show up here as one that never returns.
  const active = { count: 0 };
  const server = createServer();
  server.add(
    TaskCommandRouterDefinition,
    fakeWorker(requestedOffsets, chunksPerStream, active) as any,
  );
  const workerPort = await server.listen("127.0.0.1:0");
  const proxy = await countingProxy(workerPort);

  const { mockClient, mockCpClient } = createMockModalClients();
  // A localhost server URL is what makes the SDK dial the worker without TLS.
  mockClient.profile.serverUrl = "http://127.0.0.1:1";
  mockClient.profile.sandboxChannelIdleTimeoutMs = IDLE_TIMEOUT_MS;
  mockCpClient.handleUnary("SandboxGetTaskIdV2", () => ({ taskId: TASK_ID }));
  mockCpClient.handleUnary("SandboxGetCommandRouterAccess", () => ({
    url: `https://127.0.0.1:${proxy.port}`,
    jwt: mockJwt(),
  }));

  return {
    mockClient,
    requestedOffsets,
    activeStreams: () => active.count,
    live: proxy.live,
    shutdown: async () => {
      await proxy.close();
      server.forceShutdown();
    },
  };
}

async function waitFor(check: () => boolean, timeoutMs: number) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (check()) return true;
    forceSubchannelSweep();
    await new Promise((r) => globalThis.setTimeout(r, 20));
  }
  forceSubchannelSweep();
  return check();
}

test("a partial read of exec output releases the connection", async () => {
  const w = await startFakeWorker();
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"]);
    const reader = process.stdout.getReader();

    const first = await reader.read();
    expect(first.done).toBe(false);
    expect(w.live()).toBe(1);

    // Read once, then forget the Sandbox while keeping the reader referenced.
    const released = await waitFor(() => w.live() === 0, 6 * IDLE_TIMEOUT_MS);
    expect(reader).toBeDefined();
    expect(released).toBe(true);
  } finally {
    await w.shutdown();
  }
});

test("exec output resumes at the same offset after a release", async () => {
  // One chunk per stream, so the read after a release has to reopen rather than
  // being served from what the transport already buffered.
  const w = await startFakeWorker(1);
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"]);
    const reader = process.stdout.getReader();

    // Read a chunk, let the connection go, read the next. Waiting for the
    // release between reads matters: a read in flight holds the client, so a
    // reader blocked on a stream the worker is holding open would never see one.
    let received = "";
    for (let round = 0; round < 3; round++) {
      const next = await reader.read();
      expect(next.done).toBe(false);
      received += next.value as string;
      expect(await waitFor(() => w.live() === 0, 6 * IDLE_TIMEOUT_MS)).toBe(
        true,
      );
    }

    // Byte-for-byte equality is the real check on the resume offset: reopening
    // anywhere but the exact next byte would duplicate or drop output.
    const expected = new TextDecoder().decode(
      OUTPUT.subarray(0, received.length),
    );
    expect(received).toEqual(expected);
    expect(w.requestedOffsets).toEqual([0, CHUNK_SIZE, 2 * CHUNK_SIZE]);
  } finally {
    await w.shutdown();
  }
});

// The complement: a consumer that keeps up is never interrupted, so a whole
// read costs exactly one stream.
test("a prompt consumer keeps one stream throughout", async () => {
  const w = await startFakeWorker();
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"]);

    let received = "";
    const expected = new TextDecoder().decode(OUTPUT);
    const reader = process.stdout.getReader();
    while (received.length < expected.length) {
      const next = await reader.read();
      if (next.done) break;
      received += next.value as string;
    }

    expect(received).toEqual(expected);
    expect(w.requestedOffsets).toEqual([0]);
  } finally {
    await w.shutdown();
  }
});

// Detaching ends a Sandbox for good: unlike an idle release, nothing picks it
// up again.
test("a detached Sandbox does not come back", async () => {
  const w = await startFakeWorker();
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"]);
    const reader = process.stdout.getReader();
    expect((await reader.read()).done).toBe(false);

    sandbox.detach();
    expect(await waitFor(() => w.live() === 0, 4 * IDLE_TIMEOUT_MS)).toBe(true);

    // Unlike an idle release, nothing reconnects afterwards.
    await expect(sandbox.exec(["echo", "again"])).rejects.toThrow();
  } finally {
    await w.shutdown();
  }
});

// A stream held from before the detach is dead too, and says so rather than
// hanging or quietly ending.
test("a stream held across a detach reports the detach", async () => {
  const w = await startFakeWorker(1);
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"]);
    const reader = process.stdout.getReader();
    expect((await reader.read()).done).toBe(false);

    sandbox.detach();

    await expect(reader.read()).rejects.toThrow(
      "Unable to perform operation on a detached sandbox",
    );
  } finally {
    await w.shutdown();
  }
});

// A consumer can walk away from a stream by cancelling rather than by going
// quiet, and that path has to end the stream too.
test("cancelling a read ends the stream behind it", async () => {
  const w = await startFakeWorker(1);
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"]);
    const reader = process.stdout.getReader();
    expect((await reader.read()).done).toBe(false);
    expect(w.activeStreams()).toBe(1);

    await reader.cancel();

    expect(await waitFor(() => w.activeStreams() === 0, 2000)).toBe(true);
  } finally {
    await w.shutdown();
  }
});
