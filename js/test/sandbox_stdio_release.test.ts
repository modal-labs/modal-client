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
const STREAM_IDLE_TIMEOUT_MS = 150;
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
function fakeWorker(requestedOffsets: number[]) {
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
      while (offset < OUTPUT.length) {
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
    },
  };
}

async function startFakeWorker() {
  const requestedOffsets: number[] = [];
  const server = createServer();
  server.add(TaskCommandRouterDefinition, fakeWorker(requestedOffsets) as any);
  const workerPort = await server.listen("127.0.0.1:0");
  const proxy = await countingProxy(workerPort);

  const { mockClient, mockCpClient } = createMockModalClients();
  // A localhost server URL is what makes the SDK dial the worker without TLS.
  mockClient.profile.serverUrl = "http://127.0.0.1:1";
  mockClient.profile.sandboxChannelIdleTimeoutMs = IDLE_TIMEOUT_MS;
  mockClient.profile.sandboxStreamIdleTimeoutMs = STREAM_IDLE_TIMEOUT_MS;
  mockCpClient.handleUnary("SandboxGetTaskIdV2", () => ({ taskId: TASK_ID }));
  mockCpClient.handleUnary("SandboxGetCommandRouterAccess", () => ({
    url: `https://127.0.0.1:${proxy.port}`,
    jwt: mockJwt(),
  }));

  return {
    mockClient,
    requestedOffsets,
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
  const w = await startFakeWorker();
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"]);
    const reader = process.stdout.getReader();

    let received = (await reader.read()).value as string;
    expect(await waitFor(() => w.live() === 0, 6 * IDLE_TIMEOUT_MS)).toBe(true);

    const expected = new TextDecoder().decode(OUTPUT);
    while (received.length < expected.length) {
      const next = await reader.read();
      if (next.done) break;
      received += next.value as string;
    }

    // Byte-for-byte equality is the real check on the resume offset: reopening
    // anywhere but the exact next byte would duplicate or drop output.
    expect(received).toEqual(expected);
    expect(w.requestedOffsets.length).toBeGreaterThan(1);
    expect(w.requestedOffsets[0]).toBe(0);
  } finally {
    await w.shutdown();
  }
});
