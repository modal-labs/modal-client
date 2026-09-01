import { expect, test, vi } from "vitest";
import { TaskCommandRouterClientImpl } from "../src/task_command_router_client";
import {
  TaskExecStdioReadResponse,
  type TaskExecStdioReadRequest,
} from "../proto/modal_proto/task_command_router";
import { FileDescriptor } from "../proto/modal_proto/api";

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

/** Serves exec stdio from a byte offset, the way the worker seeks its stdio file. */
class FakeStdioServer {
  requestedOffsets: number[] = [];
  aborts = 0;

  stub() {
    // eslint-disable-next-line @typescript-eslint/no-this-alias -- generator functions cannot be arrows
    const server = this;
    return {
      async *taskExecStdioRead(
        request: TaskExecStdioReadRequest,
        options?: { signal?: AbortSignal },
      ) {
        let offset = Number(request.offset);
        server.requestedOffsets.push(offset);

        // Counted from a listener rather than a poll: the client aborts while
        // this generator is suspended at a yield, so nothing here would run.
        let aborted = false;
        const onAbort = () => {
          aborted = true;
          server.aborts++;
        };
        if (options?.signal?.aborted) {
          onAbort();
        } else {
          options?.signal?.addEventListener("abort", onAbort, { once: true });
        }

        while (offset < OUTPUT.length && !aborted) {
          const end = Math.min(offset + CHUNK_SIZE, OUTPUT.length);
          yield TaskExecStdioReadResponse.create({
            data: OUTPUT.subarray(offset, end),
          });
          offset = end;
        }
        if (aborted) {
          return;
        }
        // Hold the stream open until the caller gives up on it.
        await new Promise<void>((resolve) => {
          if (options?.signal?.aborted) {
            resolve();
            return;
          }
          options?.signal?.addEventListener("abort", () => resolve(), {
            once: true,
          });
        });
      },
    };
  }
}

function makeStdioClient(server: FakeStdioServer, handoffMs: number): any {
  const client = Object.create(TaskCommandRouterClientImpl.prototype) as any;
  client.stub = server.stub();
  client.logger = mockLogger;
  client.closed = false;
  client.stdioHandoffTimeoutMs = handoffMs;
  return client;
}

function concat(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

test("a prompt consumer keeps one stream throughout", async () => {
  const server = new FakeStdioServer();
  const client = makeStdioClient(server, 5000);

  const chunks: Uint8Array[] = [];
  for await (const item of client.execStdioRead(
    "ta-1",
    "ex-1",
    FileDescriptor.FILE_DESCRIPTOR_STDOUT,
    null,
  )) {
    chunks.push(item.data);
    if (concat(chunks).length >= OUTPUT.length) break;
  }

  expect(concat(chunks)).toEqual(OUTPUT);
  expect(server.requestedOffsets).toEqual([0]);
});

test("a stalled consumer gives the connection up and resumes at its offset", async () => {
  const server = new FakeStdioServer();
  const client = makeStdioClient(server, 50);

  const chunks: Uint8Array[] = [];
  let stalledOnce = false;
  for await (const item of client.execStdioRead(
    "ta-1",
    "ex-1",
    FileDescriptor.FILE_DESCRIPTOR_STDOUT,
    null,
  )) {
    chunks.push(item.data);
    if (!stalledOnce) {
      // Sit on the chunk for longer than the hand-off timeout.
      stalledOnce = true;
      await new Promise((resolve) => globalThis.setTimeout(resolve, 150));
    }
    if (concat(chunks).length >= OUTPUT.length) break;
  }

  // Byte-for-byte equality is the real check on the resume offset: reopening
  // anywhere but the exact next byte would duplicate or drop output.
  expect(concat(chunks)).toEqual(OUTPUT);
  expect(server.requestedOffsets.length).toBeGreaterThan(1);
  expect(server.requestedOffsets[0]).toBe(0);
  expect(server.requestedOffsets[1]).toBeGreaterThan(0);
  expect(server.aborts).toBeGreaterThan(0);
});

test("a consumer that walks away does not hold the read open", async () => {
  const server = new FakeStdioServer();
  const client = makeStdioClient(server, 50);

  const stream = client.execStdioRead(
    "ta-1",
    "ex-1",
    FileDescriptor.FILE_DESCRIPTOR_STDOUT,
    null,
  );
  const first = await stream.next();
  expect(first.done).toBe(false);

  // Never ask for another chunk. The hand-off timer still fires, which aborts
  // the read so it stops counting as an in-flight RPC.
  await new Promise((resolve) => globalThis.setTimeout(resolve, 200));
  expect(server.aborts).toBeGreaterThan(0);

  await stream.return?.(undefined);
});

test("a zero hand-off timeout keeps the stream open", async () => {
  const server = new FakeStdioServer();
  const client = makeStdioClient(server, 0);

  const chunks: Uint8Array[] = [];
  let stalledOnce = false;
  for await (const item of client.execStdioRead(
    "ta-1",
    "ex-1",
    FileDescriptor.FILE_DESCRIPTOR_STDOUT,
    null,
  )) {
    chunks.push(item.data);
    if (!stalledOnce) {
      stalledOnce = true;
      await new Promise((resolve) => globalThis.setTimeout(resolve, 150));
    }
    if (concat(chunks).length >= OUTPUT.length) break;
  }

  expect(concat(chunks)).toEqual(OUTPUT);
  expect(server.requestedOffsets).toEqual([0]);
  expect(server.aborts).toBe(0);
});
