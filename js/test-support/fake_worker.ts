// Copyright Modal Labs 2026
import { createServer } from "nice-grpc";

import {
  TaskCommandRouterDefinition,
  TaskExecStartResponse,
  TaskExecStdioFileDescriptor,
  TaskExecStdioReadResponse,
  type TaskExecStdioReadRequest,
} from "../proto/modal_proto/task_command_router";
import { createMockModalClients } from "./grpc_mock";

/**
 * A local worker over a real socket, with the control plane mocked, so tests
 * can enter where callers do - `sandbox.exec()` - and still see what reaches
 * the wire. Output arrives through real protobuf decoding, which is what makes
 * chunk framing observable.
 */

export const SANDBOX_ID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV";
export const TASK_ID = "ta-01ARZ3NDEKTSV4RRFFQ69G5FAV";
export const DEFAULT_OUTPUT = "line-0\nline-1\nline-2\n";

function mockJwt(): string {
  const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  const exp = Math.floor(Date.now() / 1000) + 3600;
  return `${header}.${btoa(JSON.stringify({ exp }))}.signature`;
}

/** Records which descriptors were read, and serves `chunks` when they are. */
function fakeWorker(
  reads: TaskExecStdioFileDescriptor[],
  chunks: Uint8Array[],
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
    async *taskExecStdioRead(request: TaskExecStdioReadRequest) {
      reads.push(request.fileDescriptor);
      for (const chunk of chunks) {
        yield TaskExecStdioReadResponse.create({ data: chunk });
      }
    },
  };
}

/**
 * Start a worker serving `chunks` as stdio output, and a client wired to it.
 *
 * Each chunk is sent as its own stdio response, so a caller reading the stream
 * sees them as separate chunks.
 */
export async function startFakeWorker(chunks?: Uint8Array[]) {
  const reads: TaskExecStdioFileDescriptor[] = [];
  const server = createServer();
  server.add(
    TaskCommandRouterDefinition,
    fakeWorker(
      reads,
      chunks ?? [new TextEncoder().encode(DEFAULT_OUTPUT)],
    ) as any,
  );
  const port = await server.listen("127.0.0.1:0");

  const { mockClient, mockCpClient } = createMockModalClients();
  // A localhost server URL is what makes the SDK dial the worker without TLS.
  mockClient.profile.serverUrl = "http://127.0.0.1:1";
  // Handlers are consumed as they are used, so register enough for any test here.
  for (let i = 0; i < 4; i++) {
    mockCpClient.handleUnary("SandboxGetTaskIdV2", () => ({ taskId: TASK_ID }));
    mockCpClient.handleUnary("SandboxGetCommandRouterAccess", () => ({
      url: `https://127.0.0.1:${port}`,
      jwt: mockJwt(),
    }));
  }

  return {
    mockClient,
    reads,
    shutdown: async () => {
      server.forceShutdown();
    },
  };
}

/** Let anything the SDK scheduled run, so "nothing happened" means nothing will. */
export async function settle() {
  for (let i = 0; i < 20; i++) {
    await new Promise((resolve) => globalThis.setImmediate(resolve));
  }
  await new Promise((resolve) => globalThis.setTimeout(resolve, 100));
}
