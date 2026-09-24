import { expect, test, vi } from "vitest";
import { createChannel, createServer, type CallContext } from "nice-grpc";
import {
  TaskCommandRouterDefinition,
  TaskExecStartRequest,
  TaskExecStartResponse,
} from "../proto/modal_proto/task_command_router";
import { TaskCommandRouterClientImpl } from "../src/task_command_router_client";

function mockJwt(exp: number): string {
  const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  return `${header}.${btoa(JSON.stringify({ exp }))}.fake-signature`;
}

test("task command router client logs server warnings once", async () => {
  const logger = {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  };

  const unimplemented: Record<string, unknown> = {};
  for (const methodName of Object.keys(TaskCommandRouterDefinition.methods)) {
    unimplemented[methodName] = () => {
      throw new Error(`${methodName} is not implemented in this test server`);
    };
  }
  const server = createServer();
  server.add(TaskCommandRouterDefinition, {
    ...unimplemented,
    async taskExecStart(_request: TaskExecStartRequest, context: CallContext) {
      context.trailer.set(
        "x-modal-warning",
        "Container%20is%20low%20on%20disk",
      );
      return TaskExecStartResponse.create();
    },
  } as any);
  const port = await server.listen("127.0.0.1:0");

  const client: any = new TaskCommandRouterClientImpl(
    undefined as any, // serverClient, unused: nothing here refreshes a JWT
    "ta-1",
    "sb-1",
    true,
    `https://127.0.0.1:${port}`,
    mockJwt(Math.floor(Date.now() / 1000) + 3600),
    () => createChannel(`127.0.0.1:${port}`, undefined, {}),
    logger as any,
    0,
  );

  try {
    await client.execStart(TaskExecStartRequest.create());
    await client.execStart(TaskExecStartRequest.create());
  } finally {
    client.channel?.close();
    await server.shutdown();
  }

  expect(logger.warn.mock.calls).toEqual([["Container is low on disk"]]);
});
