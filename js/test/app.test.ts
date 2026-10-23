import { App, InvalidError } from "modal";
import { expect, test } from "vitest";
import { AppStopSource } from "../proto/modal_proto/api";
import { createMockModalClients } from "../test-support/grpc_mock";

test("AppStop", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/AppGetOrCreate", () => ({
    appId: "ap-test-123",
  }));

  mock.handleUnary("/AppStop", (req: any) => {
    expect(req.appId).toBe("ap-test-123");
    expect(req.source).toBe(AppStopSource.APP_STOP_SOURCE_PYTHON_CLIENT);
    return {};
  });

  const app = await mc.apps.fromName("libmodal-test");
  await app.stop();
  mock.assertExhausted();
});

test("AppStop without an App ID", async () => {
  const { mockClient: mc } = createMockModalClients();
  const app = new App("", undefined, undefined, mc);
  await expect(app.stop()).rejects.toThrow(InvalidError);
});

test("AppStop without a client", async () => {
  const app = new App("ap-test-123");
  await expect(app.stop()).rejects.toThrow(InvalidError);
});
