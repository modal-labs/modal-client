import { checkForRenamedParams } from "../src/validation";

import { expect, test } from "vitest";
import { createMockModalClients } from "../test-support/grpc_mock";

test("checkForRenamedParams", () => {
  expect(() =>
    checkForRenamedParams({ timeout: 5000 }, { timeout: "timeoutMs" }),
  ).toThrow("Parameter 'timeout' has been renamed to 'timeoutMs'.");

  expect(() =>
    checkForRenamedParams({ timeoutMs: 5000 }, { timeout: "timeoutMs" }),
  ).not.toThrow();

  expect(() =>
    checkForRenamedParams(null, { timeout: "timeoutMs" }),
  ).not.toThrow();

  expect(() =>
    checkForRenamedParams(undefined, { timeout: "timeoutMs" }),
  ).not.toThrow();

  expect(() =>
    checkForRenamedParams({}, { timeout: "timeoutMs" }),
  ).not.toThrow();
});

test("ModalClient constructor rejects old 'timeout' parameter", async () => {
  const { ModalClient } = await import("modal");

  expect(
    () =>
      new ModalClient({
        timeout: 5000,
      } as any),
  ).toThrow("Parameter 'timeout' has been renamed to 'timeoutMs'.");
});

test("Cls.withOptions rejects old parameter names", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => ({
    functionId: "fid",
    handleMetadata: {
      methodHandleMetadata: { echo_string: {} },
      classParameterInfo: { schema: [] },
    },
  }));

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");

  await expect(
    cls.withOptions({ timeout: 5000 } as any).instance(),
  ).rejects.toThrow("Parameter 'timeout' has been renamed to 'timeoutMs'.");

  await expect(
    cls.withOptions({ memory: 512 } as any).instance(),
  ).rejects.toThrow("Parameter 'memory' has been renamed to 'memoryMiB'.");

  await expect(
    cls.withOptions({ memoryLimit: 1024 } as any).instance(),
  ).rejects.toThrow(
    "Parameter 'memoryLimit' has been renamed to 'memoryLimitMiB'.",
  );
});
