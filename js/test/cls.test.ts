import { expect, test } from "vitest";

import { tc } from "../test-support/test-client";
import { NotFoundError } from "modal";
import { createMockModalClients } from "../test-support/grpc_mock";

test("ClsCall", async () => {
  const cls = await tc.cls.fromName("libmodal-test-support", "EchoCls");
  const instance = await cls.instance();

  // Try accessing a non-existent method
  expect(() => instance.method("nonexistent")).toThrowError(NotFoundError);

  const function_ = instance.method("echo_string");
  const result = await function_.remote([], { s: "hello" });
  expect(result).toEqual("output: hello");

  const cls2 = await tc.cls.fromName(
    "libmodal-test-support",
    "EchoClsParametrized",
  );
  const instance2 = await cls2.instance({ name: "hello-init" });

  const function2 = instance2.method("echo_parameter");
  const result2 = await function2.remote();
  expect(result2).toEqual("output: hello-init");
});

test("ClsNotFound", async () => {
  const cls = tc.cls.fromName("libmodal-test-support", "NotRealClassName");
  await expect(cls).rejects.toThrowError(NotFoundError);
});

test("ClsFromNameWithVersion", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (req) => {
    expect(req).toMatchObject({
      appName: "libmodal-test-support",
      objectTag: "EchoCls.*",
      appVersion: 3,
    });
    return {
      functionId: "fid-versioned",
      handleMetadata: {},
    };
  });

  await mc.cls.fromName("libmodal-test-support", "EchoCls", { version: 3 });

  mock.assertExhausted();
});

test("ClsCallInputPlane", async () => {
  const cls = await tc.cls.fromName(
    "libmodal-test-support",
    "EchoClsInputPlane",
  );
  const instance = await cls.instance();

  const function_ = instance.method("echo_string");
  const result = await function_.remote([], { s: "hello" });
  expect(result).toEqual("output: hello");
});
