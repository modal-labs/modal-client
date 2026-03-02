import { expect, test } from "vitest";

import { Cls, NotFoundError } from "modal";

test("ClsCall", async () => {
  const cls = await Cls.lookup("libmodal-test-support", "EchoCls");
  const instance = await cls.instance();

  // Try accessing a non-existent method
  expect(() => instance.method("nonexistent")).toThrowError(NotFoundError);

  const function_ = instance.method("echo_string");
  const result = await function_.remote([], { s: "hello" });
  expect(result).toEqual("output: hello");

  const cls2 = await Cls.lookup("libmodal-test-support", "EchoClsParametrized");
  const instance2 = await cls2.instance({ name: "hello-init" });

  const function2 = instance2.method("echo_parameter");
  const result2 = await function2.remote();
  expect(result2).toEqual("output: hello-init");
});

test("ClsNotFound", async () => {
  const cls = Cls.lookup("libmodal-test-support", "NotRealClassName");
  await expect(cls).rejects.toThrowError(NotFoundError);
});

test("ClsCallInputPlane", async () => {
  const cls = await Cls.lookup("libmodal-test-support", "EchoClsInputPlane");
  const instance = await cls.instance();

  const function_ = instance.method("echo_string");
  const result = await function_.remote([], { s: "hello" });
  expect(result).toEqual("output: hello");
});
