import { tc } from "../test-support/test-client";
import { FunctionTimeoutError } from "modal";
import { expect, test } from "vitest";

test("FunctionSpawn", async () => {
  const function_ = await tc.functions.fromName(
    "libmodal-test-support",
    "echo_string",
  );

  let functionCall = await function_.spawn([], { s: "hello" });
  expect(functionCall.functionCallId).toMatch(/^fc-/);

  let resultKwargs = await functionCall.get();
  expect(resultKwargs).toBe("output: hello");

  resultKwargs = await functionCall.get();
  expect(resultKwargs).toBe("output: hello");

  const sleep = await tc.functions.fromName("libmodal-test-support", "sleep");
  functionCall = await sleep.spawn([], { t: 5 });
  expect(functionCall.functionCallId).toMatch(/^fc-/);

  const promise = functionCall.get({ timeoutMs: 1000 }); // 1000ms
  await expect(promise).rejects.toThrowError(FunctionTimeoutError);
});

test("FunctionCallGet0", async () => {
  const sleep = await tc.functions.fromName("libmodal-test-support", "sleep");

  const call = await sleep.spawn([0.5]);
  // Polling for output with timeout 0 should raise an error, since the
  // function call has not finished yet.
  await expect(call.get({ timeoutMs: 0 })).rejects.toThrowError(
    FunctionTimeoutError,
  );

  expect(await call.get()).toBe(null);
  expect(await call.get({ timeoutMs: 0 })).toBe(null);
});
