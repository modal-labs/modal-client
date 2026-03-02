import { Function_, FunctionTimeoutError } from "modal";
import { expect, test } from "vitest";

test("FunctionSpawn", async () => {
  const function_ = await Function_.lookup(
    "libmodal-test-support",
    "echo_string",
  );

  // Spawn function with kwargs.
  let functionCall = await function_.spawn([], { s: "hello" });
  expect(functionCall.functionCallId).toBeDefined();

  // Get results after spawn.
  let resultKwargs = await functionCall.get();
  expect(resultKwargs).toBe("output: hello");

  // Try the same again; same results should still be available.
  resultKwargs = await functionCall.get();
  expect(resultKwargs).toBe("output: hello");

  // Lookup function that takes a long time to complete.
  const sleep = await Function_.lookup("libmodal-test-support", "sleep");

  // Spawn with long running input.
  functionCall = await sleep.spawn([], { t: 5 });
  expect(functionCall.functionCallId).toBeDefined();

  // Getting outputs with timeout raises error.
  const promise = functionCall.get({ timeoutMs: 1000 });
  await expect(promise).rejects.toThrowError(FunctionTimeoutError);
});

test("FunctionCallGet0", async () => {
  const sleep = await Function_.lookup("libmodal-test-support", "sleep");

  const call = await sleep.spawn([0.5]);
  // Polling for output with timeout 0 should raise an error, since the
  // function call has not finished yet.
  await expect(call.get({ timeoutMs: 0 })).rejects.toThrowError(
    FunctionTimeoutError,
  );

  expect(await call.get()).toBe(null); // Wait for the function call to finish.
  expect(await call.get({ timeoutMs: 0 })).toBe(null); // Now we can get the result.
});
