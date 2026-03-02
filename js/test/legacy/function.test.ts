import { Function_, NotFoundError } from "modal";
import { expect, test } from "vitest";
import { createMockModalClients } from "../../test-support/grpc_mock";

test("FunctionCall", async () => {
  const function_ = await Function_.lookup(
    "libmodal-test-support",
    "echo_string",
  );

  // Represent Python kwargs.
  const resultKwargs = await function_.remote([], { s: "hello" });
  expect(resultKwargs).toBe("output: hello");

  // Try the same, but with args.
  const resultArgs = await function_.remote(["hello"]);
  expect(resultArgs).toBe("output: hello");
});

test("FunctionCallLargeInput", async () => {
  const function_ = await Function_.lookup(
    "libmodal-test-support",
    "bytelength",
  );
  const len = 3 * 1000 * 1000; // More than 2 MiB, offload to blob storage
  const input = new Uint8Array(len);
  const result = await function_.remote([input]);
  expect(result).toBe(len);
});

test("FunctionNotFound", async () => {
  const promise = Function_.lookup(
    "libmodal-test-support",
    "not_a_real_function",
  );
  await expect(promise).rejects.toThrowError(NotFoundError);
});

test("FunctionCallInputPlane", async () => {
  const function_ = await Function_.lookup(
    "libmodal-test-support",
    "input_plane",
  );
  const result = await function_.remote(["hello"]);
  expect(result).toBe("output: hello");
});

test("FunctionGetCurrentStats", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/FunctionGetCurrentStats", (req) => {
    expect(req).toMatchObject({ functionId: "fid-stats" });
    return { backlog: 3, numTotalTasks: 7 };
  });

  const function_ = new Function_(mc, "fid-stats");
  const stats = await function_.getCurrentStats();
  expect(stats).toEqual({ backlog: 3, numTotalRunners: 7 });

  mock.assertExhausted();
});

test("FunctionUpdateAutoscaler", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/FunctionUpdateSchedulingParams", (req) => {
    expect(req).toMatchObject({
      functionId: "fid-auto",
      settings: {
        minContainers: 1,
        maxContainers: 10,
        bufferContainers: 2,
        scaledownWindow: 300,
      },
    });
    return {};
  });

  const function_ = new Function_(mc, "fid-auto");
  await function_.updateAutoscaler({
    minContainers: 1,
    maxContainers: 10,
    bufferContainers: 2,
    scaledownWindowMs: 300 * 1000,
  });

  mock.handleUnary("/FunctionUpdateSchedulingParams", (req) => {
    expect(req).toMatchObject({
      functionId: "fid-auto",
      settings: { minContainers: 2 },
    });
    return {};
  });

  await function_.updateAutoscaler({ minContainers: 2 });

  mock.assertExhausted();
});

test("FunctionGetWebUrl", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (req) => {
    expect(req).toMatchObject({
      appName: "libmodal-test-support",
      objectTag: "web_endpoint",
    });
    return {
      functionId: "fid-web",
      handleMetadata: { webUrl: "https://endpoint.internal" },
    };
  });

  const web_endpoint = await mc.functions.fromName(
    "libmodal-test-support",
    "web_endpoint",
  );
  expect(await web_endpoint.getWebUrl()).toBe("https://endpoint.internal");

  mock.assertExhausted();
});

test("FunctionGetWebUrlOnNonWebFunction", async () => {
  const function_ = await Function_.lookup(
    "libmodal-test-support",
    "echo_string",
  );
  expect(await function_.getWebUrl()).toBeUndefined();
});
