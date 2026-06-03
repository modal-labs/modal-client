import { tc } from "../test-support/test-client";
import { InvalidError, NotFoundError } from "modal";
import { expect, test } from "vitest";
import { createMockModalClients } from "../test-support/grpc_mock";
import { Function_ } from "../src/function";

test("FunctionCall", async () => {
  const function_ = await tc.functions.fromName(
    "libmodal-test-support",
    "echo_string",
  );

  const resultKwargs = await function_.remote([], { s: "hello" });
  expect(resultKwargs).toBe("output: hello");

  const resultArgs = await function_.remote(["hello"]);
  expect(resultArgs).toBe("output: hello");
});

test("FunctionCallJsMap", async () => {
  const function_ = await tc.functions.fromName(
    "libmodal-test-support",
    "identity_with_repr",
  );

  const resultKwargs = await function_.remote([new Map([["a", "b"]])]);
  expect(resultKwargs).toStrictEqual([{ a: "b" }, "{'a': 'b'}"]);
});

test("FunctionCallDateTimeRoundtrip", async () => {
  const function_ = await tc.functions.fromName(
    "libmodal-test-support",
    "identity_with_repr",
  );

  const testDate = new Date("2024-01-15T10:30:45.123Z");
  const result = await function_.remote([testDate]);

  expect(Array.isArray(result)).toBe(true);
  expect(result).toHaveLength(2);

  const [identityResult, reprResult] = result as [unknown, string];

  expect(reprResult).toContain("datetime.datetime");
  expect(reprResult).toContain("2024");

  expect(identityResult).toBeInstanceOf(Date);
  const receivedDate = identityResult as Date;

  // Check precision - JavaScript Date has millisecond precision
  // Python datetime has microsecond precision
  // We should get back millisecond precision (lose sub-millisecond)
  const timeDiff = Math.abs(testDate.getTime() - receivedDate.getTime());

  // JavaScript Date only has millisecond precision, so we should have no loss
  expect(timeDiff).toBeLessThan(1); // Less than 1 millisecond
  expect(receivedDate.getTime()).toBe(testDate.getTime());
});

test("FunctionCallLargeInput", async () => {
  const function_ = await tc.functions.fromName(
    "libmodal-test-support",
    "bytelength",
  );
  const len = 3 * 1000 * 1000; // More than 2 MiB, offload to blob storage
  const input = new Uint8Array(len);
  const result = await function_.remote([input]);
  expect(result).toBe(len);
});

test("FunctionNotFound", async () => {
  const promise = tc.functions.fromName(
    "libmodal-test-support",
    "not_a_real_function",
  );
  await expect(promise).rejects.toThrowError(NotFoundError);
});

test("FunctionCallInputPlane", async () => {
  const function_ = await tc.functions.fromName(
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
  const function_ = await tc.functions.fromName(
    "libmodal-test-support",
    "echo_string",
  );
  expect(await function_.getWebUrl()).toBeUndefined();
});

test("FunctionFromNameWithDotNotation", async () => {
  const promise = tc.functions.fromName(
    "libmodal-test-support",
    "MyClass.myMethod",
  );
  await expect(promise).rejects.toThrowError(
    `Cannot retrieve Cls methods using 'functions.fromName()'. Use:\n  const cls = await client.cls.fromName("libmodal-test-support", "MyClass");\n  const instance = await cls.instance();\n  const m = instance.method("myMethod");`,
  );
});

test("FunctionCallPreCborVersionError", async () => {
  // test that calling a pre 1.2 function raises an error
  const function_ = await tc.functions.fromName(
    "test-support-1-1",
    "identity_with_repr",
  );

  // Represent Python kwargs.
  const promise = function_.remote([], { s: "hello" });
  await expect(promise).rejects.toThrowError(
    /Redeploy with Modal Python SDK >= 1.2/,
  );
});

test("WebEndpointRemoteCallError", async () => {
  const function_ = await tc.functions.fromName(
    "libmodal-test-support",
    "web_endpoint_echo",
  );

  const promise = function_.remote(["hello"]);
  await expect(promise).rejects.toThrowError(InvalidError);
  await expect(promise).rejects.toThrowError(
    /A webhook Function cannot be invoked for remote execution with '\.remote'/,
  );
});

test("WebEndpointSpawnCallError", async () => {
  const function_ = await tc.functions.fromName(
    "libmodal-test-support",
    "web_endpoint_echo",
  );

  const promise = function_.spawn(["hello"]);
  await expect(promise).rejects.toThrowError(InvalidError);
  await expect(promise).rejects.toThrowError(
    /A webhook Function cannot be invoked for remote execution with '\.spawn'/,
  );
});

test("DynamicConfigurationE2ETest", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (req) => {
    expect(req).toMatchObject({
      appName: "libmodal-test-support",
      objectTag: "echo_string",
    });

    return {
      functionId: "fid-unbound",
      handleMetadata: {},
    };
  });

  mock.handleUnary("FunctionBindParams", (req) => {
    expect(req).toMatchObject({
      functionId: "fid-unbound",
      serializedParams: new Uint8Array(),
      functionOptions: {
        resources: {
          milliCpu: 2000,
          milliCpuMax: 2000,
        },
        maxConcurrentInputs: 10,
        batchMaxSize: 10,
        batchLingerMs: 100,
      },
    });

    return {
      boundFunctionId: "fid-bound",
    };
  });

  const f = await mc.functions.fromName("libmodal-test-support", "echo_string");

  const configured = f
    .withOptions({ cpu: 2.0 })
    .withOptions({ cpuLimit: 2.0 })
    .withConcurrency({ maxInputs: 10 })
    .withBatching({ maxBatchSize: 10, waitMs: 100 });

  expect(configured).toMatchObject({
    functionId: "fid-unbound",
  });

  // note(ayush): we check that the proto is created correctly in the mock rpc call
  const boundF = await configured.instance();

  expect(boundF.functionId).not.toBe(f.functionId);
});

test("TestOptionsMutations", async () => {
  const { mockClient: mc } = createMockModalClients();

  const opts = {};
  const f1 = new Function_(mc, "fid", "", undefined, opts);
  const f2 = f1.withOptions({ cpu: 2.0 });
  const f3 = f1.withOptions({ cpu: 2.0 });

  // assert that opts wasn't mutated, reverse ordering of `{}` and opts is intentional
  expect({}).toMatchObject(opts);

  expect(f1).not.toBe(f2);
  expect(f1).not.toBe(f3);
  expect(f2).not.toBe(f3);
});

test("TestEmptyInstance", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (req) => {
    expect(req).toMatchObject({
      appName: "libmodal-test-support",
      objectTag: "echo_string",
    });

    return {
      functionId: "fid-unbound",
      handleMetadata: {},
    };
  });

  const f = await mc.functions.fromName("libmodal-test-support", "echo_string");

  // shouldn't fail because it won't call the RPC
  const bound = await f.instance();

  expect(f.functionId).toBe(bound.functionId);
});
