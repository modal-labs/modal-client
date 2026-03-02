import { expect, test } from "vitest";
import type { Secret, Volume } from "modal";
import { createMockModalClients } from "../../test-support/grpc_mock";
import { Retries } from "modal";

const _mockFunctionProto = {
  functionId: "fid",
  handleMetadata: {
    methodHandleMetadata: { echo_string: {} },
    classParameterInfo: { schema: [] },
  },
};

test("Cls.withOptions stacking", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");

  mock.handleUnary("FunctionBindParams", (req: any) => {
    expect(req).toMatchObject({ functionId: "fid" });
    const fo = req.functionOptions;
    expect(fo.timeoutSecs).toBe(60);
    expect(fo.resources).toBeDefined();
    expect(fo.resources.milliCpu).toBe(250);
    expect(fo.resources.memoryMb).toBe(256);
    expect(fo.resources.gpuConfig).toBeDefined();
    expect(fo.secretIds).toEqual(["sec-1"]);
    expect(fo.replaceSecretIds).toBe(true);
    expect(fo.replaceVolumeMounts).toBe(true);
    expect(fo.volumeMounts).toEqual([
      {
        mountPath: "/mnt/test",
        volumeId: "vol-1",
        allowBackgroundCommits: true,
        readOnly: false,
      },
    ]);
    return { boundFunctionId: "fid-1", handleMetadata: {} };
  });

  const secret = { secretId: "sec-1" } as Secret;
  const volume = { volumeId: "vol-1" } as Volume;

  const optioned = cls
    .withOptions({ timeoutMs: 45_000, cpu: 0.25 })
    .withOptions({ timeoutMs: 60_000, memoryMiB: 256, gpu: "T4" })
    .withOptions({ secrets: [secret], volumes: { "/mnt/test": volume } });

  const instance = await optioned.instance();
  expect(instance).toBeTruthy();

  mock.assertExhausted();
});

test("Cls.withConcurrency/withConcurrency/withBatching chaining", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");

  mock.handleUnary("FunctionBindParams", (req: any) => {
    expect(req).toMatchObject({ functionId: "fid" });
    const fo = req.functionOptions;
    expect(fo).toBeDefined();
    expect(fo.timeoutSecs).toBe(60);
    expect(fo.maxConcurrentInputs).toBe(10);
    expect(fo.batchMaxSize).toBe(11);
    expect(fo.batchLingerMs).toBe(12);
    return { boundFunctionId: "fid-1", handleMetadata: {} };
  });

  const chained = cls
    .withOptions({ timeoutMs: 60_000 })
    .withConcurrency({ maxInputs: 10 })
    .withBatching({ maxBatchSize: 11, waitMs: 12 });

  const instance = await chained.instance();
  expect(instance).toBeTruthy();

  mock.assertExhausted();
});

test("Cls.withOptions retries", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");

  mock.handleUnary("FunctionBindParams", (req: any) => {
    const fo = req.functionOptions;
    expect(fo).toBeDefined();
    expect(fo.retryPolicy).toMatchObject({
      retries: 3,
      backoffCoefficient: 1.0,
      initialDelayMs: 1000,
      maxDelayMs: 60000,
    });
    return { boundFunctionId: "fid-1", handleMetadata: {} };
  });

  await cls.withOptions({ retries: 3 }).instance();

  mock.handleUnary("FunctionBindParams", (req: any) => {
    const fo = req.functionOptions;
    expect(fo).toBeDefined();
    expect(fo.retryPolicy).toMatchObject({
      retries: 2,
      backoffCoefficient: 2.0,
      initialDelayMs: 2000,
      maxDelayMs: 5000,
    });
    return { boundFunctionId: "fid-2", handleMetadata: {} };
  });

  const retries = new Retries({
    maxRetries: 2,
    backoffCoefficient: 2.0,
    initialDelayMs: 2000,
    maxDelayMs: 5000,
  });
  await cls.withOptions({ retries }).instance();

  mock.assertExhausted();
});

test("Cls.withOptions invalid values", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");
  await expect(cls.withOptions({ timeoutMs: 1500 }).instance()).rejects.toThrow(
    /timeoutMs must be a multiple of 1000ms/,
  );

  await expect(
    cls.withOptions({ scaledownWindowMs: 2500 }).instance(),
  ).rejects.toThrow(/scaledownWindowMs must be a multiple of 1000ms/);

  mock.assertExhausted();
});

test("withOptions({ secrets: [] }) binds and does not replace secrets", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  mock.handleUnary("FunctionBindParams", (req: any) => {
    expect(req).toMatchObject({ functionId: "fid" });
    const fo = req.functionOptions;
    expect(Array.isArray(fo.secretIds)).toBe(true);
    expect(fo.secretIds.length).toBe(0);
    expect(fo.replaceSecretIds).toBe(false);

    return { boundFunctionId: "fid-1", handleMetadata: {} };
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");
  const instance = await cls.withOptions({ secrets: [] }).instance();
  expect(instance).toBeTruthy();

  mock.assertExhausted();
});

test("withOptions({ volumes: {} }) binds and does not replace volumes", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  mock.handleUnary("FunctionBindParams", (req: any) => {
    expect(req).toMatchObject({ functionId: "fid" });
    const fo = req.functionOptions;
    expect(Array.isArray(fo.volumeMounts)).toBe(true);
    expect(fo.volumeMounts.length).toBe(0);
    expect(fo.replaceVolumeMounts).toBe(false);

    return { boundFunctionId: "fid-1", handleMetadata: {} };
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");
  const instance = await cls.withOptions({ volumes: {} }).instance();
  expect(instance).toBeTruthy();

  mock.assertExhausted();
});
