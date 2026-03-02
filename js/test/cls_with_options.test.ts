import { expect, test } from "vitest";
import type { Secret, Volume } from "modal";
import { createMockModalClients } from "../test-support/grpc_mock";
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

test("withOptions({ cpu, cpuLimit }) sets milliCpu and milliCpuMax", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  mock.handleUnary("FunctionBindParams", (req: any) => {
    expect(req).toMatchObject({ functionId: "fid" });
    const fo = req.functionOptions;
    expect(fo.resources.milliCpu).toBe(2000);
    expect(fo.resources.milliCpuMax).toBe(4500);
    return { boundFunctionId: "fid-1", handleMetadata: {} };
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");
  const instance = await cls
    .withOptions({ cpu: 2.0, cpuLimit: 4.5 })
    .instance();
  expect(instance).toBeTruthy();

  mock.assertExhausted();
});

test("withOptions cpuLimit lower than cpu throws error", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");
  await expect(
    cls.withOptions({ cpu: 4.0, cpuLimit: 2.0 }).instance(),
  ).rejects.toThrow("cpu (4) cannot be higher than cpuLimit (2)");

  mock.assertExhausted();
});

test("withOptions cpuLimit without cpu throws error", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");
  await expect(cls.withOptions({ cpuLimit: 4.0 }).instance()).rejects.toThrow(
    "must also specify cpu when cpuLimit is specified",
  );

  mock.assertExhausted();
});

test("withOptions({ memory, memoryLimit }) sets memoryMb and memoryMbMax", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  mock.handleUnary("FunctionBindParams", (req: any) => {
    expect(req).toMatchObject({ functionId: "fid" });
    const fo = req.functionOptions;
    expect(fo.resources.memoryMb).toBe(1024);
    expect(fo.resources.memoryMbMax).toBe(2048);
    return { boundFunctionId: "fid-1", handleMetadata: {} };
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");
  const instance = await cls
    .withOptions({ memoryMiB: 1024, memoryLimitMiB: 2048 })
    .instance();
  expect(instance).toBeTruthy();

  mock.assertExhausted();
});

test("withOptions memoryLimit lower than memory throws error", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");
  await expect(
    cls.withOptions({ memoryMiB: 2048, memoryLimitMiB: 1024 }).instance(),
  ).rejects.toThrow(
    "memoryMiB (2048) cannot be higher than memoryLimitMiB (1024)",
  );

  mock.assertExhausted();
});

test("withOptions memoryLimit without memory throws error", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");
  await expect(
    cls.withOptions({ memoryLimitMiB: 2048 }).instance(),
  ).rejects.toThrow(
    "must also specify memoryMiB when memoryLimitMiB is specified",
  );

  mock.assertExhausted();
});

test("withOptions negative cpu throws error", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");
  await expect(cls.withOptions({ cpu: -1.0 }).instance()).rejects.toThrow(
    "must be a positive number",
  );

  mock.assertExhausted();
});

test("withOptions zero cpu throws error", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");
  await expect(cls.withOptions({ cpu: 0.0 }).instance()).rejects.toThrow(
    "must be a positive number",
  );

  mock.assertExhausted();
});

test("withOptions negative memory throws error", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");
  await expect(cls.withOptions({ memoryMiB: -100 }).instance()).rejects.toThrow(
    "must be a positive number",
  );

  mock.assertExhausted();
});

test("withOptions zero memory throws error", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("FunctionGet", (_: any) => {
    return _mockFunctionProto;
  });

  const cls = await mc.cls.fromName("libmodal-test-support", "EchoCls");
  await expect(cls.withOptions({ memoryMiB: 0 }).instance()).rejects.toThrow(
    "must be a positive number",
  );

  mock.assertExhausted();
});
