import { expect, onTestFinished, test, vi } from "vitest";
import jwt from "jsonwebtoken";
import { ModalClient } from "../src/client";
import { MockGrpcClient } from "../test-support/grpc_mock";

function createMockEnvironmentClient(environment: string): {
  mockClient: ModalClient;
  mock: MockGrpcClient;
  getEnvironmentGetOrCreateCallCount: () => number;
} {
  let environmentGetOrCreateCallCount = 0;
  const mock = new MockGrpcClient();

  mock.handleUnary("/AuthTokenGet", () => {
    return { token: jwt.sign({ exp: 9999999999 }, "env-test") };
  });

  const originalHandleUnary = mock.handleUnary.bind(mock);
  mock.handleUnary = (rpcName: string, handler: any) => {
    if (rpcName === "/EnvironmentGetOrCreate") {
      originalHandleUnary(rpcName, (req: any) => {
        environmentGetOrCreateCallCount++;
        return handler(req);
      });
    } else {
      originalHandleUnary(rpcName, handler);
    }
  };

  const mockClient = new ModalClient({
    cpClient: mock as any,
    tokenId: "test-token-id",
    tokenSecret: "test-token-secret",
    environment,
  });

  return {
    mockClient,
    mock,
    getEnvironmentGetOrCreateCallCount: () => environmentGetOrCreateCallCount,
  };
}

test("GetEnvironmentCached", async () => {
  vi.stubEnv("MODAL_IMAGE_BUILDER_VERSION", "2024.04");
  onTestFinished(() => {
    vi.unstubAllEnvs();
  });

  const { mockClient, mock, getEnvironmentGetOrCreateCallCount } =
    createMockEnvironmentClient("");

  mock.handleUnary("/EnvironmentGetOrCreate", () => {
    return {
      environmentId: "en-test",
      metadata: {
        name: "",
        settings: {
          imageBuilderVersion: "2024.10",
          webhookSuffix: "",
        },
      },
    };
  });

  const mockClientWithConfig = new ModalClient({
    cpClient: mock as any,
    tokenId: "test-token-id",
    tokenSecret: "test-token-secret",
  });
  onTestFinished(() => {
    mockClient.close();
    mockClientWithConfig.close();
  });

  const version = await mockClientWithConfig.getImageBuilderVersion();
  expect(version).toBe("2024.04");

  // Does not fetch from server
  expect(getEnvironmentGetOrCreateCallCount()).toBe(0);
});

test("GetEnvironmentWithServer", async () => {
  // Do not set env var so version is pulled from server
  const originalValue = process.env.MODAL_IMAGE_BUILDER_VERSION;
  delete process.env.MODAL_IMAGE_BUILDER_VERSION;
  onTestFinished(() => {
    if (originalValue !== undefined) {
      process.env.MODAL_IMAGE_BUILDER_VERSION = originalValue;
    }
  });

  const { mockClient, mock, getEnvironmentGetOrCreateCallCount } =
    createMockEnvironmentClient("");

  mock.handleUnary("/EnvironmentGetOrCreate", () => {
    return {
      environmentId: "en-main-123",
      metadata: {
        name: "main",
        settings: {
          imageBuilderVersion: "2024.10",
          webhookSuffix: "modal.run",
        },
      },
    };
  });
  onTestFinished(() => {
    mockClient.close();
  });

  const version1 = await mockClient.getImageBuilderVersion();
  expect(version1).toBe("2024.10");
  expect(getEnvironmentGetOrCreateCallCount()).toBe(1);

  const version2 = await mockClient.getImageBuilderVersion();
  expect(version2).toBe("2024.10");

  // Uses the cache
  expect(getEnvironmentGetOrCreateCallCount()).toBe(1);

  vi.stubEnv("MODAL_PROFILE", "dev");
  const {
    mockClient: mockClientDev,
    mock: mockDev,
    getEnvironmentGetOrCreateCallCount: getDevEnvironmentGetOrCreateCallCount,
  } = createMockEnvironmentClient("dev");
  mockDev.handleUnary("/EnvironmentGetOrCreate", () => {
    return {
      environmentId: "en-dev",
      metadata: {
        name: "dev",
        settings: {
          imageBuilderVersion: "2025.06",
          webhookSuffix: "",
        },
      },
    };
  });

  const versionDev = await mockClientDev.getImageBuilderVersion();
  expect(versionDev).toBe("2025.06");
  expect(getDevEnvironmentGetOrCreateCallCount()).toBe(1);

  const version2Dev = await mockClientDev.getImageBuilderVersion();
  expect(version2Dev).toBe("2025.06");

  // Uses the cache
  expect(getDevEnvironmentGetOrCreateCallCount()).toBe(1);
});

test("GetImageBuilderVersionWithEnvironmentOverride", async () => {
  // Do not set env var so version is pulled from server
  const originalValue = process.env.MODAL_IMAGE_BUILDER_VERSION;
  delete process.env.MODAL_IMAGE_BUILDER_VERSION;
  onTestFinished(() => {
    if (originalValue !== undefined) {
      process.env.MODAL_IMAGE_BUILDER_VERSION = originalValue;
    }
  });

  // Profile's default environment differs from the per-call override.
  const { mockClient, mock } = createMockEnvironmentClient("dev");
  onTestFinished(() => {
    mockClient.close();
  });

  const requestedEnvironments: string[] = [];
  const environmentHandler = (req: any) => {
    requestedEnvironments.push(req.deploymentName);
    return {
      environmentId: `en-${req.deploymentName}`,
      metadata: {
        name: req.deploymentName,
        settings: {
          imageBuilderVersion:
            req.deploymentName === "prod" ? "2025.06" : "2024.10",
          webhookSuffix: "",
        },
      },
    };
  };
  // Two distinct environments are queried, so register a handler for each call.
  mock.handleUnary("/EnvironmentGetOrCreate", environmentHandler);
  mock.handleUnary("/EnvironmentGetOrCreate", environmentHandler);

  // The override environment, not the profile default, is queried.
  const version = await mockClient.getImageBuilderVersion("prod");
  expect(version).toBe("2025.06");
  expect(requestedEnvironments).toEqual(["prod"]);

  // No override falls back to the profile's default environment.
  const versionDefault = await mockClient.getImageBuilderVersion();
  expect(versionDefault).toBe("2024.10");
  expect(requestedEnvironments).toEqual(["prod", "dev"]);
});
