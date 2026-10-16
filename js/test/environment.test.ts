import { expect, onTestFinished, test, vi } from "vitest";
import jwt from "jsonwebtoken";
import {
  AlreadyExistsError,
  App,
  type AppInfo,
  Environment,
  type EnvironmentCreateParams,
  InvalidError,
  ModalClient,
  NotFoundError,
} from "modal";
import { MockGrpcClient } from "../test-support/grpc_mock";
import { createMockModalClients } from "../test-support/grpc_mock";
import {
  AppListRequest,
  AppListResponse,
  AppState,
  AppGetInfoRequest,
  AppGetInfoResponse,
  AppGetOrCreateResponse,
  AppHandleMetadata,
  EnvironmentCreateRequest,
  EnvironmentDeleteRequest,
  EnvironmentGetOrCreateRequest,
  EnvironmentGetOrCreateResponse,
  EnvironmentListResponse,
  ObjectCreationType,
} from "../proto/modal_proto/api";
import { ClientError, Status } from "nice-grpc";
import { Empty } from "../proto/google/protobuf/empty";

test("AppFromName caches AppInfo and refreshes it on request", async () => {
  const { mockClient: client, mockCpClient: mock } = createMockModalClients();
  onTestFinished(() => client.close());
  const metadata = AppHandleMetadata.create({
    description: "App description",
    functions: { echo: "fu-echo" },
    servers: { api: "fu-api" },
    lifecycle: {
      appState: AppState.APP_STATE_DEPLOYED,
      createdAt: 1700000000.125,
      createdBy: "creator",
      deployedAt: 1700000100,
      deployedBy: "deployer",
      version: 3,
    },
  });
  mock.handleUnary("/AppGetOrCreate", () =>
    AppGetOrCreateResponse.create({
      appId: "ap-test",
      handleMetadata: metadata,
    }),
  );
  const app = await client.apps.fromName("libmodal-test");
  const expected: AppInfo = {
    appId: "ap-test",
    description: "App description",
    functions: { echo: "fu-echo" },
    servers: { api: "fu-api" },
    lifecycle: {
      state: "deployed",
      version: 3,
      createdAt: new Date(1700000000125),
      createdBy: "creator",
      deployedAt: new Date(1700000100000),
      deployedBy: "deployer",
      stoppedAt: undefined,
      stoppedBy: undefined,
    },
  };
  const info = await app.info();
  expect(info).toEqual(expected);
  info.functions.echo = "changed";
  info.lifecycle.createdAt.setTime(0);
  expect(await app.info()).toEqual(expected);

  mock.handleUnary("/AppGetInfo", (req) => {
    expect(req).toEqual(AppGetInfoRequest.create({ appId: "ap-test" }));
    return AppGetInfoResponse.create({
      info: {
        ...metadata,
        lifecycle: {
          ...metadata.lifecycle,
          appState: AppState.APP_STATE_STOPPED,
          stoppedAt: 1700000200,
          stoppedBy: "stopper",
        },
      },
    });
  });
  const refreshed = await app.info({ refresh: true });
  expect(refreshed.lifecycle).toEqual({
    ...expected.lifecycle,
    state: "stopped",
    stoppedAt: new Date(1700000200000),
    stoppedBy: "stopper",
  });
  expect(await app.info()).toEqual(refreshed);
  mock.assertExhausted();
});

test.each([
  [AppState.APP_STATE_UNSPECIFIED, "ephemeral"],
  [AppState.APP_STATE_EPHEMERAL, "ephemeral"],
  [AppState.APP_STATE_DETACHED, "detached"],
  [AppState.APP_STATE_DETACHED_DISCONNECTED, "detached"],
  [AppState.APP_STATE_INITIALIZING, "initializing"],
  [AppState.APP_STATE_STOPPING, "stopping"],
  [AppState.APP_STATE_DISABLED, "disabled"],
  [999 as AppState, "unknown"],
] as const)("AppInfo maps lifecycle state (%s)", async (state, expected) => {
  const { mockClient: client, mockCpClient: mock } = createMockModalClients();
  onTestFinished(() => client.close());
  mock.handleUnary("/AppGetOrCreate", () =>
    AppGetOrCreateResponse.create({
      appId: "ap-test",
      handleMetadata: { lifecycle: { appState: state, createdAt: 1700000000 } },
    }),
  );
  const info = await (await client.apps.fromName("libmodal-test")).info();
  expect(info.lifecycle).toEqual({
    state: expected,
    createdAt: new Date(1700000000000),
    createdBy: "",
    version: undefined,
    deployedAt: undefined,
    deployedBy: undefined,
    stoppedAt: undefined,
    stoppedBy: undefined,
  });
  expect(info.functions).toEqual({});
  expect(info.servers).toEqual({});
  mock.assertExhausted();
});

test("AppInfo fetches missing metadata and preserves its cache on refresh failure", async () => {
  const { mockClient: client, mockCpClient: mock } = createMockModalClients();
  onTestFinished(() => client.close());
  mock.handleUnary("/AppGetOrCreate", () =>
    AppGetOrCreateResponse.create({ appId: "ap-test" }),
  );
  mock.handleUnary("/AppGetInfo", (req) => {
    expect(req).toEqual(AppGetInfoRequest.create({ appId: "ap-test" }));
    return AppGetInfoResponse.create({
      info: { description: "Fetched description" },
    });
  });
  const app = await client.apps.fromName("libmodal-test");
  const info = await app.info();
  expect(info.description).toBe("Fetched description");
  for (const code of [Status.PERMISSION_DENIED, Status.NOT_FOUND]) {
    const error = new ClientError("/AppGetInfo", code, "Refresh failed");
    mock.handleUnary("/AppGetInfo", () => {
      throw error;
    });
    const refresh = app.info({ refresh: true });
    if (code === Status.NOT_FOUND) {
      await expect(refresh).rejects.toThrow(NotFoundError);
    } else {
      await expect(refresh).rejects.toBe(error);
    }
    expect(await app.info()).toEqual(info);
  }
  mock.assertExhausted();
});

test.each([undefined, { restricted: true, defaultRole: "viewer" }])(
  "EnvironmentObjectsCreate (%j)",
  async (params: EnvironmentCreateParams | undefined) => {
    const { mockClient: client, mockCpClient: mock } = createMockModalClients();
    onTestFinished(() => client.close());
    mock.handleUnary("/EnvironmentCreate", (req) => {
      expect(req).toEqual(
        EnvironmentCreateRequest.create({
          name: "prod",
          isManaged: params?.restricted ?? false,
          defaultMemberRoleStr: params?.defaultRole ?? "",
        }),
      );
      return Empty.create();
    });
    await expect(
      client.environments.objects.create("prod", params),
    ).resolves.toBeUndefined();
    mock.assertExhausted();
  },
);

test.each([
  "",
  "prod.v2",
  "en-prod",
  "EN-prod",
  "-prod",
  "prod/v2",
  "prod:v2",
  "a".repeat(65),
])("EnvironmentObjectsCreate rejects invalid names (%j)", async (name) => {
  const { mockClient: client, mockCpClient: mock } = createMockModalClients();
  onTestFinished(() => client.close());
  await expect(client.environments.objects.create(name)).rejects.toThrow(
    InvalidError,
  );
  mock.assertExhausted();
});

test("EnvironmentObjectsList returns hydrated handles with working Apps namespaces", async () => {
  const { mockClient: client, mockCpClient: mock } = createMockModalClients();
  onTestFinished(() => client.close());
  let listCalls = 0;
  mock.handleUnary("/EnvironmentList", (req) => {
    listCalls++;
    expect(req).toEqual(Empty.create());
    return EnvironmentListResponse.create({
      items: [
        { environmentId: "en-prod", name: "prod" },
        { environmentId: "en-dev", name: "dev" },
      ],
    });
  });
  const environments = client.environments.objects.list();
  expect(listCalls).toBe(0);
  const handles: Environment[] = [];
  for await (const environment of environments) {
    expect(environment).toBeInstanceOf(Environment);
    expect(environment).not.toHaveProperty("settings");
    handles.push(environment);
  }
  expect(
    handles.map((environment) => [environment.environmentId, environment.name]),
  ).toEqual([
    ["en-prod", "prod"],
    ["en-dev", "dev"],
  ]);
  for (const environment of handles) {
    mock.handleUnary("/AppList", (req) => {
      expect(req).toEqual(
        AppListRequest.create({ environmentName: environment.name }),
      );
      return AppListResponse.create({
        apps: [
          {
            appId: `ap-${environment.name}`,
            name: "libmodal-test",
            state: AppState.APP_STATE_DEPLOYED,
          },
        ],
      });
    });
    const apps: App[] = [];
    for await (const app of environment.apps.list()) {
      apps.push(app);
    }
    expect(apps).toEqual([
      new App(`ap-${environment.name}`, "libmodal-test", environment.name),
    ]);
  }
  expect(listCalls).toBe(1);
  mock.assertExhausted();
});

test("EnvironmentObjectsList handles empty results", async () => {
  const { mockClient: client, mockCpClient: mock } = createMockModalClients();
  onTestFinished(() => client.close());
  mock.handleUnary("/EnvironmentList", () => EnvironmentListResponse.create());
  expect((await client.environments.objects.list().next()).done).toBe(true);
  mock.assertExhausted();
});

test("EnvironmentObjectsDelete uses the Environment name", async () => {
  const { mockClient: client, mockCpClient: mock } = createMockModalClients();
  onTestFinished(() => client.close());
  mock.handleUnary("/EnvironmentDelete", (req) => {
    expect(req).toEqual(EnvironmentDeleteRequest.create({ name: "prod" }));
    return Empty.create();
  });
  await expect(
    client.environments.objects.delete("prod"),
  ).resolves.toBeUndefined();
  mock.assertExhausted();
});

test("EnvironmentObjectsCreate translates validation and duplicate errors", async () => {
  const { mockClient: client, mockCpClient: mock } = createMockModalClients();
  onTestFinished(() => client.close());
  mock.handleUnary("/EnvironmentCreate", () => {
    throw new ClientError(
      "/EnvironmentCreate",
      Status.ALREADY_EXISTS,
      "Already exists",
    );
  });
  mock.handleUnary("/EnvironmentCreate", () => {
    throw new ClientError(
      "/EnvironmentCreate",
      Status.INVALID_ARGUMENT,
      "Invalid role",
    );
  });
  await expect(client.environments.objects.create("prod")).rejects.toThrow(
    AlreadyExistsError,
  );
  await expect(
    client.environments.objects.create("prod", { defaultRole: "invalid" }),
  ).rejects.toThrow(InvalidError);
  mock.assertExhausted();
});

test("EnvironmentObjectsDelete translates missing Environment errors", async () => {
  const { mockClient: client, mockCpClient: mock } = createMockModalClients();
  onTestFinished(() => client.close());
  mock.handleUnary("/EnvironmentDelete", () => {
    throw new ClientError("/EnvironmentDelete", Status.NOT_FOUND, "Not found");
  });
  await expect(client.environments.objects.delete("prod")).rejects.toThrow(
    NotFoundError,
  );
  mock.assertExhausted();
});

test.each(["create", "list", "delete"] as const)(
  "EnvironmentObjects propagates RPC failures (%s)",
  async (operation) => {
    const { mockClient: client, mockCpClient: mock } = createMockModalClients();
    onTestFinished(() => client.close());
    const rpcName = {
      create: "/EnvironmentCreate",
      list: "/EnvironmentList",
      delete: "/EnvironmentDelete",
    }[operation];
    const error = new ClientError(
      rpcName,
      Status.PERMISSION_DENIED,
      "Access denied",
    );
    mock.handleUnary(rpcName, () => {
      throw error;
    });
    const result =
      operation === "list"
        ? client.environments.objects.list().next()
        : client.environments.objects[operation]("prod");
    await expect(result).rejects.toBe(error);
    mock.assertExhausted();
  },
);

test.each([false, true])(
  "EnvironmentFromName (createIfMissing=%s)",
  async (createIfMissing) => {
    const { mockClient: client, mockCpClient: mock } = createMockModalClients();
    onTestFinished(() => client.close());
    mock.handleUnary("/EnvironmentGetOrCreate", (req) => {
      expect(req).toEqual(
        EnvironmentGetOrCreateRequest.create({
          deploymentName: "prod",
          objectCreationType: createIfMissing
            ? ObjectCreationType.OBJECT_CREATION_TYPE_CREATE_IF_MISSING
            : ObjectCreationType.OBJECT_CREATION_TYPE_UNSPECIFIED,
        }),
      );
      return EnvironmentGetOrCreateResponse.create({
        environmentId: "en-prod",
        metadata: { name: "prod" },
      });
    });

    const environment = await client.environments.fromName(
      "prod",
      createIfMissing ? { createIfMissing } : undefined,
    );
    expect(environment).toBeInstanceOf(Environment);
    expect(environment.environmentId).toBe("en-prod");
    expect(environment.name).toBe("prod");
    mock.assertExhausted();
  },
);

test.each([
  "",
  "en-prod",
  "EN-prod",
  "En-prod",
  "_prod",
  "-prod",
  ".prod",
  "prod/v2",
  "prod:v2",
  "prod dev",
  "prod\n",
  "pröd",
  "a".repeat(65),
])(
  "EnvironmentFromName rejects invalid names before lookup (%j)",
  async (name) => {
    const { mockClient: client, mockCpClient: mock } = createMockModalClients();
    onTestFinished(() => client.close());
    await expect(client.environments.fromName(name)).rejects.toThrow(
      InvalidError,
    );
    mock.assertExhausted();
  },
);

test.each(["a", "9prod", "Prod_v2-dev", "prod.v2", "a".repeat(64)])(
  "EnvironmentFromName accepts valid names (%j)",
  async (name) => {
    const { mockClient: client, mockCpClient: mock } = createMockModalClients();
    onTestFinished(() => client.close());
    mock.handleUnary("/EnvironmentGetOrCreate", (req) => {
      expect(req).toEqual(
        EnvironmentGetOrCreateRequest.create({ deploymentName: name }),
      );
      return EnvironmentGetOrCreateResponse.create({
        environmentId: "en-test",
        metadata: { name },
      });
    });
    expect((await client.environments.fromName(name)).name).toBe(name);
    mock.assertExhausted();
  },
);

test.each(["prod", "alias"])(
  "EnvironmentAppsList uses the resolved Environment (%j)",
  async (name) => {
    const mock = new MockGrpcClient();
    const client = new ModalClient({
      cpClient: mock as any,
      tokenId: "test-token-id",
      tokenSecret: "test-token-secret",
      environment: "dev",
    });
    onTestFinished(() => client.close());
    const resolvedName = name === "alias" ? "main" : name;
    mock.handleUnary("/EnvironmentGetOrCreate", (req) => {
      expect(req).toEqual(
        EnvironmentGetOrCreateRequest.create({ deploymentName: name }),
      );
      return EnvironmentGetOrCreateResponse.create({
        environmentId: "en-resolved",
        metadata: { name: resolvedName },
      });
    });
    mock.handleUnary("/AppList", (req) => {
      expect(req).toEqual(
        AppListRequest.create({ environmentName: resolvedName }),
      );
      return AppListResponse.create({
        apps: [
          {
            appId: "ap-deployed",
            name: "libmodal-test",
            state: AppState.APP_STATE_DEPLOYED,
            metadata: {
              description: "Listed App description",
              functions: { echo: "fu-echo" },
              servers: { api: "fu-api" },
              lifecycle: {
                appState: AppState.APP_STATE_DEPLOYED,
                createdAt: 1700000000,
                createdBy: "creator",
                version: 2,
              },
            },
          },
          { appId: "ap-stopped", state: AppState.APP_STATE_STOPPED },
          {
            appId: "ap-ephemeral",
            description: "libmodal-test",
            state: AppState.APP_STATE_EPHEMERAL,
          },
          { appId: "ap-disabled", state: AppState.APP_STATE_DISABLED },
          {
            appId: "ap-stopping",
            name: "libmodal-test-support",
            state: AppState.APP_STATE_STOPPING,
          },
        ],
      });
    });

    const environment = await client.environments.fromName(name);
    expect(environment.name).toBe(resolvedName);
    const apps: App[] = [];
    for await (const app of environment.apps.list()) {
      apps.push(app);
    }
    expect(apps).toEqual([
      new App("ap-deployed", "libmodal-test", resolvedName),
      new App("ap-ephemeral", undefined, resolvedName),
      new App("ap-stopping", "libmodal-test-support", resolvedName),
    ]);
    expect(await apps[0].info()).toEqual({
      appId: "ap-deployed",
      description: "Listed App description",
      functions: { echo: "fu-echo" },
      servers: { api: "fu-api" },
      lifecycle: {
        state: "deployed",
        version: 2,
        createdAt: new Date(1700000000000),
        createdBy: "creator",
        deployedAt: undefined,
        deployedBy: undefined,
        stoppedAt: undefined,
        stoppedBy: undefined,
      },
    });
    mock.assertExhausted();
  },
);

test("EnvironmentAppsList is lazy and supports early exit", async () => {
  const { mockClient: client, mockCpClient: mock } = createMockModalClients();
  onTestFinished(() => client.close());
  mock.handleUnary("/EnvironmentGetOrCreate", () =>
    EnvironmentGetOrCreateResponse.create({
      environmentId: "en-prod",
      metadata: { name: "prod" },
    }),
  );
  let listCalls = 0;
  mock.handleUnary("/AppList", () => {
    listCalls++;
    return AppListResponse.create({
      apps: [
        { appId: "ap-first", state: AppState.APP_STATE_DEPLOYED },
        { appId: "ap-second", state: AppState.APP_STATE_DEPLOYED },
      ],
    });
  });
  const environment = await client.environments.fromName("prod");
  const apps = environment.apps.list();
  expect(listCalls).toBe(0);
  let count = 0;
  for await (const app of apps) {
    expect(app.appId).toBe("ap-first");
    count++;
    break;
  }
  expect(count).toBe(1);
  expect(listCalls).toBe(1);
  expect((await apps.next()).done).toBe(true);
  mock.assertExhausted();
});

test("EnvironmentAppsList returns an empty iterator", async () => {
  const { mockClient: client, mockCpClient: mock } = createMockModalClients();
  onTestFinished(() => client.close());
  mock.handleUnary("/EnvironmentGetOrCreate", () =>
    EnvironmentGetOrCreateResponse.create({
      environmentId: "en-prod",
      metadata: { name: "prod" },
    }),
  );
  mock.handleUnary("/AppList", () => AppListResponse.create());
  const environment = await client.environments.fromName("prod");
  expect((await environment.apps.list().next()).done).toBe(true);
  mock.assertExhausted();
});

test.each([Status.NOT_FOUND, Status.PERMISSION_DENIED])(
  "EnvironmentFromName handles RPC errors (%s)",
  async (code) => {
    const { mockClient: client, mockCpClient: mock } = createMockModalClients();
    onTestFinished(() => client.close());
    const error = new ClientError(
      "/EnvironmentGetOrCreate",
      code,
      "Lookup failed",
    );
    mock.handleUnary("/EnvironmentGetOrCreate", () => {
      throw error;
    });
    const result = client.environments.fromName("prod");
    if (code === Status.NOT_FOUND) {
      await expect(result).rejects.toThrow(NotFoundError);
    } else {
      await expect(result).rejects.toBe(error);
    }
    mock.assertExhausted();
  },
);

test("EnvironmentFromContext uses the client's environment", async () => {
  const mock = new MockGrpcClient();
  const client = new ModalClient({
    cpClient: mock as any,
    tokenId: "test-token-id",
    tokenSecret: "test-token-secret",
    environment: "dev",
  });
  onTestFinished(() => client.close());
  mock.handleUnary("/EnvironmentGetOrCreate", (req: unknown) => {
    expect(req).toMatchObject(
      EnvironmentGetOrCreateRequest.create({
        deploymentName: "dev",
        objectCreationType: ObjectCreationType.OBJECT_CREATION_TYPE_UNSPECIFIED,
      }),
    );
    return EnvironmentGetOrCreateResponse.create({
      environmentId: "en-dev",
      metadata: { name: "dev" },
    });
  });

  const environment = await client.environments.fromContext();
  expect(environment).toBeInstanceOf(Environment);
  expect(environment.environmentId).toBe("en-dev");
  expect(environment.name).toBe("dev");
  mock.assertExhausted();
});

test("EnvironmentFromContext throws NotFoundError for a missing environment", async () => {
  const mock = new MockGrpcClient();
  const client = new ModalClient({
    cpClient: mock as any,
    tokenId: "test-token-id",
    tokenSecret: "test-token-secret",
    environment: "missing",
  });
  onTestFinished(() => client.close());
  mock.handleUnary("/EnvironmentGetOrCreate", () => {
    throw new ClientError(
      "/EnvironmentGetOrCreate",
      Status.NOT_FOUND,
      "Lookup failed",
    );
  });

  await expect(client.environments.fromContext()).rejects.toThrow(
    new NotFoundError("Environment 'missing' not found"),
  );
  mock.assertExhausted();
});

test("EnvironmentAppsList propagates RPC errors", async () => {
  const { mockClient: client, mockCpClient: mock } = createMockModalClients();
  onTestFinished(() => client.close());
  mock.handleUnary("/EnvironmentGetOrCreate", () =>
    EnvironmentGetOrCreateResponse.create({
      environmentId: "en-prod",
      metadata: { name: "prod" },
    }),
  );
  const error = new ClientError(
    "/AppList",
    Status.PERMISSION_DENIED,
    "Access denied",
  );
  mock.handleUnary("/AppList", () => {
    throw error;
  });
  const environment = await client.environments.fromName("prod");
  await expect(environment.apps.list().next()).rejects.toBe(error);
  mock.assertExhausted();
});

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
