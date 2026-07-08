import { tc } from "../test-support/test-client";
import { afterEach, expect, onTestFinished, test, vi } from "vitest";
import {
  mergeEnvIntoSecrets,
  splitEnvDictAndResolvableSecrets,
  hydrateSecrets,
  secretEnvDictHydrator,
  Secret,
  SecretFromObjectHydrator,
} from "../src/secret";
import {
  createMockModalClients,
  MockGrpcClient,
} from "../test-support/grpc_mock";
import { NotFoundError } from "../src/errors";
import { ClientError, Status } from "nice-grpc";
import {
  GenericResult_GenericStatus,
  ObjectCreationType,
} from "../proto/modal_proto/api";

const V1_SANDBOX_ID = "sb-nGEijt9WbBMlGrsPH9FOaC";
const V2_SANDBOX_ID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV";

afterEach(() => {
  vi.unstubAllEnvs();
});

// Registers mock handlers for the RPCs that a Sandbox create issues before the
// SandboxCreate/SandboxCreateV2 call: app lookup and image build. The image
// builder version is pinned via MODAL_IMAGE_BUILDER_VERSION so no
// EnvironmentGetOrCreate RPC is issued (see createMockClientWithPinnedBuilder).
function registerSandboxCreateDeps(mock: MockGrpcClient): void {
  mock.handleUnary("/AppGetOrCreate", () => ({ appId: "ap-123" }));
  mock.handleUnary("/ImageGetOrCreate", () => ({
    imageId: "im-123",
    result: { status: GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS },
  }));
}

// Creates mock clients with the image builder version pinned so building an
// Image does not issue an EnvironmentGetOrCreate RPC. Must be called after
// stubbing so the pinned version is read into the profile at construction.
function createMockClientWithPinnedBuilder(): ReturnType<
  typeof createMockModalClients
> {
  vi.stubEnv("MODAL_IMAGE_BUILDER_VERSION", "2025.06");
  return createMockModalClients();
}

test("SecretFromName", async () => {
  const secret = await tc.secrets.fromName("libmodal-test-secret");
  expect(secret.secretId).toMatch(/^st-/);
  expect(secret.name).toBe("libmodal-test-secret");

  const promise = tc.secrets.fromName("missing-secret");
  await expect(promise).rejects.toThrowError(
    /Secret 'missing-secret' not found/,
  );
});

test("SecretFromNameWithRequiredKeys", async () => {
  const secret = await tc.secrets.fromName("libmodal-test-secret", {
    requiredKeys: ["a", "b", "c"],
  });
  expect(secret.secretId).toMatch(/^st-/);

  const promise = tc.secrets.fromName("libmodal-test-secret", {
    requiredKeys: ["a", "b", "c", "missing-key"],
  });
  await expect(promise).rejects.toThrowError(
    /Secret is missing key\(s\): missing-key/,
  );
});

test("SecretFromObject", async () => {
  const secret = await tc.secrets.fromObject({ key: "value" });
  // fromObject is lazy: no Secret is created on the server until it is used.
  expect(secret.secretId).toBe("");

  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image, {
    command: ["printenv", "key"],
    secrets: [secret],
  });
  onTestFinished(async () => await sb.terminate());

  // Using the Secret in a Sandbox hydrates it into a server-side Secret.
  expect(secret.secretId).toMatch(/^st-/);

  const output = await sb.stdout.readText();
  expect(output).toBe("value\n");
});

test("SecretFromObjectInvalid", async () => {
  await expect(
    // @ts-expect-error testing runtime validation
    tc.secrets.fromObject({ key: 123 }),
  ).rejects.toThrowError(
    /entries must be an object mapping string keys to string values/,
  );
});

test("SecretFromObjectInvalidKey", async () => {
  // Keys that aren't valid environment variable names are rejected up front.
  for (const badKey of ["1KEY", "with-dash", "with space", "with.dot", ""]) {
    await expect(
      tc.secrets.fromObject({ [badKey]: "value" }),
    ).rejects.toThrowError(/is invalid for environment variables/);
  }

  // Valid keys (letters, numbers, underscores, not starting with a number).
  const secret = await tc.secrets.fromObject({ VALID_KEY_1: "value", _x: "y" });
  expect(secretEnvDictHydrator(secret)?.envDict).toEqual({
    VALID_KEY_1: "value",
    _x: "y",
  });
});

test("SecretFromObject is lazy and makes no control-plane call", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  // No RPC handlers are registered: fromObject must not make any control-plane
  // call, otherwise the mock would throw on an unexpected RPC.
  const secret = await mc.secrets.fromObject({ key: "value" });
  expect(secret.secretId).toBe("");
  expect(secretEnvDictHydrator(secret)?.envDict).toEqual({ key: "value" });

  mock.assertExhausted();
});

test("mergeEnvIntoSecrets merges env with existing secrets", async () => {
  const existingSecret = await tc.secrets.fromObject({ A: "1" });
  const env = { B: "2", C: "3" };

  const result = await mergeEnvIntoSecrets(tc, env, [existingSecret]);

  expect(result).toHaveLength(2);
  expect(result[0]).toBe(existingSecret);
  // The appended env Secret is lazy: no secretId until hydrated.
  expect(result[1].secretId).toBe("");
  expect(secretEnvDictHydrator(result[1])?.envDict).toEqual({ B: "2", C: "3" });
});

test("mergeEnvIntoSecrets with only env parameter", async () => {
  const env = { B: "2", C: "3" };

  const result = await mergeEnvIntoSecrets(tc, env);

  expect(result).toHaveLength(1);
  expect(result[0].secretId).toBe("");
  expect(secretEnvDictHydrator(result[0])?.envDict).toEqual({ B: "2", C: "3" });
});

test("mergeEnvIntoSecrets with empty env object returns existing secrets", async () => {
  const existingSecret = await tc.secrets.fromObject({ A: "1" });
  const env = {};

  const result = await mergeEnvIntoSecrets(tc, env, [existingSecret]);

  expect(result).toHaveLength(1);
  expect(result[0]).toBe(existingSecret);
});

test("mergeEnvIntoSecrets with undefined env returns existing secrets", async () => {
  const existingSecret = await tc.secrets.fromObject({ A: "1" });

  const result = await mergeEnvIntoSecrets(tc, undefined, [existingSecret]);

  expect(result).toHaveLength(1);
  expect(result[0]).toBe(existingSecret);
});

test("mergeEnvIntoSecrets with only existing secrets", async () => {
  const secret1 = await tc.secrets.fromObject({ A: "1" });
  const secret2 = await tc.secrets.fromObject({ B: "2" });

  const result = await mergeEnvIntoSecrets(tc, undefined, [secret1, secret2]);

  expect(result).toHaveLength(2);
  expect(result[0]).toBe(secret1);
  expect(result[1]).toBe(secret2);
});

test("mergeEnvIntoSecrets with no env and no secrets returns empty array", async () => {
  const result = await mergeEnvIntoSecrets(tc);

  expect(result).toHaveLength(0);
  expect(result).toEqual([]);
});

test("splitEnvDictAndResolvableSecrets partitions local and resolvable secrets", () => {
  const local1 = new Secret(
    "",
    undefined,
    new SecretFromObjectHydrator({ A: "1", B: "2" }),
  );
  const local2 = new Secret(
    "",
    undefined,
    new SecretFromObjectHydrator({ B: "override", C: "3" }),
  );
  const named = new Secret("st-named");

  // Local Secrets are merged in list order (so local2's B wins); the named
  // Secret is kept in the resolvable list.
  const [envDict, resolvable] = splitEnvDictAndResolvableSecrets([
    local1,
    named,
    local2,
  ]);

  expect(envDict).toEqual({ A: "1", B: "override", C: "3" });
  expect(resolvable).toEqual([named]);
});

test("splitEnvDictAndResolvableSecrets with no local secrets", () => {
  const named = new Secret("st-named");
  const [envDict, resolvable] = splitEnvDictAndResolvableSecrets([named]);

  expect(envDict).toEqual({});
  expect(resolvable).toEqual([named]);
});

test("hydrateSecrets rejects null secrets", async () => {
  await expect(
    // @ts-expect-error testing runtime validation
    hydrateSecrets(tc, [new Secret("st-1"), null]),
  ).rejects.toThrowError(/secret at index 1 must not be null/);
});

test("hydrateSecrets skips already-hydrated secrets", () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  // Secrets that already have a secretId have no env dict, so no RPC is
  // attempted (an unexpected RPC would throw against the mock).
  return hydrateSecrets(mc, [new Secret("st-1"), new Secret("st-2")]).then(() =>
    mock.assertExhausted(),
  );
});

test("SandboxCreate hydrates a fromObject Secret", async () => {
  const { mockClient: mc, mockCpClient: mock } =
    createMockClientWithPinnedBuilder();
  registerSandboxCreateDeps(mock);

  mock.handleUnary("/SecretGetOrCreate", (req: any) => {
    // The lazy fromObject Secret is hydrated as an ephemeral Secret on create.
    expect(req.objectCreationType).toBe(
      ObjectCreationType.OBJECT_CREATION_TYPE_EPHEMERAL,
    );
    expect(req.envDict).toEqual({ FOO: "bar" });
    return { secretId: "st-ephemeral" };
  });
  mock.handleUnary("/SandboxCreate", (req: any) => {
    expect(req.definition?.secretIds).toContain("st-ephemeral");
    return { sandboxId: V1_SANDBOX_ID };
  });

  const app = await mc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = mc.images.fromRegistry("alpine:3.21");

  const secret = await mc.secrets.fromObject({ FOO: "bar" });
  const sb = await mc.sandboxes.create(app, image, { secrets: [secret] });
  expect(sb.sandboxId).toBe(V1_SANDBOX_ID);

  // The Secret was hydrated in place.
  expect(secret.secretId).toBe("st-ephemeral");

  mock.assertExhausted();
});

test("ExperimentalCreate passes env as ephemeral secrets", async () => {
  const { mockClient: mc, mockCpClient: mock } =
    createMockClientWithPinnedBuilder();
  registerSandboxCreateDeps(mock);

  // Note: no SecretGetOrCreate handler is registered. The V2 path must pass env
  // vars via ephemeralSecrets rather than creating a Secret for them, so no
  // SecretGetOrCreate RPC should occur.
  mock.handleUnary("/SandboxCreateV2", (req: any) => {
    expect(req.ephemeralSecrets?.contents).toEqual({ FOO: "bar" });
    expect(req.definition?.secretIds ?? []).toEqual([]);
    return { sandboxId: V2_SANDBOX_ID, taskId: "ta-v2-123", tunnels: [] };
  });

  const app = await mc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = mc.images.fromRegistry("alpine:3.21");

  const sb = await mc.sandboxes.experimentalCreate(app, image, {
    env: { FOO: "bar" },
  });
  expect(sb.sandboxId).toBe(V2_SANDBOX_ID);

  mock.assertExhausted();
});

test("ExperimentalCreate rejects invalid env var keys", async () => {
  const { mockClient: mc, mockCpClient: mock } =
    createMockClientWithPinnedBuilder();
  registerSandboxCreateDeps(mock);

  const app = await mc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = mc.images.fromRegistry("alpine:3.21");

  // env vars are sent directly to the worker in the V2 path, so invalid keys
  // must be caught client-side before any SandboxCreateV2 RPC is issued (no
  // handler is registered, so an unexpected RPC would throw).
  await expect(
    mc.sandboxes.experimentalCreate(app, image, {
      env: { "bad-key": "value" },
    }),
  ).rejects.toThrowError(/is invalid for environment variables/);
});

test("ExperimentalCreate passes a fromObject Secret as ephemeral secrets", async () => {
  const { mockClient: mc, mockCpClient: mock } =
    createMockClientWithPinnedBuilder();
  registerSandboxCreateDeps(mock);

  // No SecretGetOrCreate handler is registered. Locally-created fromObject
  // Secrets must be folded into ephemeralSecrets in the V2 path rather than
  // hydrated into a server-side Secret, so no SecretGetOrCreate RPC should occur.
  mock.handleUnary("/SandboxCreateV2", (req: any) => {
    // params.env takes precedence over the fromObject value on key collisions.
    expect(req.ephemeralSecrets?.contents).toEqual({
      FOO: "from-env",
      BAZ: "qux",
    });
    expect(req.definition?.secretIds ?? []).toEqual([]);
    return { sandboxId: V2_SANDBOX_ID, taskId: "ta-v2-123", tunnels: [] };
  });

  const app = await mc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = mc.images.fromRegistry("alpine:3.21");

  const secret = await mc.secrets.fromObject({
    FOO: "from-secret",
    BAZ: "qux",
  });
  const sb = await mc.sandboxes.experimentalCreate(app, image, {
    secrets: [secret],
    env: { FOO: "from-env" },
  });
  expect(sb.sandboxId).toBe(V2_SANDBOX_ID);

  // The fromObject Secret was never hydrated into a server-side Secret.
  expect(secret.secretId).toBe("");

  mock.assertExhausted();
});

test("SecretDelete success", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SecretGetOrCreate", () => ({
    secretId: "st-test-123",
  }));

  mock.handleUnary("/SecretDelete", (req: any) => {
    expect(req.secretId).toBe("st-test-123");
    return {};
  });

  await mc.secrets.delete("test-secret");
  mock.assertExhausted();
});

test("SecretDelete with allowMissing=true", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SecretGetOrCreate", () => {
    throw new NotFoundError("Secret 'missing' not found");
  });

  await mc.secrets.delete("missing", { allowMissing: true });
  mock.assertExhausted();
});

test("SecretDelete with allowMissing=true when delete RPC returns NOT_FOUND", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SecretGetOrCreate", () => ({
    secretId: "st-test-123",
  }));

  mock.handleUnary("/SecretDelete", () => {
    throw new ClientError(
      "/modal.client.ModalClient/SecretDelete",
      Status.NOT_FOUND,
      "No Secret with ID 'st-test-123' found",
    );
  });

  await mc.secrets.delete("test-secret", { allowMissing: true });
  mock.assertExhausted();
});

test("SecretDelete with allowMissing=false throws", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SecretGetOrCreate", () => {
    throw new NotFoundError("Secret 'missing' not found");
  });

  await expect(
    mc.secrets.delete("missing", { allowMissing: false }),
  ).rejects.toThrow(NotFoundError);
});
