import { tc } from "../test-support/test-client";
import { expect, onTestFinished, test } from "vitest";
import { mergeEnvIntoSecrets } from "../src/secret";
import { createMockModalClients } from "../test-support/grpc_mock";
import { NotFoundError } from "../src/errors";
import { ClientError, Status } from "nice-grpc";

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
  expect(secret.secretId).toMatch(/^st-/);

  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image, {
    command: ["printenv", "key"],
    secrets: [secret],
  });
  onTestFinished(async () => await sb.terminate());

  const output = await sb.stdout.readText();
  expect(output).toBe("value\n");
});

test("SecretFromObjectInvalid", async () => {
  // @ts-expect-error testing runtime validation
  await expect(tc.secrets.fromObject({ key: 123 })).rejects.toThrowError(
    /entries must be an object mapping string keys to string values/,
  );
});

test("mergeEnvIntoSecrets merges env with existing secrets", async () => {
  const existingSecret = await tc.secrets.fromObject({ A: "1" });
  const env = { B: "2", C: "3" };

  const result = await mergeEnvIntoSecrets(tc, env, [existingSecret]);

  expect(result).toHaveLength(2);
  expect(result[0]).toBe(existingSecret);
  expect(result[1].secretId).toMatch(/^st-/);
});

test("mergeEnvIntoSecrets with only env parameter", async () => {
  const env = { B: "2", C: "3" };

  const result = await mergeEnvIntoSecrets(tc, env);

  expect(result).toHaveLength(1);
  expect(result[0].secretId).toMatch(/^st-/);
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
