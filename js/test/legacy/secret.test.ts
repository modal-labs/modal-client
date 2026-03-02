import { App, Image, Secret } from "modal";
import { expect, test } from "vitest";

test("SecretFromName", async () => {
  const secret = await Secret.fromName("libmodal-test-secret");
  expect(secret).toBeDefined();
  expect(secret.secretId).toBeDefined();
  expect(secret.secretId).toMatch(/^st-/);
  expect(secret.name).toBe("libmodal-test-secret");

  const promise = Secret.fromName("missing-secret");
  await expect(promise).rejects.toThrowError(
    /Secret 'missing-secret' not found/,
  );
});

test("SecretFromNameWithRequiredKeys", async () => {
  const secret = await Secret.fromName("libmodal-test-secret", {
    requiredKeys: ["a", "b", "c"],
  });
  expect(secret).toBeDefined();

  const promise = Secret.fromName("libmodal-test-secret", {
    requiredKeys: ["a", "b", "c", "missing-key"],
  });
  await expect(promise).rejects.toThrowError(
    /Secret is missing key\(s\): missing-key/,
  );
});

test("SecretFromObject", async () => {
  const secret = await Secret.fromObject({ key: "value" });
  expect(secret).toBeDefined();

  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = Image.fromRegistry("alpine:3.21");

  const sandbox = await app.createSandbox(image, {
    command: ["printenv", "key"],
    secrets: [secret],
  });

  const output = await sandbox.stdout.readText();
  expect(output).toBe("value\n");
});

test("SecretFromObjectInvalid", async () => {
  // @ts-expect-error testing runtime validation
  await expect(Secret.fromObject({ key: 123 })).rejects.toThrowError(
    /entries must be an object mapping string keys to string values/,
  );
});
