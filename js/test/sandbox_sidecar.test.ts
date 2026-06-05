import { expect, onTestFinished, test } from "vitest";

import { tc } from "../test-support/test-client";
import {
  InvalidError,
  NotFoundError,
  SandboxFilesystemNotFoundError,
} from "../src/errors";
import type { Image } from "../src/image";
import type { Sandbox } from "../src/sandbox";

async function createSandbox(): Promise<Sandbox> {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");
  const sb = await tc.sandboxes.create(app, image, {
    command: ["sleep", "infinity"],
  });
  onTestFinished(async () => await sb.terminate());
  return sb;
}

async function buildAlpineImage(): Promise<Image> {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = await tc.images.fromRegistry("alpine:3.21").build(app);
  expect(image.imageId).toBeTruthy();
  return image;
}

test("SidecarBasicLifecycle", async () => {
  const sb = await createSandbox();
  const image = await buildAlpineImage();

  const container = await sb.experimentalSidecars.create("worker", image, {
    command: ["sleep", "100"],
  });
  expect(container.containerId).toBeTruthy();
  expect(container.containerName).toBe("worker");

  expect(await container.poll()).toBeNull();

  expect(await container.terminate({ wait: true })).toBe(137);
  expect(await container.wait()).toBe(137);
  expect(await container.poll()).toBe(137);

  const terminated = await sb.experimentalSidecars.list({
    includeTerminated: true,
  });
  expect(terminated.map((c) => c.containerName)).toEqual(["worker"]);
});

test("SidecarWaitAfterNaturalExit", async () => {
  const sb = await createSandbox();
  const image = await buildAlpineImage();

  const container = await sb.experimentalSidecars.create("oneshot", image, {
    command: ["sh", "-c", "exit 42"],
  });

  expect(await container.wait()).toBe(42);
  expect(await container.wait()).toBe(42);

  await expect(sb.experimentalSidecars.get("oneshot")).rejects.toThrowError(
    NotFoundError,
  );

  const got = await sb.experimentalSidecars.get("oneshot", {
    includeTerminated: true,
  });
  expect(got.containerId).toBe(container.containerId);

  const replacement = await sb.experimentalSidecars.create("oneshot", image, {
    command: ["sleep", "100"],
  });
  expect(replacement.containerId).not.toBe(container.containerId);

  const listed = await sb.experimentalSidecars.list({
    includeTerminated: true,
  });
  const ids = listed.map((c) => c.containerId);
  expect(ids).toContain(container.containerId);
  expect(ids).toContain(replacement.containerId);
});

test("SidecarCreateRejectsMainName", async () => {
  const sb = await createSandbox();
  const image = await buildAlpineImage();

  await expect(
    sb.experimentalSidecars.create("main", image, {
      command: ["sleep", "100"],
    }),
  ).rejects.toThrowError(InvalidError);

  await expect(
    sb.experimentalSidecars.create("", image, { command: ["sleep", "100"] }),
  ).rejects.toThrowError(InvalidError);

  await expect(sb.experimentalSidecars.get("main")).rejects.toThrowError(
    InvalidError,
  );
});

test("SidecarCreateImageMustBeBuilt", async () => {
  const sb = await createSandbox();

  const unbuilt = tc.images.fromRegistry("alpine:3.21");
  expect(unbuilt.imageId).toBe("");

  await expect(
    sb.experimentalSidecars.create("worker", unbuilt, {
      command: ["sleep", "100"],
    }),
  ).rejects.toThrowError(InvalidError);
});

test("SidecarCreateForwardsSecretsAndEnv", async () => {
  const secret = await tc.secrets.fromObject({ API_KEY: "secret-value" });

  const sb = await createSandbox();
  const image = await buildAlpineImage();

  const container = await sb.experimentalSidecars.create("worker", image, {
    command: ["sleep", "100"],
    env: { API_KEY: "override", PLAIN_ENV: "plain" },
    secrets: [secret],
  });

  const proc = await container.exec([
    "sh",
    "-c",
    `printf '%s:%s' "$API_KEY" "$PLAIN_ENV"`,
  ]);
  const output = await proc.stdout.readText();
  expect(await proc.wait()).toBe(0);
  expect(output).toBe("override:plain");
});

test("SidecarExec", async () => {
  const sb = await createSandbox();
  const image = await buildAlpineImage();

  const container = await sb.experimentalSidecars.create("worker", image, {
    command: ["sleep", "100"],
  });

  const proc = await container.exec(["echo", "hello"]);
  expect(await proc.stdout.readText()).toBe("hello\n");
  expect(await proc.wait()).toBe(0);
});

test("SidecarFilesystem", async () => {
  const sb = await createSandbox();
  const image = await buildAlpineImage();

  const container = await sb.experimentalSidecars.create("worker", image, {
    command: ["sleep", "100"],
  });

  await container.filesystem.writeText("hi from sidecar", "/tmp/sidecar-hello");
  expect(await container.filesystem.readText("/tmp/sidecar-hello")).toBe(
    "hi from sidecar",
  );

  // The main container should not see the file in the sidecar's filesystem.
  await expect(sb.filesystem.stat("/tmp/sidecar-hello")).rejects.toThrowError(
    SandboxFilesystemNotFoundError,
  );
});
