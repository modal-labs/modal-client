import { tc } from "../test-support/test-client";
import { expect, test, onTestFinished } from "vitest";

test("SandboxMountDirectoryEmpty", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("debian:12-slim");

  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  await (await sb.exec(["mkdir", "-p", "/mnt/empty"])).wait();
  await sb.mountImage("/mnt/empty");

  const dirCheck = await sb.exec(["test", "-d", "/mnt/empty"]);
  expect(await dirCheck.wait()).toBe(0);
});

test("SandboxMountDirectoryWithImage", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const baseImage = tc.images.fromRegistry("debian:12-slim");

  const sb1 = await tc.sandboxes.create(app, baseImage);

  const echoProc = await sb1.exec([
    "sh",
    "-c",
    "echo -n 'mounted content' > /tmp/test.txt",
  ]);
  await echoProc.wait();

  const mountImage = await sb1.snapshotFilesystem();
  expect(mountImage.imageId).toMatch(/^im-/);

  await sb1.terminate();

  const sb2 = await tc.sandboxes.create(app, baseImage);
  onTestFinished(async () => await sb2.terminate());

  await (await sb2.exec(["mkdir", "-p", "/mnt/data"])).wait();
  await sb2.mountImage("/mnt/data", mountImage);

  const catProc = await sb2.exec(["cat", "/mnt/data/tmp/test.txt"]);
  const output = await catProc.stdout.readText();
  expect(output).toBe("mounted content");
});

test("SandboxSnapshotDirectory", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const baseImage = tc.images.fromRegistry("debian:12-slim");

  const sb1 = await tc.sandboxes.create(app, baseImage);

  await (await sb1.exec(["mkdir", "-p", "/mnt/data"])).wait();
  await sb1.mountImage("/mnt/data");

  const echoProc = await sb1.exec([
    "sh",
    "-c",
    "echo -n 'snapshot test content' > /mnt/data/snapshot.txt",
  ]);
  await echoProc.wait();

  const snapshotImage = await sb1.snapshotDirectory("/mnt/data");
  expect(snapshotImage.imageId).toMatch(/^im-/);

  await sb1.terminate();

  const sb2 = await tc.sandboxes.create(app, baseImage);
  onTestFinished(async () => await sb2.terminate());

  await (await sb2.exec(["mkdir", "-p", "/mnt/data"])).wait();
  await sb2.mountImage("/mnt/data", snapshotImage);

  const catProc = await sb2.exec(["cat", "/mnt/data/snapshot.txt"]);
  const output = await catProc.stdout.readText();
  expect(output).toBe("snapshot test content");
});

test("SandboxMountDirectoryWithUnbuiltImageThrows", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const baseImage = tc.images.fromRegistry("debian:12-slim");

  const sb = await tc.sandboxes.create(app, baseImage);
  onTestFinished(async () => await sb.terminate());

  await (await sb.exec(["mkdir", "-p", "/mnt/data"])).wait();

  const unbuiltImage = tc.images.fromRegistry("alpine:3.21");
  expect(unbuiltImage.imageId).toBe("");

  await expect(sb.mountImage("/mnt/data", unbuiltImage)).rejects.toThrow(
    "Image must be built before mounting",
  );
});
