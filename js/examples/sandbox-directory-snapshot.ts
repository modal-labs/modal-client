// This example demonstrates the directory snapshots feature, which allows you to:
// - Take a snapshot of a directory in a Sandbox using `Sandbox.snapshotDirectory`,
//   which will create a new Modal Image.
// - Mount a Modal Image at a specific directory within an already running Sandbox
//   using `Sandbox.mountImage`.
//
// For example, you can use this to mount user specific dependencies into a running
// Sandbox, that is started with a base Image with shared system dependencies. This
// way, you can update system dependencies and user projects independently.

import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});
const baseImage = modal.images
  .fromRegistry("alpine:3.21")
  .dockerfileCommands(["RUN apk add --no-cache git"]);

const sb = await modal.sandboxes.create(app, baseImage);

const gitClone = await sb.exec([
  "git",
  "clone",
  "https://github.com/modal-labs/libmodal.git",
  "/repo",
]);
await gitClone.wait();

const repoSnapshot = await sb.snapshotDirectory("/repo");
console.log(
  "Took a snapshot of the /repo directory, Image ID:",
  repoSnapshot.imageId,
);

await sb.terminate();

// Start a new Sandbox, and mount the repo directory:
const sb2 = await modal.sandboxes.create(app, baseImage);

await (await sb2.exec(["mkdir", "-p", "/repo"])).wait();
await sb2.mountImage("/repo", repoSnapshot);

const repoLs = await sb2.exec(["ls", "/repo"]);
console.log(
  "Contents of /repo directory in new Sandbox sb2:\n",
  await repoLs.stdout.readText(),
);

await sb2.terminate();
await modal.images.delete(repoSnapshot.imageId);
