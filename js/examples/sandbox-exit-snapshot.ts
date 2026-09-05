// This example creates a Sandbox with exit snapshots enabled, terminates it,
// and then starts a new Sandbox from the filesystem captured at exit.

import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});
const baseImage = modal.images.fromRegistry("alpine:3.21");

const sb = await modal.sandboxes.create(app, baseImage, {
  experimentalOptions: { enable_exit_snapshot: true },
});
console.log("Started Sandbox:", sb.sandboxId);

await (await sb.exec(["mkdir", "-p", "/app/data"])).wait();
await (
  await sb.exec([
    "sh",
    "-c",
    "echo 'This file was created in the first Sandbox' > /app/data/info.txt",
  ])
).wait();
console.log("Created file in first Sandbox");

await sb.terminate();
console.log("Terminated first Sandbox");

const exitSnapshotImage = await sb.experimentalGetExitSnapshot();
console.log("Exit snapshot created with Image ID:", exitSnapshotImage.imageId);

const sb2 = await modal.sandboxes.create(app, exitSnapshotImage);
console.log("\nStarted new Sandbox from exit snapshot:", sb2.sandboxId);

const proc = await sb2.exec(["cat", "/app/data/info.txt"]);
const info = await proc.stdout.readText();
console.log("File data read in second Sandbox:", info);

await sb2.terminate();
