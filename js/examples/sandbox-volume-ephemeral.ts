import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});
const image = modal.images.fromRegistry("alpine:3.21");

const volume = await modal.volumes.ephemeral();

const writerSandbox = await modal.sandboxes.create(app, image, {
  command: [
    "sh",
    "-c",
    "echo 'Hello from writer Sandbox!' > /mnt/volume/message.txt",
  ],
  volumes: { "/mnt/volume": volume },
});
console.log("Writer Sandbox:", writerSandbox.sandboxId);

await writerSandbox.wait();
console.log("Writer finished");
await writerSandbox.terminate();

const readerSandbox = await modal.sandboxes.create(app, image, {
  command: ["cat", "/mnt/volume/message.txt"],
  volumes: { "/mnt/volume": volume.readOnly() },
});
console.log("Reader Sandbox:", readerSandbox.sandboxId);
console.log("Reader output:", await readerSandbox.stdout.readText());

await readerSandbox.terminate();
volume.closeEphemeral();
