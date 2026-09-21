import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});

const image = await modal.images.fromRegistry("alpine:3.21").build(app);

const volume = await modal.volumes.ephemeral();

const sb = await modal.sandboxes.create(app, image, {
  command: ["sleep", "infinity"],
});
console.log("Started Sandbox:", sb.sandboxId);

try {
  const writer = await sb.experimentalSidecars.create("writer", image, {
    command: [
      "sh",
      "-c",
      "echo 'Hello from the writer sidecar!' > /mnt/volume/message.txt",
    ],
    volumes: { "/mnt/volume": volume },
  });
  console.log("Writer sidecar finished with exit code:", await writer.wait());

  const reader = await sb.experimentalSidecars.create("reader", image, {
    command: ["sleep", "100"],
    volumes: { "/mnt/volume": volume.withMountOptions({ readOnly: true }) },
  });

  const proc = await reader.exec(["cat", "/mnt/volume/message.txt"]);
  const output = await proc.stdout.readText();
  await proc.wait();
  console.log("Reader sidecar output:", output.trim());
} finally {
  await sb.terminate();
  volume.closeEphemeral();
}
