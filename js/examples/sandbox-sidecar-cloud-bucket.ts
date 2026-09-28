import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});

const image = await modal.images.fromRegistry("alpine:3.21").build(app);

const secret = await modal.secrets.fromName("libmodal-aws-bucket-secret");

const sb = await modal.sandboxes.create(app, image, {
  command: ["sleep", "infinity"],
});
console.log("Started Sandbox:", sb.sandboxId);

try {
  const sidecar = await sb.experimentalSidecars.create("reader", image, {
    command: ["sleep", "100"],
    cloudBucketMounts: {
      "/mnt/s3-bucket": modal.cloudBucketMounts.create("my-s3-bucket", {
        secret,
        keyPrefix: "data/",
        readOnly: true,
      }),
    },
  });

  const proc = await sidecar.exec(["ls", "-la", "/mnt/s3-bucket"]);
  const output = await proc.stdout.readText();
  await proc.wait();
  console.log("Sidecar directory listing of /mnt/s3-bucket:", output);
} finally {
  await sb.terminate();
}
