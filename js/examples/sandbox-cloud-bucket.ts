import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});
const image = modal.images.fromRegistry("alpine:3.21");

const secret = await modal.secrets.fromName("libmodal-aws-bucket-secret");

const sb = await modal.sandboxes.create(app, image, {
  command: ["sh", "-c", "ls -la /mnt/s3-bucket"],
  cloudBucketMounts: {
    "/mnt/s3-bucket": modal.cloudBucketMounts.create("my-s3-bucket", {
      secret,
      keyPrefix: "data/",
      readOnly: true,
    }),
  },
});

console.log("S3 Sandbox:", sb.sandboxId);
console.log(
  "Sandbox directory listing of /mnt/s3-bucket:",
  await sb.stdout.readText(),
);

await sb.terminate();
