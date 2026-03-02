import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});
const image = modal.images.fromAwsEcr(
  "459781239556.dkr.ecr.us-east-1.amazonaws.com/ecr-private-registry-test-7522615:python",
  await modal.secrets.fromName("libmodal-aws-ecr-test", {
    requiredKeys: ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
  }),
);

// Spawn a Sandbox running a simple Python version of the "cat" command.
const sb = await modal.sandboxes.create(app, image, {
  command: ["python", "-c", `import sys; sys.stdout.write(sys.stdin.read())`],
});
console.log("Sandbox:", sb.sandboxId);

await sb.stdin.writeText(
  "this is input that should be mirrored by the Python one-liner",
);
await sb.stdin.close();
console.log("output:", await sb.stdout.readText());

await sb.terminate();
