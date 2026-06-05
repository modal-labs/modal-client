import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});

const image = await modal.images.fromRegistry("alpine:3.21").build(app);

const sb = await modal.sandboxes.create(app, image, {
  command: ["sleep", "infinity"],
});
console.log("Started Sandbox:", sb.sandboxId);

try {
  const container = await sb.experimentalSidecars.create("worker", image, {
    command: ["sleep", "100"],
  });
  console.log("Started sidecar:", container.containerId);

  const proc = await container.exec(
    ["sh", "-c", `echo "$GREETING from sidecar"`],
    { env: { GREETING: "hello" } },
  );
  const output = await proc.stdout.readText();
  await proc.wait();
  console.log("Sidecar said:", output.trim());

  const exitCode = await container.terminate({ wait: true });
  console.log("Sidecar terminated with exit code:", exitCode);
} finally {
  await sb.terminate();
}
