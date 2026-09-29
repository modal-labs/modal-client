import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});

const image = await modal.images.fromRegistry("alpine:3.21").build(app);

const sb = await modal.sandboxes.experimentalCreate(app, image, {
  command: ["sleep", "infinity"],
});
console.log("Started Sandbox:", sb.sandboxId);

try {
  // The token identifies this sidecar container, not the main container, and
  // can be exchanged for cloud credentials with OIDC federation.
  const sidecar = await sb.experimentalSidecars.create("worker", image, {
    command: ["sleep", "100"],
    includeOidcIdentityToken: true,
  });
  console.log("Started sidecar:", sidecar.containerId);

  const proc = await sidecar.exec([
    "sh",
    "-c",
    'test -n "$MODAL_IDENTITY_TOKEN" && echo "MODAL_IDENTITY_TOKEN is set in the sidecar"',
  ]);
  const output = await proc.stdout.readText();
  await proc.wait();
  console.log(output.trim());
} finally {
  await sb.terminate();
}
