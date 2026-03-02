import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});
const image = modal.images.fromRegistry("alpine:3.21");

const secret = await modal.secrets.fromName("libmodal-test-secret", {
  requiredKeys: ["c"],
});

const ephemeralSecret = await modal.secrets.fromObject({
  d: "123",
});

const sb = await modal.sandboxes.create(app, image, {
  command: ["sh", "-lc", "printenv | grep -E '^c|d='"],
  secrets: [secret, ephemeralSecret],
});

console.log("Sandbox created:", sb.sandboxId);

console.log("Sandbox environment variables from Secrets:");
console.log(await sb.stdout.readText());
