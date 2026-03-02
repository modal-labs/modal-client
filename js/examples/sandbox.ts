import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});
const image = modal.images.fromRegistry("alpine:3.21");

const sb = await modal.sandboxes.create(app, image, { command: ["cat"] });
console.log("Sandbox:", sb.sandboxId);

const sbFromId = await modal.sandboxes.fromId(sb.sandboxId);
console.log("Queried Sandbox from ID:", sbFromId.sandboxId);

await sb.stdin.writeText("this is input that should be mirrored by cat");
await sb.stdin.close();
console.log("output:", await sb.stdout.readText());

await sb.terminate();
