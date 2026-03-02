import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});
const image = modal.images.fromRegistry("alpine:3.21");

// Create a Sandbox that waits for input, then exits with code 42
const sb = await modal.sandboxes.create(app, image, {
  command: ["sh", "-c", "read line; exit 42"],
});

console.log("Started Sandbox:", sb.sandboxId);

console.log("Poll result while running:", await sb.poll());

console.log("\nSending input to trigger completion...");
await sb.stdin.writeText("hello, goodbye");
await sb.stdin.close();

const exitCode = await sb.wait();
console.log("\nSandbox completed with exit code:", exitCode);
console.log("Poll result after completion:", await sb.poll());
