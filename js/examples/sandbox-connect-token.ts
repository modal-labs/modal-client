import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});

// Create a Sandbox with Python's built-in HTTP server.
const image = modal.images.fromRegistry("python:3.12-alpine");
// Connect Tokens route requests to port 8080 by default.
// Pass `port` to route to a different container port.
const sb = await modal.sandboxes.create(app, image, {
  command: ["python3", "-m", "http.server", "8000"],
});

const creds = await sb.createConnectToken({ userMetadata: "abc", port: 8000 });
console.log(`Got url: ${creds.url}, credentials: ${creds.token}`);

console.log("\nConnecting to HTTP server...");
const response = await fetch(creds.url, {
  headers: {
    Authorization: `Bearer ${creds.token}`,
  },
});

console.log(`Response status: ${response.status}`);
console.log(`Response body:\n${await response.text()}`);

await sb.terminate();
