import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});
const image = modal.images.fromRegistry("alpine/curl:8.14.1");

const proxy = await modal.proxies.fromName("libmodal-test-proxy", {
  environment: "libmodal",
});
console.log("Using Proxy with ID:", proxy.proxyId);

const sb = await modal.sandboxes.create(app, image, {
  proxy,
});
console.log("Created Sandbox with Proxy:", sb.sandboxId);

try {
  const p = await sb.exec(["curl", "-s", "ifconfig.me"]);
  const ip = await p.stdout.readText();

  console.log("External IP:", ip.trim());
} finally {
  await sb.terminate();
}
