import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});
const image = modal.images.fromRegistry("alpine:3.21");

// Create a sandbox with only modal.com allowed (empty CIDR list blocks raw IP traffic).
const sb = await modal.sandboxes.create(app, image, {
  command: ["sleep", "infinity"],
  outboundDomainAllowlist: ["modal.com"],
  outboundCidrAllowlist: [],
});
console.log("Created Sandbox:", sb.sandboxId);

// Try to reach example.com — should fail because only modal.com is allowed.
const p1 = await sb.exec([
  "wget",
  "-q",
  "-O",
  "-",
  "--timeout=5",
  "http://example.com",
]);
console.log("wget example.com (blocked): exit=", await p1.wait());

// Unblock: widen the policy to allow all domains.
await sb.updateNetworkPolicy({
  outboundDomainAllowlist: ["*"],
  outboundCidrAllowlist: ["0.0.0.0/0"],
});
console.log("Widened policy to allow all domains.");

// Try again — should succeed now.
const p2 = await sb.exec([
  "wget",
  "-q",
  "-O",
  "-",
  "--timeout=5",
  "http://example.com",
]);
const body = await p2.stdout.readText();
console.log(
  "wget example.com (allowed): exit=",
  await p2.wait(),
  "body_len=",
  body.length,
);

await sb.terminate();
