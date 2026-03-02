import { ModalClient, AlreadyExistsError } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});
const image = modal.images.fromRegistry("alpine:3.21");

const sandboxName = `libmodal-example-named-sandbox`;

const sb = await modal.sandboxes.create(app, image, {
  name: sandboxName,
  command: ["cat"],
});

console.log(`Created Sandbox with name: ${sandboxName}`);
console.log(`Sandbox ID: ${sb.sandboxId}`);

try {
  await modal.sandboxes.create(app, image, {
    name: sandboxName,
    command: ["cat"],
  });
} catch (e) {
  if (e instanceof AlreadyExistsError) {
    console.log(
      "Trying to create one more Sandbox with the same name throws:",
      e.message,
    );
  } else {
    throw e;
  }
}

const sbFromName = await modal.sandboxes.fromName(
  "libmodal-example",
  sandboxName,
);
console.log(`Retrieved the same Sandbox from name: ${sbFromName.sandboxId}`);

await sbFromName.stdin.writeText("hello, named Sandbox");
await sbFromName.stdin.close();

console.log("Reading output:");
console.log(await sbFromName.stdout.readText());

await sb.terminate();
console.log("Sandbox terminated");
