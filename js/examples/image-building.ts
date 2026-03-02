import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});

const image = modal.images
  .fromRegistry("alpine:3.21")
  .dockerfileCommands(["RUN apk add --no-cache curl=$CURL_VERSION"], {
    secrets: [
      await modal.secrets.fromObject({
        CURL_VERSION: "8.12.1-r1",
      }),
    ],
  })
  .dockerfileCommands(["ENV SERVER=ipconfig.me"]);

const sb = await modal.sandboxes.create(app, image, {
  command: ["sh", "-c", "curl -Ls $SERVER"],
});
console.log("Created Sandbox with ID:", sb.sandboxId);

console.log("Sandbox output:", await sb.stdout.readText());
await sb.terminate();
