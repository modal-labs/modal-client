import { ModalClient } from "modal";

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});

const image = await modal.images.fromRegistry("alpine:3.21").build(app);

const sb = await modal.sandboxes.create(app, image, {
  command: ["sleep", "infinity"],
});

try {
  const container = await sb.experimentalSidecars.create("worker", image, {
    command: ["sleep", "100"],
  });

  await container.mountImage("/workspace");

  const writeProc = await container.exec([
    "sh",
    "-c",
    "echo persisted > /workspace/message.txt",
  ]);
  await writeProc.wait();

  const workspaceSnapshot = await container.snapshotDirectory("/workspace");

  await container.terminate({ wait: true });

  const nextContainer = await sb.experimentalSidecars.create(
    "next-worker",
    image,
    {
      command: ["sleep", "100"],
    },
  );
  await nextContainer.mountImage("/workspace", workspaceSnapshot);
  const readProc = await nextContainer.exec(["cat", "/workspace/message.txt"]);
  console.log(
    "Restored workspace contains:",
    (await readProc.stdout.readText()).trim(),
  );
  await readProc.wait();
  await nextContainer.unmountImage("/workspace");
  await nextContainer.terminate({ wait: true });
} finally {
  await sb.terminate();
}
