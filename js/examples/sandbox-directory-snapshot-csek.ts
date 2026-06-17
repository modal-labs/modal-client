// This example demonstrates customer-supplied encryption keys (CSEK) for
// Sandbox directory snapshots.
//
// Create and restore a CSEK-encrypted directory snapshot:
//
//   node --import tsx examples/sandbox-directory-snapshot-csek.ts
//
// Restore an existing CSEK-encrypted directory snapshot:
//
//   node --import tsx examples/sandbox-directory-snapshot-csek.ts \
//     --image-id=im-... \
//     --encryption-key=...
//
// You may pass --encryption-key to use your own base64-encoded key.
// If omitted, this example generates one and prints it so you can store it.

import { randomBytes } from "node:crypto";

import { type Image, ModalClient } from "modal";

const modal = new ModalClient();

const args = parseArgs(process.argv.slice(2));
const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});

if (args.imageId) {
  const encryptionKey = decodeKey(args.encryptionKey);
  await restoreSnapshot(args.imageId, encryptionKey);
} else {
  const encryptionKey = args.encryptionKey
    ? decodeKey(args.encryptionKey)
    : randomBytes(32);
  const snapshot = await takeSnapshot(encryptionKey);
  await restoreImage(snapshot, encryptionKey);
}

function parseArgs(argv: string[]): {
  imageId?: string;
  encryptionKey?: string;
} {
  const parsed: {
    imageId?: string;
    encryptionKey?: string;
  } = {};

  for (const arg of argv) {
    const [name, value = ""] = arg.split("=", 2);
    switch (name) {
      case "--image-id":
        parsed.imageId = value;
        break;
      case "--encryption-key":
        parsed.encryptionKey = value;
        break;
      default:
        throw new Error(`Unsupported argument: ${name}`);
    }
  }

  return {
    imageId: parsed.imageId,
    encryptionKey: parsed.encryptionKey,
  };
}

function decodeKey(encryptionKey: string | undefined): Uint8Array {
  if (!encryptionKey) throw new Error("Set --encryption-key");
  return Buffer.from(encryptionKey, "base64");
}

async function takeSnapshot(encryptionKey: Uint8Array): Promise<Image> {
  const image = modal.images.fromRegistry("alpine:3.21");
  const sb = await modal.sandboxes.create(app, image);
  try {
    console.log("Started Sandbox:", sb.sandboxId);
    await (
      await sb.exec([
        "sh",
        "-c",
        "mkdir -p /project && echo 'private data' > /project/state.txt",
      ])
    ).wait();

    const snapshot = await sb.snapshotDirectory("/project", {
      experimentalEncryptionKey: encryptionKey,
    });

    console.log("Snapshot Image ID:", snapshot.imageId);
    console.log(
      "Encryption key (base64):",
      Buffer.from(encryptionKey).toString("base64"),
    );
    return snapshot;
  } finally {
    await sb.terminate();
  }
}

async function restoreSnapshot(
  imageId: string,
  encryptionKey: Uint8Array,
): Promise<void> {
  const snapshot = await modal.images.fromId(imageId);
  await restoreImage(snapshot, encryptionKey);
}

async function restoreImage(
  snapshot: Image,
  encryptionKey: Uint8Array,
): Promise<void> {
  const image = modal.images.fromRegistry("alpine:3.21");
  const sb = await modal.sandboxes.create(app, image);
  try {
    console.log("Started Sandbox:", sb.sandboxId);
    await (await sb.exec(["mkdir", "-p", "/project"])).wait();
    await sb.mountImage("/project", snapshot, {
      experimentalEncryptionKey: encryptionKey,
    });

    const process = await sb.exec(["cat", "/project/state.txt"]);
    const output = await process.stdout.readText();
    console.log("Restored file contents:", output.trim());
  } finally {
    await sb.terminate();
  }
}
