import { ModalClient } from "modal";

/**
 * Example demonstrating the sandbox.filesystem namespace API.
 *
 * This example shows how to:
 * - Write and read text files
 * - Write and read binary files
 * - Inspect file metadata
 * - Create directories and list their contents
 * - Upload and download files
 * - Delete files and directories
 */

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});
const image = modal.images.fromRegistry("alpine:3.21");

const sb = await modal.sandboxes.create(app, image);
console.log("Started Sandbox:", sb.sandboxId);

try {
  const fs = sb.filesystem;

  // ── write & read text ────────────────────────────────────────────────────

  await fs.writeText("Hello from sandbox.filesystem!\n", "/tmp/hello.txt");

  const text = await fs.readText("/tmp/hello.txt");
  console.log("readText:", text);

  // ── write & read bytes ───────────────────────────────────────────────────

  const payload = new Uint8Array([0x48, 0x65, 0x6c, 0x6c, 0x6f]); // "Hello"
  await fs.writeBytes(payload, "/tmp/hello.bin");

  const bytes = await fs.readBytes("/tmp/hello.bin");
  console.log("readBytes:", bytes);

  // ── stat ─────────────────────────────────────────────────────────────────

  const info = await fs.stat("/tmp/hello.txt");
  console.log(
    `stat: name=${info.name}, size=${info.size}, permissions=${info.permissions}`,
  );

  // ── make_directory & list_files ──────────────────────────────────────────

  await fs.makeDirectory("/tmp/mydir/nested");
  await fs.writeText("nested file\n", "/tmp/mydir/nested/file.txt");
  await fs.writeText("top-level file\n", "/tmp/mydir/top.txt");

  const entries = await fs.listFiles("/tmp/mydir");
  console.log("listFiles /tmp/mydir:");
  for (const entry of entries) {
    console.log(`  ${entry.name}  (${entry.type})`);
  }

  // ── copy_from_local & copy_to_local ──────────────────────────────────────

  // Write a local temp file, upload it, then download it back.
  const { writeFile, readFile, mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");

  const tmpDir = await mkdtemp(join(tmpdir(), "modal-fs-example-"));
  try {
    const localSrc = join(tmpDir, "upload.txt");
    await writeFile(localSrc, "Uploaded via copyFromLocal\n", "utf-8");

    await fs.copyFromLocal(localSrc, "/tmp/uploaded.txt");

    const localDst = join(tmpDir, "download.txt");
    await fs.copyToLocal("/tmp/uploaded.txt", localDst);

    const downloaded = await readFile(localDst, "utf-8");
    console.log("copy round-trip:", downloaded);
  } finally {
    await rm(tmpDir, { recursive: true, force: true });
  }

  // ── remove ───────────────────────────────────────────────────────────────

  await fs.remove("/tmp/hello.bin");
  await fs.remove("/tmp/mydir", { recursive: true });
  console.log("remove: cleaned up");
} catch (error) {
  console.error("Filesystem operation failed:", error);
} finally {
  await sb.terminate();
}
