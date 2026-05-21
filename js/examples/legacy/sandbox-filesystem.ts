import { ModalClient } from "modal";

/**
 * Example demonstrating filesystem operations in a Modal Sandbox.
 *
 * This example shows how to:
 * - Open files for reading and writing
 * - Read file contents as binary data
 * - Write data to files
 * - Close file handles
 */

const modal = new ModalClient();

const app = await modal.apps.fromName("libmodal-example", {
  createIfMissing: true,
});
const image = modal.images.fromRegistry("alpine:3.21");

const sb = await modal.sandboxes.create(app, image);
console.log("Started Sandbox:", sb.sandboxId);

try {
  const writeHandle = await sb.open("/tmp/example.txt", "w");
  const encoder = new TextEncoder();
  const deocder = new TextDecoder();

  await writeHandle.write(encoder.encode("Hello, Modal filesystem!\n"));
  await writeHandle.write(encoder.encode("This is line 2.\n"));
  await writeHandle.write(encoder.encode("And this is line 3.\n"));
  await writeHandle.close();

  const readHandle = await sb.open("/tmp/example.txt", "r");
  const content = await readHandle.read();
  console.log("File content:", deocder.decode(content));
  await readHandle.close();

  const appendHandle = await sb.open("/tmp/example.txt", "a");
  await appendHandle.write(encoder.encode("This line was appended.\n"));
  await appendHandle.close();

  const seekHandle = await sb.open("/tmp/example.txt", "r");
  const appendedContent = await seekHandle.read();
  console.log("File with appended:", deocder.decode(appendedContent));
  await seekHandle.close();

  const binaryHandle = await sb.open("/tmp/data.bin", "w");
  const binaryData = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  await binaryHandle.write(binaryData);
  await binaryHandle.close();

  const readBinaryHandle = await sb.open("/tmp/data.bin", "r");
  const readData = await readBinaryHandle.read();
  console.log("Binary data:", readData);
  await readBinaryHandle.close();
} catch (error) {
  console.error("Filesystem operation failed:", error);
} finally {
  await sb.terminate();
}
