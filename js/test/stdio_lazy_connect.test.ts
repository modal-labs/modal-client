import { expect, test } from "vitest";
import { FileDescriptor } from "../proto/modal_proto/api";
import { createMockModalClients } from "../test-support/grpc_mock";
import {
  DEFAULT_OUTPUT as OUTPUT,
  SANDBOX_ID,
  settle,
  startFakeWorker,
} from "../test-support/fake_worker";
import { TaskExecStdioFileDescriptor } from "../proto/modal_proto/task_command_router";

/**
 * Output is fetched when a caller reads it, and not before. An exec whose output
 * nobody wants should put nothing on the wire for it.
 *
 * These enter where callers do - `sandbox.exec()` - against a local worker over
 * a real socket with the control plane mocked, because whether a request is
 * issued depends on everything between the caller and the wire, which is what a
 * test starting further down cannot see.
 */

const V1_SANDBOX_ID = "sb-nGEijt9WbBMlGrsPH9FOaC";

test("exec() alone reads no output", async () => {
  const w = await startFakeWorker();
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"]);
    expect(process).toBeDefined();

    await settle();
    expect(w.reads).toEqual([]);
  } finally {
    await w.shutdown();
  }
});

test("reaching for stdout without reading reads no output", async () => {
  const w = await startFakeWorker();
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"]);

    const stream = process.stdout;
    expect(stream).toBeDefined();

    await settle();
    expect(w.reads).toEqual([]);
  } finally {
    await w.shutdown();
  }
});

test("reading stdout reads that descriptor only, and returns the output", async () => {
  const w = await startFakeWorker();
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"]);

    expect(await process.stdout.readText()).toBe(OUTPUT);
    // Reading one descriptor must not fetch the other.
    expect(w.reads).toEqual([
      TaskExecStdioFileDescriptor.TASK_EXEC_STDIO_FILE_DESCRIPTOR_STDOUT,
    ]);
  } finally {
    await w.shutdown();
  }
});

/**
 * A Sandbox's own output comes from the control plane rather than a worker, so
 * it gets the same treatment against a streaming mock of `SandboxGetLogs`.
 */
function serveLogs(
  mockCpClient: any,
  batches: string[][],
): { reads: FileDescriptor[] } {
  const reads: FileDescriptor[] = [];
  mockCpClient.sandboxGetLogs = async function* (request: {
    fileDescriptor: FileDescriptor;
  }) {
    reads.push(request.fileDescriptor);
    for (const [index, items] of batches.entries()) {
      yield {
        entryId: String(index + 1),
        items: items.map((data) => ({ data })),
        // The reader keeps asking until a batch says it is done.
        eof: index === batches.length - 1,
      };
    }
  };
  return { reads };
}

test("attaching to a Sandbox reads no logs", async () => {
  const { mockClient, mockCpClient } = createMockModalClients();
  const { reads } = serveLogs(mockCpClient, [["line-0\n"]]);

  const sandbox = await mockClient.sandboxes.fromId(V1_SANDBOX_ID);
  expect(sandbox).toBeDefined();

  await settle();
  expect(reads).toEqual([]);
});

test("reaching for Sandbox.stdout without reading reads no logs", async () => {
  const { mockClient, mockCpClient } = createMockModalClients();
  const { reads } = serveLogs(mockCpClient, [["line-0\n"]]);

  const sandbox = await mockClient.sandboxes.fromId(V1_SANDBOX_ID);
  const stream = sandbox.stdout;
  expect(stream).toBeDefined();

  await settle();
  expect(reads).toEqual([]);
});

test("reading Sandbox.stdout reads that descriptor only, and returns the output", async () => {
  const { mockClient, mockCpClient } = createMockModalClients();
  const { reads } = serveLogs(mockCpClient, [
    ["line-0\n", "line-1\n"],
    ["line-2\n"],
  ]);

  const sandbox = await mockClient.sandboxes.fromId(V1_SANDBOX_ID);
  expect(await sandbox.stdout.readText()).toBe(OUTPUT);
  // Reading one descriptor must not fetch the other.
  expect(reads).toEqual([FileDescriptor.FILE_DESCRIPTOR_STDOUT]);
});

test("reading Sandbox.stderr reads stderr", async () => {
  const { mockClient, mockCpClient } = createMockModalClients();
  const { reads } = serveLogs(mockCpClient, [["oh dear\n"]]);

  const sandbox = await mockClient.sandboxes.fromId(V1_SANDBOX_ID);
  expect(await sandbox.stderr.readText()).toBe("oh dear\n");
  expect(reads).toEqual([FileDescriptor.FILE_DESCRIPTOR_STDERR]);
});
