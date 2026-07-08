// Copyright Modal Labs 2026
import { expect, test } from "vitest";

import {
  InvalidError,
  NotFoundError,
  SandboxFilesystemError,
} from "../src/errors";
import {
  SandboxFilesystem,
  expandWatchFilter,
  type FileWatchEvent,
} from "../src/sandbox_fs";

// ---------------------------------------------------------------------------
// Unit tests that throw before exec
// ---------------------------------------------------------------------------

const fs = new SandboxFilesystem(async () => {
  throw new Error("Unexpected exec call");
});

test("SandboxFsCopyFromLocalErrorsOnRelativeRemotePath", async () => {
  await expect(
    fs.copyFromLocal("/any-local.bin", "relative/path.bin"),
  ).rejects.toThrow(InvalidError);
});

test("SandboxFsCopyToLocalErrorsOnRelativeRemotePath", async () => {
  await expect(
    fs.copyToLocal("relative/path.bin", "/any-local.bin"),
  ).rejects.toThrow(InvalidError);
});

test("SandboxFsListFilesErrorsOnRelativeRemotePath", async () => {
  await expect(fs.listFiles("relative/path")).rejects.toThrow(InvalidError);
});

test("SandboxFsMakeDirectoryNoParentsErrorsOnRelativeRemotePath", async () => {
  await expect(
    fs.makeDirectory("relative/path", { createParents: false }),
  ).rejects.toThrow(InvalidError);
});

test("SandboxFsReadBytesErrorsOnRelativeRemotePath", async () => {
  await expect(fs.readBytes("relative/path.bin")).rejects.toThrow(InvalidError);
});

test("SandboxFsReadTextErrorsOnRelativeRemotePath", async () => {
  await expect(fs.readText("relative/path.txt")).rejects.toThrow(InvalidError);
});

test("SandboxFsRemoveErrorsOnRelativeRemotePath", async () => {
  await expect(fs.remove("relative/path.txt")).rejects.toThrow(InvalidError);
});

test("SandboxFsStatErrorsOnRelativeRemotePath", async () => {
  await expect(fs.stat("relative/path.txt")).rejects.toThrow(InvalidError);
});

test("SandboxFsWatchErrorsOnRelativeRemotePath", async () => {
  const watch = fs.watch("relative/path");
  const iter = watch[Symbol.asyncIterator]();
  await expect(iter.next()).rejects.toThrow(InvalidError);
});

test("expandWatchFilter expands Modify to include Rust rename variants", () => {
  expect(expandWatchFilter(["Modify"])).toEqual(
    expect.arrayContaining(["Modify", "Rename", "RenameFrom", "RenameTo"]),
  );
});

test("expandWatchFilter does not expand non-Modify types", () => {
  expect(expandWatchFilter(["Create", "Remove"])).toEqual(["Create", "Remove"]);
});

test("expandWatchFilter handles empty filter", () => {
  expect(expandWatchFilter([])).toEqual([]);
});

// ---------------------------------------------------------------------------
// watch lifecycle (mocked exec)
// ---------------------------------------------------------------------------

type WatchExecFn = ConstructorParameters<typeof SandboxFilesystem>[0];
type FakeProcess = Awaited<ReturnType<WatchExecFn>>;

/** Minimal ContainerProcess stand-in exercising watch()'s stream lifecycle. */
function fakeWatchProcess(opts: {
  lines: string[];
  returnCode: number;
  stderr?: Uint8Array;
  streamError?: unknown;
}): FakeProcess {
  return {
    stdout: (async function* () {
      for (const line of opts.lines) yield line;
      if (opts.streamError) throw opts.streamError;
    })(),
    stderr: { readBytes: async () => opts.stderr ?? new Uint8Array(0) },
    closeStdin: async () => {},
    wait: async () => opts.returnCode,
  } as unknown as FakeProcess;
}

test("SandboxFsWatchDoesNotThrowOnEarlyBreakWhenProcessExitsNonZero", async () => {
  const watchFs = new SandboxFilesystem(async () =>
    fakeWatchProcess({
      lines: ['{"event_type":"Create","paths":["/tmp/x"]}\n'],
      returnCode: 137, // e.g. SIGKILL surfaces as 128 + 9
    }),
  );

  // Breaking after the first event must not surface the non-zero exit.
  const events: FileWatchEvent[] = [];
  for await (const event of watchFs.watch("/tmp/w")) {
    events.push(event);
    break;
  }
  expect(events).toHaveLength(1);
  expect(events[0].eventType).toBe("Create");
});

test("SandboxFsWatchThrowsWhenStreamEndsWithNonZeroExit", async () => {
  const watchFs = new SandboxFilesystem(async () =>
    fakeWatchProcess({ lines: [], returnCode: 1 }),
  );

  const iter = watchFs.watch("/tmp/w")[Symbol.asyncIterator]();
  await expect(iter.next()).rejects.toThrow(SandboxFilesystemError);
});

test("SandboxFsWatchTranslatesMidStreamSandboxUnavailable", async () => {
  const watchFs = new SandboxFilesystem(async () =>
    fakeWatchProcess({
      lines: ['{"event_type":"Create","paths":["/tmp/x"]}\n'],
      returnCode: 0,
      // Sandbox terminated mid-watch surfaces as a raw transport error.
      streamError: new NotFoundError("sandbox gone"),
    }),
  );

  const iter = watchFs.watch("/tmp/w")[Symbol.asyncIterator]();
  // First event is delivered, then the translated error surfaces.
  expect((await iter.next()).value).toEqual({
    eventType: "Create",
    paths: ["/tmp/x"],
  });
  await expect(iter.next()).rejects.toThrow(/Sandbox is unavailable/);
});

test("SandboxFsWatchSkipsMalformedAndUnknownEvents", async () => {
  const watchFs = new SandboxFilesystem(async () =>
    fakeWatchProcess({
      lines: [
        "not json\n",
        '{"paths":["/tmp/x"]}\n', // missing event_type
        '{"event_type":"Bogus","paths":["/tmp/x"]}\n', // unknown type
        '{"event_type":"Create","paths":[]}\n', // empty paths
        '{"event_type":"Create","paths":["/tmp/x"]}\n', // valid
      ],
      returnCode: 0,
    }),
  );

  const events: FileWatchEvent[] = [];
  for await (const event of watchFs.watch("/tmp/w")) {
    events.push(event);
  }
  expect(events).toEqual([{ eventType: "Create", paths: ["/tmp/x"] }]);
});

test("SandboxFsWriteBytesErrorsOnRelativeRemotePath", async () => {
  await expect(
    fs.writeBytes(new Uint8Array([1]), "relative/path.bin"),
  ).rejects.toThrow(InvalidError);
});

test("SandboxFsWriteTextErrorsOnRelativeRemotePath", async () => {
  await expect(fs.writeText("data", "relative/path.txt")).rejects.toThrow(
    InvalidError,
  );
});

test("SandboxFsWriteBytesErrorsOnUnsupportedDataType", async () => {
  await expect(
    // @ts-expect-error intentional wrong type
    fs.writeBytes("not-bytes", "/tmp/unused.bin"),
  ).rejects.toThrow(TypeError);
});

test("SandboxFsWriteTextErrorsOnNonStringData", async () => {
  await expect(
    // @ts-expect-error intentional wrong type
    fs.writeText(new Uint8Array([1]), "/tmp/unused.txt"),
  ).rejects.toThrow(TypeError);
});
