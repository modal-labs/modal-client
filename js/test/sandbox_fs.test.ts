// Copyright Modal Labs 2026
import { expect, test } from "vitest";

import { InvalidError } from "../src/errors";
import { SandboxFilesystem } from "../src/sandbox_fs";

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
