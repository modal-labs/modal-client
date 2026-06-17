// Copyright Modal Labs 2026
/**
 * End-to-end tests for SandboxFilesystem against a live Modal sandbox.
 */

import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { expect, onTestFinished, test } from "vitest";

import type { Sandbox } from "../src/sandbox";
import {
  SandboxFilesystemDirectoryNotEmptyError,
  SandboxFilesystemFileTooLargeError,
  SandboxFilesystemIsADirectoryError,
  SandboxFilesystemNotADirectoryError,
  SandboxFilesystemNotFoundError,
  SandboxFilesystemPathAlreadyExistsError,
} from "../src/errors";
import {
  createSparseFile,
  isDirRemote,
  mkdirRemote,
  pathExists,
  randomBytes,
  readRemoteFile,
  statRemoteFile,
  symlinkRemote,
  tmpPath,
  writeRemoteFile,
} from "../test-support/sandbox-exec-helpers";
import { tc } from "../test-support/test-client";

const WRITE_CHUNK_SIZE = 4 * 1024 * 1024;

async function newTestSandbox(): Promise<Sandbox> {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(() => sb.terminate());
  return sb;
}

// ---------------------------------------------------------------------------
// round-trips
// ---------------------------------------------------------------------------

test("SandboxFsE2eWriteBytesReadBytesRoundTrip", async () => {
  const sb = await newTestSandbox();
  const payload = randomBytes(4096, 50);

  await sb.filesystem.writeBytes(payload, "/tmp/e2e-rt-bytes.bin");
  const result = await sb.filesystem.readBytes("/tmp/e2e-rt-bytes.bin");

  expect(Buffer.from(result).equals(Buffer.from(payload))).toBe(true);
});

test("SandboxFsE2eWriteTextReadTextRoundTrip", async () => {
  const sb = await newTestSandbox();
  const text = "round-trip text\nwith unicode: ☃🎉\n";

  await sb.filesystem.writeText(text, "/tmp/e2e-rt-text.txt");
  const result = await sb.filesystem.readText("/tmp/e2e-rt-text.txt");

  expect(result).toBe(text);
});

test("SandboxFsE2eCopyFromLocalThenCopyToLocalRoundTrip", async () => {
  const sb = await newTestSandbox();
  const t = await tmpPath();
  const payload = randomBytes(8192, 60);
  await writeFile(join(t, "upload.bin"), payload);

  await sb.filesystem.copyFromLocal(
    join(t, "upload.bin"),
    "/tmp/e2e-copy-round-trip.bin",
  );
  await sb.filesystem.copyToLocal(
    "/tmp/e2e-copy-round-trip.bin",
    join(t, "download.bin"),
  );

  const downloaded = new Uint8Array(await readFile(join(t, "download.bin")));
  expect(Buffer.from(downloaded).equals(Buffer.from(payload))).toBe(true);
});

// ---------------------------------------------------------------------------
// copy_from_local
// ---------------------------------------------------------------------------

test("SandboxFsE2eCopyFromLocalCopiesTextFile", async () => {
  const sb = await newTestSandbox();
  const t = await tmpPath();
  await writeFile(join(t, "text.txt"), "text content", "utf-8");

  await sb.filesystem.copyFromLocal(
    join(t, "text.txt"),
    "/tmp/e2e-cfl-text.txt",
  );

  expect(
    new TextDecoder().decode(await readRemoteFile(sb, "/tmp/e2e-cfl-text.txt")),
  ).toBe("text content");
});

test("SandboxFsE2eCopyFromLocalCopiesEmptyFile", async () => {
  const sb = await newTestSandbox();
  const t = await tmpPath();
  await writeFile(join(t, "empty.bin"), new Uint8Array(0));

  await sb.filesystem.copyFromLocal(
    join(t, "empty.bin"),
    "/tmp/e2e-cfl-empty.bin",
  );

  expect(await readRemoteFile(sb, "/tmp/e2e-cfl-empty.bin")).toEqual(
    new Uint8Array(0),
  );
});

test("SandboxFsE2eCopyFromLocalHandlesLargeFile", async () => {
  const sb = await newTestSandbox();
  const t = await tmpPath();
  const payload = randomBytes(2 * 1024 * 1024, 70);
  await writeFile(join(t, "large.bin"), payload);

  await sb.filesystem.copyFromLocal(
    join(t, "large.bin"),
    "/tmp/e2e-cfl-large.bin",
  );

  const result = await readRemoteFile(sb, "/tmp/e2e-cfl-large.bin");
  expect(result.length).toBe(payload.length);
  expect(Buffer.from(result).equals(Buffer.from(payload))).toBe(true);
}, 120_000);

test("SandboxFsE2eCopyFromLocalErrorsWhenLocalPathMissing", async () => {
  const sb = await newTestSandbox();
  const t = await tmpPath();

  await expect(
    sb.filesystem.copyFromLocal(
      join(t, "missing.bin"),
      "/tmp/e2e-cfl-unused.bin",
    ),
  ).rejects.toThrow(/ENOENT/);
});

test("SandboxFsE2eCopyFromLocalErrorsWhenLocalPathIsDirectory", async () => {
  const sb = await newTestSandbox();
  const t = await tmpPath();
  await mkdir(join(t, "src-dir"));

  await expect(
    sb.filesystem.copyFromLocal(join(t, "src-dir"), "/tmp/e2e-cfl-unused.bin"),
  ).rejects.toThrow();
});

// ---------------------------------------------------------------------------
// copy_to_local
// ---------------------------------------------------------------------------

test("SandboxFsE2eCopyToLocalCreatesParentDirectoriesIfNeeded", async () => {
  const sb = await newTestSandbox();
  const t = await tmpPath();
  const payload = randomBytes(2048, 2);
  await writeRemoteFile(sb, "/tmp/e2e-ctl-parent.bin", payload);

  const localPath = join(t, "deep", "nested", "path", "copied.bin");
  await sb.filesystem.copyToLocal("/tmp/e2e-ctl-parent.bin", localPath);

  expect(new Uint8Array(await readFile(localPath))).toEqual(payload);
});

test("SandboxFsE2eCopyToLocalCopiesEmptyFile", async () => {
  const sb = await newTestSandbox();
  const t = await tmpPath();
  await writeRemoteFile(sb, "/tmp/e2e-ctl-empty.bin", new Uint8Array(0));

  await sb.filesystem.copyToLocal(
    "/tmp/e2e-ctl-empty.bin",
    join(t, "empty.bin"),
  );

  expect(new Uint8Array(await readFile(join(t, "empty.bin")))).toEqual(
    new Uint8Array(0),
  );
});

test("SandboxFsE2eCopyToLocalOverwritesExistingLocalFile", async () => {
  const sb = await newTestSandbox();
  const t = await tmpPath();
  const payload = randomBytes(4096, 5);
  await writeRemoteFile(sb, "/tmp/e2e-ctl-overwrite.bin", payload);
  await writeFile(join(t, "overwrite.bin"), "old-data");

  await sb.filesystem.copyToLocal(
    "/tmp/e2e-ctl-overwrite.bin",
    join(t, "overwrite.bin"),
  );

  expect(new Uint8Array(await readFile(join(t, "overwrite.bin")))).toEqual(
    payload,
  );
});

test("SandboxFsE2eCopyToLocalPreservesExistingFileOnRemoteError", async () => {
  const sb = await newTestSandbox();
  const t = await tmpPath();
  const existing = join(t, "existing.bin");
  await writeFile(existing, "stable-content");

  await expect(
    sb.filesystem.copyToLocal("/tmp/e2e-copy-to-local-missing.bin", existing),
  ).rejects.toThrow(SandboxFilesystemNotFoundError);

  expect((await readFile(existing)).toString()).toBe("stable-content");
});

test("SandboxFsE2eCopyToLocalErrorsWhenLocalPathIsDirectory", async () => {
  const sb = await newTestSandbox();
  const t = await tmpPath();
  await writeRemoteFile(sb, "/tmp/e2e-ctl-dir-source.bin", randomBytes(512, 6));
  const localDir = join(t, "local-dir");
  await mkdir(localDir, { recursive: true });

  await expect(
    sb.filesystem.copyToLocal("/tmp/e2e-ctl-dir-source.bin", localDir),
  ).rejects.toThrow();

  // Verify no temp file was leaked.
  const entries = await readdir(t);
  expect(entries.filter((e) => e.startsWith(".modal-sandbox-fs-tmp-"))).toEqual(
    [],
  );
});

test("SandboxFsE2eCopyToLocalErrorsWhenRemotePathIsDirectory", async () => {
  const sb = await newTestSandbox();
  const t = await tmpPath();
  await mkdirRemote(sb, "/tmp/e2e-ctl-remote-dir");

  await expect(
    sb.filesystem.copyToLocal("/tmp/e2e-ctl-remote-dir", join(t, "unused.bin")),
  ).rejects.toThrow(SandboxFilesystemIsADirectoryError);
});

test("SandboxFsE2eCopyToLocalErrorsWhenFileTooLarge", async () => {
  const sb = await newTestSandbox();
  const t = await tmpPath();
  const localPath = join(t, "too-large-out.bin");
  await createSparseFile(
    sb,
    "/tmp/e2e-copy-too-large.bin",
    6 * 1024 * 1024 * 1024,
  );

  await expect(
    sb.filesystem.copyToLocal("/tmp/e2e-copy-too-large.bin", localPath),
  ).rejects.toThrow(SandboxFilesystemFileTooLargeError);

  await expect(readFile(localPath)).rejects.toThrow(/ENOENT/);
});

// ---------------------------------------------------------------------------
// list_files
// ---------------------------------------------------------------------------

test("SandboxFsE2eListFilesReturnsFilesAndDirectories", async () => {
  const sb = await newTestSandbox();
  const base = "/tmp/e2e-list-files-basic";
  const fileContent = new TextEncoder().encode("hello list_files");
  await mkdirRemote(sb, base);
  await writeRemoteFile(sb, `${base}/file.txt`, fileContent);
  await mkdirRemote(sb, `${base}/subdir`);

  const entries = await sb.filesystem.listFiles(base);
  const names = new Set(entries.map((e) => e.name));

  expect(names).toContain("file.txt");
  expect(names).toContain("subdir");
  expect(entries.find((e) => e.name === "file.txt")!.type).toBe("file");
  expect(entries.find((e) => e.name === "file.txt")!.size).toBe(
    fileContent.length,
  );
  expect(entries.find((e) => e.name === "subdir")!.type).toBe("directory");
});

test("SandboxFsE2eListFilesIsNotRecursive", async () => {
  const sb = await newTestSandbox();
  const base = "/tmp/e2e-ls-nonrecursive";
  await mkdirRemote(sb, base);
  await writeRemoteFile(sb, `${base}/top.txt`, new TextEncoder().encode("top"));
  await mkdirRemote(sb, `${base}/child`);
  await writeRemoteFile(
    sb,
    `${base}/child/nested.txt`,
    new TextEncoder().encode("nested"),
  );
  await mkdirRemote(sb, `${base}/child/grandchild`);

  const entries = await sb.filesystem.listFiles(base);

  expect(new Set(entries.map((e) => e.name))).toEqual(
    new Set(["top.txt", "child"]),
  );
});

test("SandboxFsE2eListFilesFileInfoHasCorrectMetadata", async () => {
  const sb = await newTestSandbox();
  const base = "/tmp/e2e-list-files-fields";
  const fileContent = new TextEncoder().encode("field check");
  await mkdirRemote(sb, base);
  await writeRemoteFile(sb, `${base}/check.txt`, fileContent);
  const expected = await statRemoteFile(sb, `${base}/check.txt`);

  const entries = await sb.filesystem.listFiles(base);
  const entry = entries.find((e) => e.name === "check.txt")!;

  expect(entry.type).toBe("file");
  expect(entry.size).toBe(fileContent.length);
  expect(entry.permissions).toBe(expected.permissions);
  expect(entry.mode).toBe(expected.mode);
  expect(entry.owner).toBe(expected.owner);
  expect(entry.group).toBe(expected.group);
  expect(Math.abs(entry.modifiedTime - expected.mtime)).toBeLessThanOrEqual(10);
  expect(entry.symlinkTarget).toBeNull();
});

test("SandboxFsE2eListFilesReturnsEmptyListForEmptyDir", async () => {
  const sb = await newTestSandbox();
  await mkdirRemote(sb, "/tmp/e2e-ls-empty");

  expect(await sb.filesystem.listFiles("/tmp/e2e-ls-empty")).toEqual([]);
});

test("SandboxFsE2eListFilesEntriesSortedByName", async () => {
  const sb = await newTestSandbox();
  const base = "/tmp/e2e-ls-sorted";
  await mkdirRemote(sb, base);
  await writeRemoteFile(sb, `${base}/zebra.txt`, new TextEncoder().encode("z"));
  await writeRemoteFile(sb, `${base}/alpha.txt`, new TextEncoder().encode("a"));
  await writeRemoteFile(
    sb,
    `${base}/middle.txt`,
    new TextEncoder().encode("m"),
  );

  const result = await sb.filesystem.listFiles(base);

  expect(result.map((e) => e.name)).toEqual([
    "alpha.txt",
    "middle.txt",
    "zebra.txt",
  ]);
});

test("SandboxFsE2eListFilesErrorsWhenPathDoesNotExist", async () => {
  const sb = await newTestSandbox();
  await expect(
    sb.filesystem.listFiles("/tmp/e2e-ls-nonexistent"),
  ).rejects.toThrow(SandboxFilesystemNotFoundError);
});

test("SandboxFsE2eListFilesErrorsWhenPathIsAFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(
    sb,
    "/tmp/e2e-ls-isfile.txt",
    new TextEncoder().encode("not a dir"),
  );

  await expect(
    sb.filesystem.listFiles("/tmp/e2e-ls-isfile.txt"),
  ).rejects.toThrow(SandboxFilesystemNotADirectoryError);
});

test("SandboxFsE2eListFilesErrorsWhenPathComponentIsAFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(
    sb,
    "/tmp/e2e-ls-blocker.txt",
    new TextEncoder().encode("file"),
  );

  await expect(
    sb.filesystem.listFiles("/tmp/e2e-ls-blocker.txt/subdir"),
  ).rejects.toThrow(SandboxFilesystemNotADirectoryError);
});

test("SandboxFsE2eListFilesSymlinkReportedAsSymlinkWithTarget", async () => {
  const sb = await newTestSandbox();
  const base = "/tmp/e2e-ls-symlink";
  await mkdirRemote(sb, base);
  await writeRemoteFile(
    sb,
    `${base}/target.txt`,
    new TextEncoder().encode("hi"),
  );
  await symlinkRemote(sb, `${base}/target.txt`, `${base}/link.txt`);

  const entries = await sb.filesystem.listFiles(base);
  const link = entries.find((e) => e.name === "link.txt")!;

  expect(link.type).toBe("symlink");
  expect(link.symlinkTarget).toBe(`${base}/target.txt`);
});

test("SandboxFsE2eListFilesDoesNotShowSymlinkTargetForNonsymlinkedFile", async () => {
  const sb = await newTestSandbox();
  const base = "/tmp/e2e-ls-no-symlink-file";
  await mkdirRemote(sb, base);
  await writeRemoteFile(
    sb,
    `${base}/file.txt`,
    new TextEncoder().encode("hello"),
  );

  const entries = await sb.filesystem.listFiles(base);

  expect(entries).toHaveLength(1);
  expect(entries[0].symlinkTarget).toBeNull();
});

test("SandboxFsE2eListFilesDoesNotShowSymlinkTargetForNonsymlinkedDirectory", async () => {
  const sb = await newTestSandbox();
  const base = "/tmp/e2e-ls-no-symlink-dir";
  await mkdirRemote(sb, base);
  await mkdirRemote(sb, `${base}/subdir`);

  const entries = await sb.filesystem.listFiles(base);

  expect(entries).toHaveLength(1);
  expect(entries[0].symlinkTarget).toBeNull();
});

test("SandboxFsE2eListFilesDanglingSymlinkReportedAsSymlink", async () => {
  const sb = await newTestSandbox();
  const base = "/tmp/e2e-ls-dangling";
  await mkdirRemote(sb, base);
  await symlinkRemote(
    sb,
    "/tmp/e2e-ls-dangling/nonexistent",
    `${base}/dangling`,
  );

  const entries = await sb.filesystem.listFiles(base);

  expect(entries).toHaveLength(1);
  expect(entries[0].type).toBe("symlink");
  expect(entries[0].symlinkTarget).toBe("/tmp/e2e-ls-dangling/nonexistent");
});

test("SandboxFsE2eListFilesFollowsSymlinkIfPathIsDirectory", async () => {
  const sb = await newTestSandbox();
  const target = "/tmp/e2e-ls-follow-target";
  const link = "/tmp/e2e-ls-follow-link";
  await mkdirRemote(sb, target);
  await writeRemoteFile(
    sb,
    `${target}/file.txt`,
    new TextEncoder().encode("hello"),
  );
  await symlinkRemote(sb, target, link);

  const entries = await sb.filesystem.listFiles(link);

  expect(entries).toHaveLength(1);
  expect(entries[0].name).toBe("file.txt");
  expect(entries[0].type).toBe("file");
  expect(entries[0].path).toContain("e2e-ls-follow-link");
});

test("SandboxFsE2eListFilesSymlinkToDirectoryReportedAsSymlink", async () => {
  const sb = await newTestSandbox();
  const base = "/tmp/e2e-ls-dirlink";
  const target = "/tmp/e2e-ls-dirlink-target";
  await mkdirRemote(sb, base);
  await mkdirRemote(sb, target);
  await symlinkRemote(sb, target, `${base}/link-to-dir`);

  const entries = await sb.filesystem.listFiles(base);
  const link = entries.find((e) => e.name === "link-to-dir")!;

  expect(link.type).toBe("symlink");
  expect(link.symlinkTarget).toBe(target);
});

// ---------------------------------------------------------------------------
// make_directory
// ---------------------------------------------------------------------------

test("SandboxFsE2eMakeDirectoryCreatesNestedDirectories", async () => {
  const sb = await newTestSandbox();
  await sb.filesystem.makeDirectory("/tmp/e2e-make-dir-a/b/c");

  expect(await isDirRemote(sb, "/tmp/e2e-make-dir-a/b/c")).toBe(true);
});

test("SandboxFsE2eMakeDirectoryNoParentsCreatesDirectory", async () => {
  const sb = await newTestSandbox();
  await mkdirRemote(sb, "/tmp/e2e-make-dir-parent");
  await sb.filesystem.makeDirectory("/tmp/e2e-make-dir-parent/new-subdir", {
    createParents: false,
  });

  expect(await isDirRemote(sb, "/tmp/e2e-make-dir-parent/new-subdir")).toBe(
    true,
  );
});

test("SandboxFsE2eMakeDirectoryIsIdempotentWhenAlreadyExists", async () => {
  const sb = await newTestSandbox();
  await mkdirRemote(sb, "/tmp/e2e-mkdir-idem");

  await sb.filesystem.makeDirectory("/tmp/e2e-mkdir-idem");

  expect(await isDirRemote(sb, "/tmp/e2e-mkdir-idem")).toBe(true);
});

test("SandboxFsE2eMakeDirectoryNoParentsErrorsWhenAlreadyExists", async () => {
  const sb = await newTestSandbox();
  await mkdirRemote(sb, "/tmp/e2e-make-dir-existing");

  await expect(
    sb.filesystem.makeDirectory("/tmp/e2e-make-dir-existing", {
      createParents: false,
    }),
  ).rejects.toThrow(SandboxFilesystemPathAlreadyExistsError);
});

test("SandboxFsE2eMakeDirectoryNoParentsErrorsWhenParentMissing", async () => {
  const sb = await newTestSandbox();
  await expect(
    sb.filesystem.makeDirectory("/tmp/e2e-mkdir-missing-parent/child", {
      createParents: false,
    }),
  ).rejects.toThrow(SandboxFilesystemNotFoundError);
});

test("SandboxFsE2eMakeDirectoryNoParentsErrorsWhenTargetIsAFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(
    sb,
    "/tmp/e2e-mkdir-target-file",
    new TextEncoder().encode("file"),
  );

  await expect(
    sb.filesystem.makeDirectory("/tmp/e2e-mkdir-target-file", {
      createParents: false,
    }),
  ).rejects.toThrow(SandboxFilesystemPathAlreadyExistsError);
});

test("SandboxFsE2eMakeDirectoryErrorsWhenTargetIsAFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(
    sb,
    "/tmp/e2e-mkdir-target-file-parents",
    new TextEncoder().encode("file"),
  );

  await expect(
    sb.filesystem.makeDirectory("/tmp/e2e-mkdir-target-file-parents"),
  ).rejects.toThrow(SandboxFilesystemPathAlreadyExistsError);
});

test("SandboxFsE2eMakeDirectoryErrorsWhenAncestorIsAFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(
    sb,
    "/tmp/e2e-mkdir-blocker",
    new TextEncoder().encode("file"),
  );

  await expect(
    sb.filesystem.makeDirectory("/tmp/e2e-mkdir-blocker/child"),
  ).rejects.toThrow(SandboxFilesystemNotADirectoryError);
});

// ---------------------------------------------------------------------------
// read_bytes
// ---------------------------------------------------------------------------

test("SandboxFsE2eReadBytesReturnsExpectedBytes", async () => {
  const sb = await newTestSandbox();
  const payload = new Uint8Array([0x00, 0x01, 0x02, 0x62, 0x69, 0x6e, 0xff]);
  await writeRemoteFile(sb, "/tmp/e2e-read-bytes.bin", payload);

  expect(await sb.filesystem.readBytes("/tmp/e2e-read-bytes.bin")).toEqual(
    payload,
  );
});

test("SandboxFsE2eReadBytesReturnsEmptyBytesForEmptyFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(sb, "/tmp/e2e-read-bytes-empty.bin", new Uint8Array(0));

  expect(
    await sb.filesystem.readBytes("/tmp/e2e-read-bytes-empty.bin"),
  ).toEqual(new Uint8Array(0));
});

test("SandboxFsE2eReadBytesErrorsWhenRemotePathMissing", async () => {
  const sb = await newTestSandbox();
  await expect(
    sb.filesystem.readBytes("/tmp/e2e-read-bytes-missing.bin"),
  ).rejects.toThrow(SandboxFilesystemNotFoundError);
});

test("SandboxFsE2eReadBytesErrorsWhenRemotePathIsDirectory", async () => {
  const sb = await newTestSandbox();
  await mkdirRemote(sb, "/tmp/e2e-read-bytes-dir");

  await expect(
    sb.filesystem.readBytes("/tmp/e2e-read-bytes-dir"),
  ).rejects.toThrow(SandboxFilesystemIsADirectoryError);
});

test("SandboxFsE2eReadBytesErrorsWhenFileTooLarge", async () => {
  const sb = await newTestSandbox();
  await createSparseFile(
    sb,
    "/tmp/e2e-read-bytes-large.bin",
    6 * 1024 * 1024 * 1024,
  );

  await expect(
    sb.filesystem.readBytes("/tmp/e2e-read-bytes-large.bin"),
  ).rejects.toThrow(SandboxFilesystemFileTooLargeError);
});

// ---------------------------------------------------------------------------
// read_text
// ---------------------------------------------------------------------------

test("SandboxFsE2eReadTextReturnsExpectedText", async () => {
  const sb = await newTestSandbox();
  const text = "hello from read_text\nsnowman: ☃\n";
  await writeRemoteFile(
    sb,
    "/tmp/e2e-read-text.txt",
    new TextEncoder().encode(text),
  );

  expect(await sb.filesystem.readText("/tmp/e2e-read-text.txt")).toBe(text);
});

test("SandboxFsE2eReadTextReturnsEmptyStringForEmptyFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(sb, "/tmp/e2e-read-text-empty.txt", new Uint8Array(0));

  expect(await sb.filesystem.readText("/tmp/e2e-read-text-empty.txt")).toBe("");
});

test("SandboxFsE2eReadTextErrorsWhenRemotePathMissing", async () => {
  const sb = await newTestSandbox();
  await expect(
    sb.filesystem.readText("/tmp/e2e-read-text-missing.txt"),
  ).rejects.toThrow(SandboxFilesystemNotFoundError);
});

test("SandboxFsE2eReadTextErrorsWhenRemotePathIsDirectory", async () => {
  const sb = await newTestSandbox();
  await mkdirRemote(sb, "/tmp/e2e-read-text-dir");

  await expect(
    sb.filesystem.readText("/tmp/e2e-read-text-dir"),
  ).rejects.toThrow(SandboxFilesystemIsADirectoryError);
});

test("SandboxFsE2eReadTextErrorsWhenFileTooLarge", async () => {
  const sb = await newTestSandbox();
  await createSparseFile(
    sb,
    "/tmp/e2e-read-text-large.txt",
    6 * 1024 * 1024 * 1024,
  );

  await expect(
    sb.filesystem.readText("/tmp/e2e-read-text-large.txt"),
  ).rejects.toThrow(SandboxFilesystemFileTooLargeError);
});

// ---------------------------------------------------------------------------
// remove
// ---------------------------------------------------------------------------

test("SandboxFsE2eRemoveRemovesAFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(
    sb,
    "/tmp/e2e-remove-file.bin",
    new TextEncoder().encode("data"),
  );

  await sb.filesystem.remove("/tmp/e2e-remove-file.bin");

  expect(await pathExists(sb, "/tmp/e2e-remove-file.bin")).toBe(false);
});

test("SandboxFsE2eRemoveRemovesEmptyDirectory", async () => {
  const sb = await newTestSandbox();
  await mkdirRemote(sb, "/tmp/e2e-rm-emptydir");

  await sb.filesystem.remove("/tmp/e2e-rm-emptydir");

  expect(await pathExists(sb, "/tmp/e2e-rm-emptydir")).toBe(false);
});

test("SandboxFsE2eRemoveRecursiveRemovesDirectoryTree", async () => {
  const sb = await newTestSandbox();
  await mkdirRemote(sb, "/tmp/e2e-remove-tree/a/b");
  await writeRemoteFile(
    sb,
    "/tmp/e2e-remove-tree/a/file.txt",
    new TextEncoder().encode("data"),
  );

  await sb.filesystem.remove("/tmp/e2e-remove-tree", { recursive: true });

  expect(await pathExists(sb, "/tmp/e2e-remove-tree")).toBe(false);
});

test("SandboxFsE2eRemoveErrorsWhenMissing", async () => {
  const sb = await newTestSandbox();
  await expect(sb.filesystem.remove("/tmp/e2e-rm-missing")).rejects.toThrow(
    SandboxFilesystemNotFoundError,
  );
});

test("SandboxFsE2eRemoveErrorsWhenTargetIsNonemptyDirectoryAndNotRecursive", async () => {
  const sb = await newTestSandbox();
  await mkdirRemote(sb, "/tmp/e2e-rm-nonempty");
  await writeRemoteFile(
    sb,
    "/tmp/e2e-rm-nonempty/file.txt",
    new TextEncoder().encode("hello"),
  );

  await expect(
    sb.filesystem.remove("/tmp/e2e-rm-nonempty", { recursive: false }),
  ).rejects.toThrow(SandboxFilesystemDirectoryNotEmptyError);
});

test("SandboxFsE2eRemoveRemovesSymlinkWithoutFollowingIt", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(
    sb,
    "/tmp/e2e-rm-symlink-target.txt",
    new TextEncoder().encode("original"),
  );
  await symlinkRemote(
    sb,
    "/tmp/e2e-rm-symlink-target.txt",
    "/tmp/e2e-rm-symlink-link.txt",
  );

  await sb.filesystem.remove("/tmp/e2e-rm-symlink-link.txt");

  expect(await pathExists(sb, "/tmp/e2e-rm-symlink-link.txt")).toBe(false);
  expect(await pathExists(sb, "/tmp/e2e-rm-symlink-target.txt")).toBe(true);
});

// ---------------------------------------------------------------------------
// stat
// ---------------------------------------------------------------------------

test("SandboxFsE2eStatReturnsMetadataForFile", async () => {
  const sb = await newTestSandbox();
  const fileContent = new TextEncoder().encode("hello stat");
  await writeRemoteFile(sb, "/tmp/e2e-stat-file.txt", fileContent);

  const info = await sb.filesystem.stat("/tmp/e2e-stat-file.txt");

  expect(info.name).toBe("e2e-stat-file.txt");
  expect(info.path).toBe("/tmp/e2e-stat-file.txt");
  expect(info.type).toBe("file");
  expect(info.size).toBe(fileContent.length);
  expect(info.permissions).toMatch(/^\d{4}$/);
  expect(info.mode).toBeGreaterThan(0);
  expect(info.modifiedTime).toBeGreaterThan(0);
  expect(info.symlinkTarget).toBeNull();
});

test("SandboxFsE2eStatReturnsMetadataForDirectory", async () => {
  const sb = await newTestSandbox();
  await mkdirRemote(sb, "/tmp/e2e-stat-dir");

  const info = await sb.filesystem.stat("/tmp/e2e-stat-dir");

  expect(info.name).toBe("e2e-stat-dir");
  expect(info.type).toBe("directory");
  expect(info.symlinkTarget).toBeNull();
});

test("SandboxFsE2eStatReturnsMetadataForEmptyFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(sb, "/tmp/e2e-stat-empty.txt", new Uint8Array(0));

  const info = await sb.filesystem.stat("/tmp/e2e-stat-empty.txt");

  expect(info.type).toBe("file");
  expect(info.size).toBe(0);
});

test("SandboxFsE2eStatExactFieldsMatchShellStat", async () => {
  const sb = await newTestSandbox();
  const content = new TextEncoder().encode("field check");
  await writeRemoteFile(sb, "/tmp/e2e-stat-fields.txt", content);
  const expected = await statRemoteFile(sb, "/tmp/e2e-stat-fields.txt");

  const info = await sb.filesystem.stat("/tmp/e2e-stat-fields.txt");

  expect(info.permissions).toBe(expected.permissions);
  expect(info.mode).toBe(expected.mode);
  expect(info.owner).toBe(expected.owner);
  expect(info.group).toBe(expected.group);
  expect(Math.abs(info.modifiedTime - expected.mtime)).toBeLessThanOrEqual(10);
  expect(info.symlinkTarget).toBeNull();
});

test("SandboxFsE2eStatSymlinkToFileReportedAsSymlink", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(
    sb,
    "/tmp/e2e-stat-lnk-target.txt",
    new TextEncoder().encode("hi"),
  );
  await symlinkRemote(
    sb,
    "/tmp/e2e-stat-lnk-target.txt",
    "/tmp/e2e-stat-lnk.txt",
  );

  const info = await sb.filesystem.stat("/tmp/e2e-stat-lnk.txt");

  expect(info.type).toBe("symlink");
  expect(info.symlinkTarget).toBe("/tmp/e2e-stat-lnk-target.txt");
});

test("SandboxFsE2eStatSymlinkToDirectoryReportedAsSymlink", async () => {
  const sb = await newTestSandbox();
  await mkdirRemote(sb, "/tmp/e2e-stat-dir-link-target");
  await symlinkRemote(
    sb,
    "/tmp/e2e-stat-dir-link-target",
    "/tmp/e2e-stat-dir-link",
  );

  const info = await sb.filesystem.stat("/tmp/e2e-stat-dir-link");

  expect(info.type).toBe("symlink");
  expect(info.symlinkTarget).toBe("/tmp/e2e-stat-dir-link-target");
});

test("SandboxFsE2eStatDanglingSymlinkReportedAsSymlink", async () => {
  const sb = await newTestSandbox();
  await symlinkRemote(
    sb,
    "/tmp/e2e-stat-dangling-target",
    "/tmp/e2e-stat-dangling.txt",
  );

  const info = await sb.filesystem.stat("/tmp/e2e-stat-dangling.txt");

  expect(info.type).toBe("symlink");
  expect(info.symlinkTarget).toBe("/tmp/e2e-stat-dangling-target");
});

test("SandboxFsE2eStatRelativeSymlinkTargetPreserved", async () => {
  const sb = await newTestSandbox();
  await symlinkRemote(sb, "target.txt", "/tmp/e2e-stat-rel-link.txt");

  const info = await sb.filesystem.stat("/tmp/e2e-stat-rel-link.txt");

  expect(info.type).toBe("symlink");
  expect(info.symlinkTarget).toBe("target.txt");
});

test("SandboxFsE2eStatErrorsWhenPathDoesNotExist", async () => {
  const sb = await newTestSandbox();
  await expect(sb.filesystem.stat("/tmp/e2e-stat-nonexistent")).rejects.toThrow(
    SandboxFilesystemNotFoundError,
  );
});

test("SandboxFsE2eStatErrorsWhenAncestorIsAFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(
    sb,
    "/tmp/e2e-stat-blocker",
    new TextEncoder().encode("I am a file"),
  );

  await expect(
    sb.filesystem.stat("/tmp/e2e-stat-blocker/child"),
  ).rejects.toThrow(SandboxFilesystemNotADirectoryError);
});

// ---------------------------------------------------------------------------
// write_bytes
// ---------------------------------------------------------------------------

test("SandboxFsE2eWriteBytesRoundTrip", async () => {
  const sb = await newTestSandbox();
  const payload = new Uint8Array([0x00, 0x01, 0x02, 0xff, 0xfe]);
  await sb.filesystem.writeBytes(payload, "/tmp/e2e-write-bytes.bin");

  expect(await readRemoteFile(sb, "/tmp/e2e-write-bytes.bin")).toEqual(payload);
});

test("SandboxFsE2eWriteBytesWritesEmptyFile", async () => {
  const sb = await newTestSandbox();
  await sb.filesystem.writeBytes(new Uint8Array(0), "/tmp/e2e-wb-empty.bin");

  expect(await readRemoteFile(sb, "/tmp/e2e-wb-empty.bin")).toEqual(
    new Uint8Array(0),
  );
});

test("SandboxFsE2eWriteBytesCreatesParentDirectories", async () => {
  const sb = await newTestSandbox();
  const payload = new TextEncoder().encode("nested");

  await sb.filesystem.writeBytes(payload, "/tmp/e2e-wb-nested/deep/file.bin");

  expect(await readRemoteFile(sb, "/tmp/e2e-wb-nested/deep/file.bin")).toEqual(
    payload,
  );
});

test("SandboxFsE2eWriteBytesOverwritesExistingFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(
    sb,
    "/tmp/e2e-wb-overwrite.bin",
    new TextEncoder().encode("old"),
  );

  await sb.filesystem.writeBytes(
    new TextEncoder().encode("new"),
    "/tmp/e2e-wb-overwrite.bin",
  );

  expect(
    new TextDecoder().decode(
      await readRemoteFile(sb, "/tmp/e2e-wb-overwrite.bin"),
    ),
  ).toBe("new");
});

test("SandboxFsE2eWriteBytesErrorsWhenRemotePathIsDirectory", async () => {
  const sb = await newTestSandbox();
  await mkdirRemote(sb, "/tmp/e2e-wb-isdir");

  await expect(
    sb.filesystem.writeBytes(
      new TextEncoder().encode("data"),
      "/tmp/e2e-wb-isdir",
    ),
  ).rejects.toThrow(SandboxFilesystemIsADirectoryError);
});

test("SandboxFsE2eWriteBytesErrorsWhenParentIsAFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(
    sb,
    "/tmp/e2e-wb-blocker",
    new TextEncoder().encode("abc"),
  );

  await expect(
    sb.filesystem.writeBytes(
      new TextEncoder().encode("data"),
      "/tmp/e2e-wb-blocker/child.bin",
    ),
  ).rejects.toThrow(SandboxFilesystemNotADirectoryError);
});

test("SandboxFsE2eWriteBytesLargeFileSpansMultipleChunks", async () => {
  const sb = await newTestSandbox();
  const payload = randomBytes(WRITE_CHUNK_SIZE + 1024, 40);
  await sb.filesystem.writeBytes(payload, "/tmp/e2e-write-large.bin");

  const result = await readRemoteFile(sb, "/tmp/e2e-write-large.bin");
  expect(result.length).toBe(payload.length);
  expect(Buffer.from(result).equals(Buffer.from(payload))).toBe(true);
});

test("SandboxFsE2eWriteBytesAcceptsArrayBuffer", async () => {
  const sb = await newTestSandbox();
  const raw = new Uint8Array([0x00, 0x01, 0x02, 0x6d, 0x76, 0xff]);

  await sb.filesystem.writeBytes(raw.buffer, "/tmp/e2e-wb-arraybuffer.bin");

  expect(await readRemoteFile(sb, "/tmp/e2e-wb-arraybuffer.bin")).toEqual(raw);
});

// ---------------------------------------------------------------------------
// write_text
// ---------------------------------------------------------------------------

test("SandboxFsE2eWriteTextRoundTrip", async () => {
  const sb = await newTestSandbox();
  const text = "round-trip text\nwith unicode: ☃🎉\n";
  await sb.filesystem.writeText(text, "/tmp/e2e-write-text.txt");

  expect(
    new TextDecoder().decode(
      await readRemoteFile(sb, "/tmp/e2e-write-text.txt"),
    ),
  ).toBe(text);
});

test("SandboxFsE2eWriteTextWritesEmptyString", async () => {
  const sb = await newTestSandbox();
  await sb.filesystem.writeText("", "/tmp/e2e-wt-empty.txt");

  expect(await readRemoteFile(sb, "/tmp/e2e-wt-empty.txt")).toEqual(
    new Uint8Array(0),
  );
});

test("SandboxFsE2eWriteTextCreatesParentDirectories", async () => {
  const sb = await newTestSandbox();
  await sb.filesystem.writeText(
    "nested text",
    "/tmp/e2e-wt-nested/deep/write.txt",
  );

  expect(
    new TextDecoder().decode(
      await readRemoteFile(sb, "/tmp/e2e-wt-nested/deep/write.txt"),
    ),
  ).toBe("nested text");
});

test("SandboxFsE2eWriteTextOverwritesExistingFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(
    sb,
    "/tmp/e2e-wt-overwrite.txt",
    new TextEncoder().encode("old"),
  );

  await sb.filesystem.writeText("new-data", "/tmp/e2e-wt-overwrite.txt");

  expect(
    new TextDecoder().decode(
      await readRemoteFile(sb, "/tmp/e2e-wt-overwrite.txt"),
    ),
  ).toBe("new-data");
});

test("SandboxFsE2eWriteTextErrorsWhenRemotePathIsDirectory", async () => {
  const sb = await newTestSandbox();
  await mkdirRemote(sb, "/tmp/e2e-wt-isdir");

  await expect(
    sb.filesystem.writeText("data", "/tmp/e2e-wt-isdir"),
  ).rejects.toThrow(SandboxFilesystemIsADirectoryError);
});

test("SandboxFsE2eWriteTextErrorsWhenParentIsAFile", async () => {
  const sb = await newTestSandbox();
  await writeRemoteFile(
    sb,
    "/tmp/e2e-wt-blocker",
    new TextEncoder().encode("file"),
  );

  await expect(
    sb.filesystem.writeText("data", "/tmp/e2e-wt-blocker/child.txt"),
  ).rejects.toThrow(SandboxFilesystemNotADirectoryError);
});
