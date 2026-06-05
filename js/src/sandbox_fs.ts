import { randomBytes } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, rename, rm, unlink } from "node:fs/promises";
import { Writable } from "node:stream";
import { dirname } from "node:path";

import { ClientError, Status } from "nice-grpc";

import type { ContainerProcess, SandboxExecParams } from "./sandbox";
import {
  SANDBOX_FS_TOOLS_PATH,
  makeListFilesCommand,
  makeMakeDirectoryCommand,
  makeReadFileCommand,
  makeRemoveCommand,
  makeStatCommand,
  makeWriteFileCommand,
  raiseListFilesError,
  raiseMakeDirectoryError,
  raiseReadFileError,
  raiseRemoveError,
  raiseStatError,
  raiseWriteFileError,
  translateExecErrors,
  validateAbsoluteRemotePath,
} from "./sandbox_fs_utils";

/** Type of a filesystem entry. */
export type FileType = "file" | "directory" | "symlink";

/** Metadata for a file or directory entry in a Sandbox. */
export interface FileInfo {
  readonly name: string;
  readonly path: string;
  readonly type: FileType;
  readonly size: number;
  readonly mode: number;
  readonly permissions: string;
  readonly owner: string;
  readonly group: string;
  /** Unix epoch seconds. */
  readonly modifiedTime: number;
  readonly symlinkTarget: string | null;
}

interface RawFileEntry {
  name: string;
  path: string;
  type: string;
  size: number;
  mode: number;
  permissions: string;
  owner: string;
  group: string;
  modified_time: number;
  symlink_target?: string;
}

function parseFileEntry(raw: RawFileEntry): FileInfo {
  return {
    name: raw.name,
    path: raw.path,
    type: raw.type as FileType,
    size: raw.size,
    mode: raw.mode,
    permissions: raw.permissions,
    owner: raw.owner,
    group: raw.group,
    modifiedTime: raw.modified_time,
    symlinkTarget: raw.symlink_target ?? null,
  };
}

// 4 MiB chunks due to Node.js's http2 10 MB maxSessionMemory.
const WRITE_CHUNK_SIZE = 4 * 1024 * 1024;

/** Normalize the user-facing data union to a Uint8Array view. */
function toUint8Array(data: Uint8Array | ArrayBuffer | Buffer): Uint8Array {
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (data instanceof Uint8Array) return data; // covers Buffer
  throw new TypeError(
    `Expected Uint8Array, ArrayBuffer, or Buffer, got ${typeof data}`,
  );
}

/** Namespace for Sandbox filesystem APIs. */
export class SandboxFilesystem {
  /** @ignore */
  constructor(
    private readonly exec: (
      command: string[],
      params?: SandboxExecParams,
    ) => Promise<ContainerProcess>,
  ) {}

  /**
   * Copy a local file into the Sandbox.
   *
   * `remotePath` must be an absolute path to a file in the Sandbox.
   * Parent directories are created if needed. The remote file is overwritten
   * if it already exists.
   *
   * @throws {SandboxFilesystemNotADirectoryError} a parent component of `remotePath` is not a directory.
   * @throws {SandboxFilesystemIsADirectoryError} `remotePath` points to a directory.
   * @throws {SandboxFilesystemPermissionError} write permission is denied in the Sandbox.
   * @throws {SandboxFilesystemError} the command fails for any other reason.
   * @throws {Error} `localPath` does not exist, is a directory, or cannot be read (`ENOENT`, `EISDIR`, `EACCES`).
   */
  async copyFromLocal(localPath: string, remotePath: string): Promise<void> {
    validateAbsoluteRemotePath(remotePath, "copyFromLocal");

    await translateExecErrors("copyFromLocal", remotePath, async () => {
      const process = await this.exec(
        [SANDBOX_FS_TOOLS_PATH, makeWriteFileCommand(remotePath)],
        { mode: "binary" },
      );

      const writer = process.stdin.getWriter();
      try {
        const fileStream = createReadStream(localPath);
        for await (const chunk of fileStream) {
          await writer.write(chunk as Buffer);
        }
        await writer.close();
      } catch (err) {
        if (
          err instanceof ClientError &&
          (err.code === Status.FAILED_PRECONDITION ||
            err.code === Status.ABORTED)
        ) {
          await process.closeStdin();
        } else {
          // Abort rather than close so the remote process sees an error, not EOF.
          // Closing with EOF would cause it to finalize the file with partial content.
          await writer.abort(err).catch(() => {});
          throw err;
        }
      } finally {
        writer.releaseLock();
      }

      const returnCode = await process.wait();
      if (returnCode !== 0) {
        const stderr = await process.stderr.readBytes();
        raiseWriteFileError(returnCode, stderr, remotePath);
      }
    });
  }

  /**
   * Copy a file from the Sandbox to a local path.
   *
   * `remotePath` must be an absolute path to a file in the Sandbox.
   * Parent directories for `localPath` are created if needed. The local file
   * is overwritten if it already exists.
   *
   * @throws {SandboxFilesystemNotFoundError} the remote path does not exist.
   * @throws {SandboxFilesystemIsADirectoryError} the remote path points to a directory.
   * @throws {SandboxFilesystemFileTooLargeError} the file exceeds the read size limit.
   * @throws {SandboxFilesystemPermissionError} read permission is denied in the Sandbox.
   * @throws {SandboxFilesystemError} the command fails for any other reason.
   * @throws {Error} `localPath` points to a directory, or writing it is not permitted.
   */
  async copyToLocal(remotePath: string, localPath: string): Promise<void> {
    validateAbsoluteRemotePath(remotePath, "copyToLocal");

    const dir = dirname(localPath);
    const createdDir = await mkdir(dir, { recursive: true });
    const tmpPath = `${dir}/.modal-sandbox-fs-tmp-${randomBytes(8).toString("hex")}`;

    try {
      await translateExecErrors("copyToLocal", remotePath, async () => {
        const process = await this.exec(
          [SANDBOX_FS_TOOLS_PATH, makeReadFileCommand(remotePath)],
          { mode: "binary" },
        );

        // After the binary exits, all stdout is already on the disk server-side.
        // Now, start the lazy gRPC stream.
        const returnCode = await process.wait();
        if (returnCode !== 0) {
          const stderr = await process.stderr.readBytes();
          raiseReadFileError(returnCode, stderr, remotePath);
        }
        await process.stdout.pipeTo(Writable.toWeb(createWriteStream(tmpPath)));
      });
      await rename(tmpPath, localPath);
    } catch (err) {
      await unlink(tmpPath).catch(() => {});
      if (createdDir !== undefined) {
        await rm(createdDir, { recursive: true, force: true }).catch(() => {});
      }
      throw err;
    }
  }

  /**
   * List files and directories in a Sandbox directory.
   *
   * `remotePath` must be an absolute path to a directory in the Sandbox.
   * Returns an array of {@link FileInfo} objects sorted by name.
   *
   * @throws {SandboxFilesystemNotFoundError} the path does not exist.
   * @throws {SandboxFilesystemNotADirectoryError} the path is not a directory.
   * @throws {SandboxFilesystemPermissionError} read permission is denied.
   * @throws {SandboxFilesystemError} the command fails for any other reason.
   */
  async listFiles(remotePath: string): Promise<FileInfo[]> {
    validateAbsoluteRemotePath(remotePath, "listFiles");

    const json = await translateExecErrors(
      "listFiles",
      remotePath,
      async () => {
        const process = await this.exec(
          [SANDBOX_FS_TOOLS_PATH, makeListFilesCommand(remotePath)],
          { mode: "text" },
        );

        const returnCode = await process.wait();
        if (returnCode !== 0) {
          const stderr = await process.stderr.readBytes();
          raiseListFilesError(returnCode, stderr, remotePath);
        }
        return process.stdout.readText();
      },
    );

    return (JSON.parse(json) as RawFileEntry[]).map(parseFileEntry);
  }

  /**
   * Create a new directory in the Sandbox.
   *
   * `remotePath` must be an absolute path in the Sandbox.
   *
   * When `createParents` is `true` (the default), any missing parent
   * directories are created and the call is idempotent (succeeds if the
   * directory already exists). When `createParents` is `false`, the immediate
   * parent must already exist and the path must not already exist.
   *
   * @throws {SandboxFilesystemNotFoundError} the parent does not exist and `createParents` is `false`.
   * @throws {SandboxFilesystemPathAlreadyExistsError} the path already exists and `createParents` is `false`.
   * @throws {SandboxFilesystemNotADirectoryError} a path component is not a directory.
   * @throws {SandboxFilesystemPermissionError} creation is not permitted.
   * @throws {InvalidError} the operation is not supported by the mount.
   * @throws {SandboxFilesystemError} the command fails for any other reason.
   */
  async makeDirectory(
    remotePath: string,
    options?: { createParents?: boolean },
  ): Promise<void> {
    validateAbsoluteRemotePath(remotePath, "makeDirectory");
    const createParents = options?.createParents ?? true;

    await translateExecErrors("makeDirectory", remotePath, async () => {
      const process = await this.exec(
        [
          SANDBOX_FS_TOOLS_PATH,
          makeMakeDirectoryCommand(remotePath, createParents),
        ],
        { mode: "binary" },
      );

      const returnCode = await process.wait();
      if (returnCode !== 0) {
        const stderr = await process.stderr.readBytes();
        raiseMakeDirectoryError(returnCode, stderr, remotePath);
      }
    });
  }

  /**
   * Read a file from the Sandbox and return its contents as bytes.
   *
   * `remotePath` must be an absolute path to a file in the Sandbox.
   *
   * @throws {SandboxFilesystemNotFoundError} the path does not exist.
   * @throws {SandboxFilesystemIsADirectoryError} the path points to a directory.
   * @throws {SandboxFilesystemFileTooLargeError} the file exceeds the read size limit.
   * @throws {SandboxFilesystemPermissionError} read permission is denied.
   * @throws {SandboxFilesystemError} the command fails for any other reason.
   */
  async readBytes(remotePath: string): Promise<Uint8Array> {
    validateAbsoluteRemotePath(remotePath, "readBytes");

    return translateExecErrors("readBytes", remotePath, async () => {
      const process = await this.exec(
        [SANDBOX_FS_TOOLS_PATH, makeReadFileCommand(remotePath)],
        { mode: "binary" },
      );

      const returnCode = await process.wait();
      if (returnCode !== 0) {
        const stderr = await process.stderr.readBytes();
        raiseReadFileError(returnCode, stderr, remotePath);
      }
      return process.stdout.readBytes();
    });
  }

  /**
   * Read a file from the Sandbox and return its contents as a UTF-8 string.
   *
   * `remotePath` must be an absolute path to a file in the Sandbox.
   *
   * @throws {SandboxFilesystemNotFoundError} the path does not exist.
   * @throws {SandboxFilesystemIsADirectoryError} the path points to a directory.
   * @throws {SandboxFilesystemFileTooLargeError} the file exceeds the read size limit.
   * @throws {SandboxFilesystemPermissionError} read permission is denied.
   * @throws {SandboxFilesystemError} the command fails for any other reason.
   */
  async readText(remotePath: string): Promise<string> {
    validateAbsoluteRemotePath(remotePath, "readText");

    return translateExecErrors("readText", remotePath, async () => {
      const process = await this.exec(
        [SANDBOX_FS_TOOLS_PATH, makeReadFileCommand(remotePath)],
        { mode: "text" },
      );

      const returnCode = await process.wait();
      if (returnCode !== 0) {
        const stderr = await process.stderr.readBytes();
        raiseReadFileError(returnCode, stderr, remotePath);
      }
      return process.stdout.readText();
    });
  }

  /**
   * Remove a file or directory in the Sandbox.
   *
   * `remotePath` must be an absolute path in the Sandbox. When `remotePath`
   * is a directory and `recursive` is `false` (the default), it is removed
   * only if empty. When `recursive` is `true`, the directory and all its
   * contents are removed. Recursive removal is not supported on all mounts —
   * `CloudBucketMount` does not support it.
   *
   * @throws {SandboxFilesystemNotFoundError} the path does not exist.
   * @throws {SandboxFilesystemDirectoryNotEmptyError} `recursive` is `false` and the directory is not empty.
   * @throws {SandboxFilesystemPermissionError} removal is not permitted.
   * @throws {InvalidError} the operation is not supported by the mount.
   * @throws {SandboxFilesystemError} the command fails for any other reason.
   */
  async remove(
    remotePath: string,
    options?: { recursive?: boolean },
  ): Promise<void> {
    validateAbsoluteRemotePath(remotePath, "remove");
    const recursive = options?.recursive ?? false;

    await translateExecErrors("remove", remotePath, async () => {
      const process = await this.exec(
        [SANDBOX_FS_TOOLS_PATH, makeRemoveCommand(remotePath, recursive)],
        { mode: "binary" },
      );

      const returnCode = await process.wait();
      if (returnCode !== 0) {
        const stderr = await process.stderr.readBytes();
        raiseRemoveError(returnCode, stderr, remotePath);
      }
    });
  }

  /**
   * Return metadata for a single file, directory, or symlink in the Sandbox.
   *
   * `remotePath` must be an absolute path in the Sandbox. If `remotePath` is
   * a symlink, the returned {@link FileInfo} describes the symlink itself, not
   * the target it points to.
   *
   * @throws {SandboxFilesystemNotFoundError} the path does not exist.
   * @throws {SandboxFilesystemNotADirectoryError} a non-leaf component of the path is not a directory.
   * @throws {SandboxFilesystemPermissionError} a path component is not searchable.
   * @throws {SandboxFilesystemError} the command fails for any other reason.
   */
  async stat(remotePath: string): Promise<FileInfo> {
    validateAbsoluteRemotePath(remotePath, "stat");

    const stdout = await translateExecErrors("stat", remotePath, async () => {
      const process = await this.exec(
        [SANDBOX_FS_TOOLS_PATH, makeStatCommand(remotePath)],
        { mode: "text" },
      );
      const returnCode = await process.wait();
      if (returnCode !== 0) {
        const stderr = await process.stderr.readBytes();
        raiseStatError(returnCode, stderr, remotePath);
      }
      return await process.stdout.readText();
    });

    return parseFileEntry(JSON.parse(stdout) as RawFileEntry);
  }

  /**
   * Write binary content to a file in the Sandbox.
   *
   * `remotePath` must be an absolute path to a file in the Sandbox.
   * Parent directories are created if needed. The remote file is overwritten
   * if it already exists.
   *
   * @throws {TypeError} `data` is not a `Uint8Array`, `ArrayBuffer`, or `Buffer`.
   * @throws {SandboxFilesystemNotADirectoryError} a parent component of `remotePath` is not a directory.
   * @throws {SandboxFilesystemIsADirectoryError} `remotePath` points to a directory.
   * @throws {SandboxFilesystemPermissionError} write permission is denied.
   * @throws {SandboxFilesystemError} the command fails for any other reason.
   */
  async writeBytes(
    data: Uint8Array | ArrayBuffer | Buffer,
    remotePath: string,
  ): Promise<void> {
    validateAbsoluteRemotePath(remotePath, "writeBytes");
    const bytes = toUint8Array(data);

    await translateExecErrors("writeBytes", remotePath, async () => {
      const process = await this.exec(
        [SANDBOX_FS_TOOLS_PATH, makeWriteFileCommand(remotePath)],
        { mode: "binary" },
      );

      const writer = process.stdin.getWriter();
      try {
        // At least one write so empty data still creates the file.
        const end = Math.max(bytes.length, 1);
        for (let offset = 0; offset < end; offset += WRITE_CHUNK_SIZE) {
          await writer.write(bytes.subarray(offset, offset + WRITE_CHUNK_SIZE));
        }
        await writer.close();
      } catch (err) {
        if (
          err instanceof ClientError &&
          (err.code === Status.FAILED_PRECONDITION ||
            err.code === Status.ABORTED)
        ) {
          await process.closeStdin();
        } else {
          // Abort rather than close so the remote process sees an error, not EOF.
          // Closing with EOF would cause it to finalize the file with partial content.
          await writer.abort(err).catch(() => {});
          throw err;
        }
      } finally {
        writer.releaseLock();
      }

      const returnCode = await process.wait();
      if (returnCode !== 0) {
        const stderr = await process.stderr.readBytes();
        raiseWriteFileError(returnCode, stderr, remotePath);
      }
    });
  }

  /**
   * Write UTF-8 text to a file in the Sandbox.
   *
   * `remotePath` must be an absolute path to a file in the Sandbox.
   * Parent directories are created if needed. The remote file is overwritten
   * if it already exists.
   *
   * @throws {TypeError} `data` is not a string.
   * @throws {SandboxFilesystemNotADirectoryError} a parent component of `remotePath` is not a directory.
   * @throws {SandboxFilesystemIsADirectoryError} `remotePath` points to a directory.
   * @throws {SandboxFilesystemPermissionError} write permission is denied.
   * @throws {SandboxFilesystemError} the command fails for any other reason.
   */
  async writeText(data: string, remotePath: string): Promise<void> {
    validateAbsoluteRemotePath(remotePath, "writeText");
    if (typeof data !== "string")
      throw new TypeError(`writeText() expects a string, got ${typeof data}`);
    await this.writeBytes(new TextEncoder().encode(data), remotePath);
  }
}
