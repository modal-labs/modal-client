import { randomBytes } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, open, rename, rm, unlink } from "node:fs/promises";
import type { FileHandle } from "node:fs/promises";
import { Writable } from "node:stream";
import { dirname } from "node:path";

import { ClientError, Status } from "nice-grpc";

import type { ContainerProcess, SandboxExecParams } from "./sandbox";
import {
  STREAMING_STDIN_CHUNK_SIZE,
  type StdinSource,
} from "./task_command_router_client";
import {
  SANDBOX_FS_TOOLS_PATH,
  makeListFilesCommand,
  makeMakeDirectoryCommand,
  makeReadFileCommand,
  makeRemoveCommand,
  makeStatCommand,
  makeWatchCommand,
  makeWriteFileCommand,
  raiseListFilesError,
  raiseMakeDirectoryError,
  raiseReadFileError,
  raiseRemoveError,
  raiseStatError,
  raiseWatchError,
  raiseWriteFileError,
  translateExecErrors,
  validateAbsoluteRemotePath,
} from "./sandbox_fs_utils";
import { checkForRenamedParams } from "./validation";

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

/** Type of a filesystem watch event. */
export type FileWatchEventType =
  | "Access"
  | "Create"
  | "Modify"
  | "Remove"
  | "Unknown";

/** A filesystem change event.
 *
 * `paths` contains the absolute path(s) affected by the event. For most event
 * types it holds a single entry. Rename operations are reported as `Modify`
 * events: when both the source and destination fall within the watched scope,
 * `paths` holds `[source, destination]`; when only one side of the rename is
 * visible, `paths` holds that single path.
 */
export interface FileWatchEvent {
  readonly eventType: FileWatchEventType;
  readonly paths: string[];
}

const RUST_RENAME_VARIANTS = ["Rename", "RenameFrom", "RenameTo"] as const;

/** Event type strings recognized after collapsing the Rust rename variants. */
const VALID_EVENT_TYPES = new Set<string>([
  "Access",
  "Create",
  "Modify",
  "Remove",
  "Unknown",
]);

/**
 * @internal
 * @hidden
 */
export function expandWatchFilter(filter: FileWatchEventType[]): string[] {
  const result: string[] = [];
  for (const eventType of filter) {
    result.push(eventType);
    if (eventType === "Modify") {
      result.push(...RUST_RENAME_VARIANTS);
    }
  }
  return result;
}

/**
 * Yield complete lines from a stream of string or byte chunks.
 */
async function* readLines(
  stream: AsyncIterable<string | Uint8Array>,
): AsyncGenerator<string, void, void> {
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of stream) {
    buffer +=
      typeof chunk === "string"
        ? chunk
        : decoder.decode(chunk, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) yield line;
  }
  const tail = decoder.decode(undefined, { stream: false });
  if (buffer + tail) yield buffer + tail;
}

/** In-memory byte source supporting resume from an arbitrary offset. */
function bytesSource(bytes: Uint8Array): StdinSource {
  return {
    async *readFrom(offset: number) {
      yield bytes.subarray(offset);
    },
  };
}

/** Local file byte source supporting resume from an arbitrary offset. */
function fileHandleSource(handle: FileHandle): StdinSource {
  return {
    readFrom(offset: number): AsyncIterable<Uint8Array> {
      // autoClose: false so a failed attempt's stream does not close the
      // handle out from under a resume attempt.
      return handle.createReadStream({
        start: offset,
        autoClose: false,
        highWaterMark: STREAMING_STDIN_CHUNK_SIZE,
      });
    },
  };
}

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

    // Open eagerly so local errors like ENOENT surface directly rather than
    // mid-stream.
    const handle = await open(localPath, "r");
    try {
      await this.#execFsToolWrite(
        fileHandleSource(handle),
        remotePath,
        "copyFromLocal",
      );
    } finally {
      await handle.close();
    }
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
   * Watch a path in the Sandbox for filesystem changes.
   *
   * `remotePath` must be an absolute path in the Sandbox. If it points to a
   * file, events for that file are reported. If it points to a directory,
   * events for entries directly inside it are reported. Set `recursive: true`
   * to also receive events for all nested subdirectories. If `remotePath` is
   * a symlink, it is followed and events reference paths under the resolved
   * target.
   *
   * Yields {@link FileWatchEvent} objects as changes occur, until either the
   * timeout elapses, the iterator is closed, or the Sandbox is terminated.
   *
   * Optionally restrict the kinds of events emitted to those included in
   * `filter`. An undefined `filter` permits all types; passing an empty array
   * suppresses all events.
   *
   * `timeoutMs` is truncated to whole seconds. Omit it to watch indefinitely.
   * When the timeout elapses, the iterator stops without raising an exception.
   *
   * @throws {SandboxFilesystemNotFoundError} `remotePath` does not exist.
   * @throws {SandboxFilesystemPermissionError} watch access is denied.
   * @throws {InvalidError} the filesystem does not support watching.
   * @throws {SandboxFilesystemError} the command fails for any other reason.
   */
  async *watch(
    remotePath: string,
    params: {
      filter?: FileWatchEventType[];
      recursive?: boolean;
      timeoutMs?: number;
    } = {},
  ): AsyncIterable<FileWatchEvent> {
    validateAbsoluteRemotePath(remotePath, "watch");
    checkForRenamedParams(params, { timeout: "timeoutMs" });
    const { filter, recursive = false, timeoutMs } = params;

    const process = await translateExecErrors("watch", remotePath, () =>
      this.exec(
        [
          SANDBOX_FS_TOOLS_PATH,
          makeWatchCommand(remotePath, {
            recursive,
            filter: filter !== undefined ? expandWatchFilter(filter) : null,
            timeoutMs: timeoutMs ?? null,
          }),
        ],
        { mode: "text" },
      ),
    );

    // Distinguishes the stream ending on its own (timeout elapsed, Sandbox
    // terminated, or the command failed at startup) from the consumer stopping
    // early via `break`/`return`. We only surface a non-zero exit as an error
    // in the former case; an early stop must not throw over the consumer.
    let streamEnded = false;
    try {
      for await (const line of readLines(process.stdout)) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        let data: { event_type?: string; paths?: string[] };
        try {
          data = JSON.parse(trimmed) as {
            event_type?: string;
            paths?: string[];
          };
        } catch {
          continue;
        }
        const paths = data.paths ?? [];
        if (paths.length === 0) continue;
        let rawType = data.event_type;
        if (typeof rawType !== "string") continue;
        if ((RUST_RENAME_VARIANTS as readonly string[]).includes(rawType)) {
          rawType = "Modify";
        }
        // Drop events with an unrecognized type rather than surfacing them.
        if (!VALID_EVENT_TYPES.has(rawType)) continue;
        yield { eventType: rawType as FileWatchEventType, paths };
      }
      streamEnded = true;
    } catch (err) {
      // A failure while consuming the stream (e.g. the Sandbox is terminated
      // mid-watch) surfaces as a raw transport error. Route it through the same
      // translation the other filesystem methods apply so callers get a
      // consistent, friendly error rather than a gRPC-level exception.
      await translateExecErrors("watch", remotePath, () => {
        throw err;
      });
      throw err; // Unreachable: translateExecErrors always rethrows.
    } finally {
      // Close stdin so the fs-tools process detects EOF and exits promptly.
      await process.closeStdin().catch(() => {});
      if (streamEnded) {
        await translateExecErrors("watch", remotePath, async () => {
          const returnCode = await process.wait();
          if (returnCode !== 0) {
            const stderr = await process.stderr.readBytes();
            raiseWatchError(returnCode, stderr, remotePath);
          }
        });
      } else {
        // Consumer stopped early; reap the process without surfacing its exit
        // status (e.g. a signal-kill) as an error over their `break`.
        await process.wait().catch(() => {});
      }
    }
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

    await this.#execFsToolWrite(bytesSource(bytes), remotePath, "writeBytes");
  }

  /**
   * Exec the FS-tools write command and stream `source` into its stdin.
   *
   * A local read error aborts the stream without sending EOF, so the remote
   * side never finalizes a partial file.
   */
  async #execFsToolWrite(
    source: StdinSource,
    remotePath: string,
    opName: string,
  ): Promise<void> {
    await translateExecErrors(opName, remotePath, async () => {
      const process = await this.exec(
        [SANDBOX_FS_TOOLS_PATH, makeWriteFileCommand(remotePath)],
        { mode: "binary" },
      );

      // TODO(saltzm): If streaming fails after resume attempts are exhausted,
      // the exec'd process remains alive indefinitely since stdin stays open.
      // We should kill the process in that case when we have a way to do so.
      try {
        await process._stdinWriteStream(source);
      } catch (err) {
        // When the FS tools binary exits early on an error, the worker
        // reports the dropped stdin write as FAILED_PRECONDITION or ABORTED.
        if (
          err instanceof ClientError &&
          (err.code === Status.FAILED_PRECONDITION ||
            err.code === Status.ABORTED)
        ) {
          // The error can come from a failure in fs-tools or server.
          // If server, the process won't exit, so the wait below would
          // hang forever — rethrow. Otherwise fall through and let
          // raiseWriteFileError below surface the real filesystem error.
          if ((await process._poll()) === null) {
            throw err;
          }
        } else {
          throw err;
        }
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
