import { posix } from "node:path";

import { ClientError, Status } from "nice-grpc";

import {
  InvalidError,
  NotFoundError,
  SandboxFilesystemDirectoryNotEmptyError,
  SandboxFilesystemError,
  SandboxFilesystemFileTooLargeError,
  SandboxFilesystemIsADirectoryError,
  SandboxFilesystemNotADirectoryError,
  SandboxFilesystemNotFoundError,
  SandboxFilesystemPathAlreadyExistsError,
  SandboxFilesystemPermissionError,
} from "./errors";
import { newLogger } from "./logger";

// Module-level logger for diagnostic debug output on failure paths.
const logger = newLogger();

export const SANDBOX_FS_TOOLS_PATH = "/__modal/.bin/modal-sandbox-fs-tools";

export function validateAbsoluteRemotePath(
  remotePath: string,
  operation: string,
): void {
  if (!posix.isAbsolute(remotePath)) {
    throw new InvalidError(
      `Sandbox.filesystem.${operation}() currently only supports absolute remotePath values`,
    );
  }
}

export interface ErrorPayload {
  error_kind: string;
  message: string;
  detail: string;
}

function stderrToText(stderr: string | Uint8Array): string {
  if (typeof stderr === "string") return stderr.trim();
  return new TextDecoder("utf-8", { fatal: false }).decode(stderr).trim();
}

export function tryParseErrorPayload(
  stderr: string | Uint8Array,
): ErrorPayload | null {
  const text = stderrToText(stderr);
  if (!text) return null;

  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof payload !== "object" || payload === null) return null;

  const obj = payload as Record<string, unknown>;
  const errorKind = obj.error_kind;
  const message = obj.message;
  if (typeof errorKind !== "string") return null;
  if (typeof message !== "string" || !message.trim()) return null;

  const detailRaw = obj.detail;
  const detail = typeof detailRaw === "string" ? detailRaw : "";

  return { error_kind: errorKind, message, detail };
}

// All make*Command outputs must match the `Command` enum in the
// modal-sandbox-fs-tools Rust crate (crates/modal-sandbox-fs-tools/src/lib.rs).
// Treat changes to this schema like protobuf changes: fields must not be
// removed or renamed, only added with backwards-compatible defaults.

export function raiseListFilesError(
  returnCode: number,
  stderr: string | Uint8Array,
  remotePath: string,
): never {
  const payload = tryParseErrorPayload(stderr);
  if (payload) {
    logger.debug(
      `sandbox-fs-tools listFiles error: path=${remotePath}, ` +
        `error_kind=${payload.error_kind}, message=${payload.message}, detail=${payload.detail}`,
    );
    if (payload.error_kind === "NotFound")
      throw new SandboxFilesystemNotFoundError(
        `${payload.message}: ${remotePath}`,
      );
    if (
      payload.error_kind === "IsFile" ||
      payload.error_kind === "NotDirectory"
    )
      throw new SandboxFilesystemNotADirectoryError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "PermissionDenied")
      throw new SandboxFilesystemPermissionError(
        `${payload.message}: ${remotePath}`,
      );
    throw new SandboxFilesystemError(payload.message);
  }
  const text = stderrToText(stderr);
  if (text) logger.debug(`Unstructured modal-sandbox-fs-tools stderr: ${text}`);
  throw new SandboxFilesystemError(
    `Operation on '${remotePath}' failed with exit code ${returnCode}`,
  );
}

export function makeListFilesCommand(remotePath: string): string {
  return JSON.stringify({ ListFiles: { path: remotePath } });
}

export function raiseMakeDirectoryError(
  returnCode: number,
  stderr: string | Uint8Array,
  remotePath: string,
): never {
  const payload = tryParseErrorPayload(stderr);
  if (payload) {
    logger.debug(
      `sandbox-fs-tools makeDirectory error: path=${remotePath}, ` +
        `error_kind=${payload.error_kind}, message=${payload.message}, detail=${payload.detail}`,
    );
    if (payload.error_kind === "NotFound")
      throw new SandboxFilesystemNotFoundError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "PathAlreadyExists")
      throw new SandboxFilesystemPathAlreadyExistsError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "NotDirectory")
      throw new SandboxFilesystemNotADirectoryError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "PermissionDenied")
      throw new SandboxFilesystemPermissionError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "NotSupported")
      throw new InvalidError(
        `${payload.message}: ${remotePath} - this operation is not supported for CloudBucketMounts`,
      );
    throw new SandboxFilesystemError(payload.message);
  }
  const text = stderrToText(stderr);
  if (text) logger.debug(`Unstructured modal-sandbox-fs-tools stderr: ${text}`);
  throw new SandboxFilesystemError(
    `Operation on '${remotePath}' failed with exit code ${returnCode}`,
  );
}

export function makeMakeDirectoryCommand(
  remotePath: string,
  createParents: boolean,
): string {
  return JSON.stringify({
    MakeDirectory: { path: remotePath, parents: createParents },
  });
}

export function raiseReadFileError(
  returnCode: number,
  stderr: string | Uint8Array,
  remotePath: string,
): never {
  const payload = tryParseErrorPayload(stderr);
  if (payload) {
    logger.debug(
      `sandbox-fs-tools read error: path=${remotePath}, ` +
        `error_kind=${payload.error_kind}, message=${payload.message}, detail=${payload.detail}`,
    );
    if (payload.error_kind === "NotFound")
      throw new SandboxFilesystemNotFoundError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "IsDirectory")
      throw new SandboxFilesystemIsADirectoryError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "PermissionDenied")
      throw new SandboxFilesystemPermissionError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "FileTooLarge")
      throw new SandboxFilesystemFileTooLargeError(
        `${payload.message}: ${remotePath}`,
      );
    throw new SandboxFilesystemError(payload.message);
  }
  const text = stderrToText(stderr);
  if (text) logger.debug(`Unstructured modal-sandbox-fs-tools stderr: ${text}`);
  throw new SandboxFilesystemError(
    `Operation on '${remotePath}' failed with exit code ${returnCode}`,
  );
}

export function makeReadFileCommand(remotePath: string): string {
  return JSON.stringify({ ReadFile: { path: remotePath } });
}

export function raiseRemoveError(
  returnCode: number,
  stderr: string | Uint8Array,
  remotePath: string,
): never {
  const payload = tryParseErrorPayload(stderr);
  if (payload) {
    logger.debug(
      `sandbox-fs-tools remove error: path=${remotePath}, ` +
        `error_kind=${payload.error_kind}, message=${payload.message}, detail=${payload.detail}`,
    );
    if (payload.error_kind === "NotFound")
      throw new SandboxFilesystemNotFoundError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "DirectoryNotEmpty")
      throw new SandboxFilesystemDirectoryNotEmptyError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "NotSupported")
      throw new InvalidError(
        `${payload.message}: ${remotePath} - this operation is not supported for CloudBucketMounts`,
      );
    if (payload.error_kind === "PermissionDenied")
      throw new SandboxFilesystemPermissionError(
        `${payload.message}: ${remotePath}`,
      );
    throw new SandboxFilesystemError(payload.message);
  }
  const text = stderrToText(stderr);
  if (text) logger.debug(`Unstructured modal-sandbox-fs-tools stderr: ${text}`);
  throw new SandboxFilesystemError(
    `Operation on '${remotePath}' failed with exit code ${returnCode}`,
  );
}

export function makeRemoveCommand(
  remotePath: string,
  recursive: boolean,
): string {
  return JSON.stringify({ Remove: { path: remotePath, recursive } });
}

export function raiseStatError(
  returnCode: number,
  stderr: string | Uint8Array,
  remotePath: string,
): never {
  const payload = tryParseErrorPayload(stderr);
  if (payload) {
    logger.debug(
      `sandbox-fs-tools stat error: path=${remotePath}, ` +
        `error_kind=${payload.error_kind}, message=${payload.message}, detail=${payload.detail}`,
    );
    if (payload.error_kind === "NotFound")
      throw new SandboxFilesystemNotFoundError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "NotDirectory")
      throw new SandboxFilesystemNotADirectoryError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "PermissionDenied")
      throw new SandboxFilesystemPermissionError(
        `${payload.message}: ${remotePath}`,
      );
    throw new SandboxFilesystemError(payload.message);
  }
  const text = stderrToText(stderr);
  if (text) logger.debug(`Unstructured modal-sandbox-fs-tools stderr: ${text}`);
  throw new SandboxFilesystemError(
    `Operation on '${remotePath}' failed with exit code ${returnCode}`,
  );
}

export function makeStatCommand(remotePath: string): string {
  return JSON.stringify({ Stat: { path: remotePath } });
}

export function raiseWatchError(
  returnCode: number,
  stderr: string | Uint8Array,
  remotePath: string,
): never {
  const payload = tryParseErrorPayload(stderr);
  if (payload) {
    logger.debug(
      `sandbox-fs-tools watch error: path=${remotePath}, ` +
        `error_kind=${payload.error_kind}, message=${payload.message}, detail=${payload.detail}`,
    );
    if (payload.error_kind === "NotFound")
      throw new SandboxFilesystemNotFoundError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "PermissionDenied")
      throw new SandboxFilesystemPermissionError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "NotSupported")
      throw new InvalidError(
        `${payload.message}: ${remotePath} - this operation is not supported for CloudBucketMounts`,
      );
    throw new SandboxFilesystemError(payload.message);
  }
  const text = stderrToText(stderr);
  if (text) logger.debug(`Unstructured modal-sandbox-fs-tools stderr: ${text}`);
  throw new SandboxFilesystemError(
    `Operation on '${remotePath}' failed with exit code ${returnCode}`,
  );
}

export function makeWatchCommand(
  remotePath: string,
  opts: {
    recursive: boolean;
    filter: string[] | null;
    timeoutMs: number | null;
  },
): string {
  return JSON.stringify({
    Watch: {
      path: remotePath,
      recursive: opts.recursive,
      filter: opts.filter,
      // The fs-tools wire protocol takes whole seconds; convert from the SDK's
      // millisecond convention, truncating any sub-second remainder.
      timeout_secs:
        opts.timeoutMs !== null ? Math.trunc(opts.timeoutMs / 1000) : null,
    },
  });
}

export function raiseWriteFileError(
  returnCode: number,
  stderr: string | Uint8Array,
  remotePath: string,
): never {
  const payload = tryParseErrorPayload(stderr);
  if (payload) {
    logger.debug(
      `sandbox-fs-tools write error: path=${remotePath}, ` +
        `error_kind=${payload.error_kind}, message=${payload.message}, detail=${payload.detail}`,
    );
    if (
      payload.error_kind === "NotDirectory" ||
      payload.error_kind === "AlreadyExists"
    )
      throw new SandboxFilesystemNotADirectoryError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "IsDirectory")
      throw new SandboxFilesystemIsADirectoryError(
        `${payload.message}: ${remotePath}`,
      );
    if (payload.error_kind === "PermissionDenied")
      throw new SandboxFilesystemPermissionError(
        `${payload.message}: ${remotePath}`,
      );
    throw new SandboxFilesystemError(payload.message);
  }
  const text = stderrToText(stderr);
  if (text) logger.debug(`Unstructured modal-sandbox-fs-tools stderr: ${text}`);
  throw new SandboxFilesystemError(
    `Operation on '${remotePath}' failed with exit code ${returnCode}`,
  );
}

export function makeWriteFileCommand(remotePath: string): string {
  return JSON.stringify({ WriteFile: { path: remotePath } });
}

const SUPPORT_ERROR_CODE_RE = /Error code:\s*([A-Z0-9]{8})/;

export function extractSupportErrorCode(err: unknown): string | null {
  const message = err instanceof Error ? err.message : String(err);
  const match = SUPPORT_ERROR_CODE_RE.exec(message);
  return match ? match[1] : null;
}

function describeError(err: unknown): string {
  if (err instanceof Error) return `${err.name}: ${err.message}`;
  return String(err);
}

/** NotFound and Service Errors */
function isSandboxUnavailableError(err: unknown): boolean {
  if (err instanceof NotFoundError) return true;
  if (err instanceof ClientError) {
    return (
      err.code === Status.NOT_FOUND ||
      err.code === Status.CANCELLED ||
      err.code === Status.UNKNOWN ||
      err.code === Status.DEADLINE_EXCEEDED ||
      err.code === Status.UNAVAILABLE
    );
  }
  return false;
}

function translateExecSandboxUnavailableError(
  operation: string,
  remotePath: string,
  err: unknown,
): Error {
  logger.debug(
    `Sandbox filesystem control-plane error for operation=${operation}, path=${remotePath}: ${describeError(err)}`,
  );
  return new NotFoundError(
    "The Sandbox is unavailable. This Sandbox may have already shut down.",
  );
}

/**
 * Translate an unexpected exec-level error into a generic user-facing error.
 *
 * This is less than ideal — it discards the original exception and returns a
 * generic message — but it's necessary to avoid surfacing "call to exec()
 * failed"-style messages to users who shouldn't be aware that the filesystem
 * API is implemented in terms of Sandbox.exec().
 */
export function translateExecUnexpectedError(
  operation: string,
  remotePath: string,
  err: unknown,
): Error {
  const code = extractSupportErrorCode(err);
  logger.debug(
    `Unexpected sandbox filesystem exec error for operation=${operation}, path=${remotePath}: ${describeError(err)}`,
  );
  const supportSuffix = code
    ? `please contact support@modal.com (Error code: ${code})`
    : "please contact support@modal.com";
  return new SandboxFilesystemError(
    `An unexpected error occurred, ${supportSuffix}`,
  );
}

/** Translate exec-level exceptions into user-facing errors. */
export async function translateExecErrors<T>(
  operation: string,
  remotePath: string,
  fn: () => Promise<T>,
): Promise<T> {
  try {
    return await fn();
  } catch (err) {
    if (isSandboxUnavailableError(err)) {
      throw translateExecSandboxUnavailableError(operation, remotePath, err);
    }
    if (err instanceof ClientError) {
      throw translateExecUnexpectedError(operation, remotePath, err);
    }
    throw err;
  }
}
