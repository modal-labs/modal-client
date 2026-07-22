/** An operation exceeds the allowed time limit. */
export class TimeoutError extends Error {
  constructor(message: string = "Operation timed out") {
    super(message);
    this.name = "TimeoutError";
  }
}

/** Function execution exceeds the allowed time limit. */
export class FunctionTimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FunctionTimeoutError";
  }
}

/** An error on the Modal server, or a Python exception. */
export class RemoteError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RemoteError";
  }
}

/** Something unexpected happened during runtime. */
export class ExecutionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ExecutionError";
  }
}

/** A retryable internal error from Modal. */
export class InternalFailure extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InternalFailure";
  }
}

/** Some resource was not found. */
export class NotFoundError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NotFoundError";
  }
}

/** A resource already exists. */
export class AlreadyExistsError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AlreadyExistsError";
  }
}

/** The current state of a resource conflicts with the requested operation. */
export class ConflictError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConflictError";
  }
}

/** A request or other operation was invalid. */
export class InvalidError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidError";
  }
}

/** The Queue is empty. */
export class QueueEmptyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "QueueEmptyError";
  }
}

/** The Queue is full. */
export class QueueFullError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "QueueFullError";
  }
}

/** Errors from invalid Sandbox FileSystem operations. */
export class SandboxFilesystemError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SandboxFilesystemError";
  }
}

/** A directory was expected to be empty but is not. */
export class SandboxFilesystemDirectoryNotEmptyError extends SandboxFilesystemError {
  constructor(message: string) {
    super(message);
    this.name = "SandboxFilesystemDirectoryNotEmptyError";
  }
}

/** A file exceeds the maximum allowed size for a read operation. */
export class SandboxFilesystemFileTooLargeError extends SandboxFilesystemError {
  constructor(message: string) {
    super(message);
    this.name = "SandboxFilesystemFileTooLargeError";
  }
}

/** A file operation was attempted on a path that resolves to a directory. */
export class SandboxFilesystemIsADirectoryError extends SandboxFilesystemError {
  constructor(message: string) {
    super(message);
    this.name = "SandboxFilesystemIsADirectoryError";
  }
}

/** A directory operation encountered a path component that is not a directory. */
export class SandboxFilesystemNotADirectoryError extends SandboxFilesystemError {
  constructor(message: string) {
    super(message);
    this.name = "SandboxFilesystemNotADirectoryError";
  }
}

/** A file or directory is not found. */
export class SandboxFilesystemNotFoundError extends SandboxFilesystemError {
  constructor(message: string) {
    super(message);
    this.name = "SandboxFilesystemNotFoundError";
  }
}

/** A path already exists and the operation requires it to be absent. */
export class SandboxFilesystemPathAlreadyExistsError extends SandboxFilesystemError {
  constructor(message: string) {
    super(message);
    this.name = "SandboxFilesystemPathAlreadyExistsError";
  }
}

/** Permission is denied for a file operation. */
export class SandboxFilesystemPermissionError extends SandboxFilesystemError {
  constructor(message: string) {
    super(message);
    this.name = "SandboxFilesystemPermissionError";
  }
}

/** Sandbox operations that exceed the allowed time limit. */
export class SandboxTimeoutError extends Error {
  constructor(message: string = "Sandbox operation timed out") {
    super(message);
    this.name = "SandboxTimeoutError";
  }
}

/** Thrown when attempting operations on a detached Sandbox. */
export class ClientClosedError extends Error {
  constructor(
    message: string = "Unable to perform operation on a detached sandbox",
  ) {
    super(message);
    this.name = "ClientClosedError";
  }
}
