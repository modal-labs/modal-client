import { ClientError, Status } from "nice-grpc";
import { expect, test } from "vitest";

import { NotFoundError, SandboxFilesystemError } from "./errors";
import {
  makeListFilesCommand,
  makeReadFileCommand,
  makeStatCommand,
  makeWatchCommand,
  makeWriteFileCommand,
  translateExecErrors,
  translateExecUnexpectedError,
  tryParseErrorPayload,
} from "./sandbox_fs_utils";

test("tryParseErrorPayload returns payload for valid JSON", () => {
  const stderr = JSON.stringify({
    error_kind: "NotFound",
    message: "file not found",
  });
  expect(tryParseErrorPayload(stderr)).toEqual({
    error_kind: "NotFound",
    message: "file not found",
    detail: "",
  });
});

test("tryParseErrorPayload accepts bytes", () => {
  const stderr = new TextEncoder().encode(
    JSON.stringify({
      error_kind: "PermissionDenied",
      message: "access denied",
    }),
  );
  expect(tryParseErrorPayload(stderr)).toEqual({
    error_kind: "PermissionDenied",
    message: "access denied",
    detail: "",
  });
});

test("tryParseErrorPayload returns null for empty stderr", () => {
  expect(tryParseErrorPayload("")).toBeNull();
  expect(tryParseErrorPayload(new Uint8Array())).toBeNull();
});

test("tryParseErrorPayload returns null for non-JSON", () => {
  expect(tryParseErrorPayload("not json at all")).toBeNull();
});

test("tryParseErrorPayload returns null for non-object JSON", () => {
  expect(tryParseErrorPayload(JSON.stringify([1, 2, 3]))).toBeNull();
});

test("tryParseErrorPayload returns null for missing error_kind", () => {
  expect(tryParseErrorPayload(JSON.stringify({ message: "oops" }))).toBeNull();
});

test("tryParseErrorPayload returns null for non-string error_kind", () => {
  expect(
    tryParseErrorPayload(JSON.stringify({ error_kind: 42, message: "oops" })),
  ).toBeNull();
});

test("tryParseErrorPayload returns null for missing message", () => {
  expect(
    tryParseErrorPayload(JSON.stringify({ error_kind: "NotFound" })),
  ).toBeNull();
});

test("tryParseErrorPayload returns null for non-string message", () => {
  expect(
    tryParseErrorPayload(
      JSON.stringify({ error_kind: "NotFound", message: 123 }),
    ),
  ).toBeNull();
});

test("tryParseErrorPayload returns null for blank message", () => {
  expect(
    tryParseErrorPayload(
      JSON.stringify({ error_kind: "NotFound", message: "  " }),
    ),
  ).toBeNull();
});

test("tryParseErrorPayload includes detail when present", () => {
  const stderr = JSON.stringify({
    error_kind: "Io",
    message: "I/O error",
    detail: "No such file or directory (os error 2)",
  });
  expect(tryParseErrorPayload(stderr)).toEqual({
    error_kind: "Io",
    message: "I/O error",
    detail: "No such file or directory (os error 2)",
  });
});

test("tryParseErrorPayload ignores non-string detail", () => {
  const stderr = JSON.stringify({
    error_kind: "Io",
    message: "I/O error",
    detail: 42,
  });
  expect(tryParseErrorPayload(stderr)).toEqual({
    error_kind: "Io",
    message: "I/O error",
    detail: "",
  });
});

test("makeListFilesCommand produces correct JSON", () => {
  expect(JSON.parse(makeListFilesCommand("/tmp/mydir"))).toEqual({
    ListFiles: { path: "/tmp/mydir" },
  });
});

test("makeListFilesCommand handles paths with special characters", () => {
  expect(JSON.parse(makeListFilesCommand("/tmp/my dir/with spaces"))).toEqual({
    ListFiles: { path: "/tmp/my dir/with spaces" },
  });
});

test("makeStatCommand produces correct JSON", () => {
  expect(JSON.parse(makeStatCommand("/tmp/file.txt"))).toEqual({
    Stat: { path: "/tmp/file.txt" },
  });
});

test("makeStatCommand handles paths with special characters", () => {
  expect(
    JSON.parse(makeStatCommand("/tmp/my dir/with spaces/file.txt")),
  ).toEqual({
    Stat: { path: "/tmp/my dir/with spaces/file.txt" },
  });
});

test("makeReadFileCommand produces correct JSON", () => {
  expect(JSON.parse(makeReadFileCommand("/tmp/file.txt"))).toEqual({
    ReadFile: { path: "/tmp/file.txt" },
  });
});

test("makeWatchCommand produces correct JSON", () => {
  expect(
    JSON.parse(
      makeWatchCommand("/tmp/dir", {
        recursive: false,
        filter: null,
        timeoutMs: null,
      }),
    ),
  ).toEqual({
    Watch: {
      path: "/tmp/dir",
      recursive: false,
      filter: null,
      timeout_secs: null,
    },
  });
});

test("makeWatchCommand serializes filter and converts timeoutMs to seconds", () => {
  expect(
    JSON.parse(
      makeWatchCommand("/tmp/dir", {
        recursive: true,
        filter: ["Create", "Remove"],
        timeoutMs: 30_000,
      }),
    ),
  ).toEqual({
    Watch: {
      path: "/tmp/dir",
      recursive: true,
      filter: ["Create", "Remove"],
      timeout_secs: 30,
    },
  });
});

test("makeWatchCommand truncates sub-second timeoutMs", () => {
  expect(
    JSON.parse(
      makeWatchCommand("/tmp/dir", {
        recursive: false,
        filter: null,
        timeoutMs: 1900,
      }),
    ),
  ).toEqual({
    Watch: {
      path: "/tmp/dir",
      recursive: false,
      filter: null,
      timeout_secs: 1,
    },
  });
});

test("makeWriteFileCommand produces correct JSON", () => {
  expect(JSON.parse(makeWriteFileCommand("/tmp/file.txt"))).toEqual({
    WriteFile: { path: "/tmp/file.txt" },
  });
});

test("translateExecUnexpectedError includes backend error code", () => {
  const err = translateExecUnexpectedError(
    "readBytes",
    "/tmp/missing.txt",
    new ClientError(
      "/svc/Method",
      Status.INTERNAL,
      "Failed to start exec command (Error code: ABCD1234)",
    ),
  );
  expect(err).toBeInstanceOf(SandboxFilesystemError);
  expect(err.message).toMatch(/Error code: ABCD1234/);
});

test("translateExecErrors converts sandbox-unavailable errors", async () => {
  await expect(
    translateExecErrors("readBytes", "/tmp/file.txt", async () => {
      throw new NotFoundError("sandbox not found");
    }),
  ).rejects.toThrow(/Sandbox is unavailable/);
});

test("translateExecErrors converts general Modal errors", async () => {
  await expect(
    translateExecErrors("readBytes", "/tmp/file.txt", async () => {
      throw new ClientError("/svc/Method", Status.INTERNAL, "something broke");
    }),
  ).rejects.toThrow(/unexpected error/);
});
