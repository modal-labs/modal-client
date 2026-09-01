// Copyright Modal Labs 2026
import { expect, test } from "vitest";

import {
  DEFAULT_OUTPUT as OUTPUT,
  SANDBOX_ID,
  startFakeWorker,
} from "../test-support/fake_worker";

/**
 * How `ModalReadStream` turns stdio chunks into text and bytes.
 *
 * These enter at `sandbox.exec()` against a local worker, so chunks arrive from
 * real protobuf decoding. That matters here: a decoded byte field is a view
 * into the wire buffer rather than a buffer of its own, and a reader that looks
 * past the view sees the framing bytes around it. A hand-built chunk hides that.
 */

const encoder = new TextEncoder();

test("binary mode readText() returns the output and nothing around it", async () => {
  const w = await startFakeWorker();
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"], { mode: "binary" });

    expect(await process.stdout.readText()).toBe(OUTPUT);
  } finally {
    await w.shutdown();
  }
});

test("binary mode readText() decodes a character split across chunks", async () => {
  // A pound sign, split down the middle of its two bytes.
  const encoded = encoder.encode("a£b");
  const w = await startFakeWorker([
    encoded.subarray(0, 2),
    encoded.subarray(2),
  ]);
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"], { mode: "binary" });

    expect(await process.stdout.readText()).toBe("a£b");
  } finally {
    await w.shutdown();
  }
});

test("text mode decodes a character split across chunks", async () => {
  const encoded = encoder.encode("a£b");
  const w = await startFakeWorker([
    encoded.subarray(0, 2),
    encoded.subarray(2),
  ]);
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"]);

    expect(await process.stdout.readText()).toBe("a£b");
  } finally {
    await w.shutdown();
  }
});

test("binary mode hands back the bytes unchanged", async () => {
  const w = await startFakeWorker();
  try {
    const sandbox = await w.mockClient.sandboxes.fromId(SANDBOX_ID);
    const process = await sandbox.exec(["echo", "hi"], { mode: "binary" });

    expect(await process.stdout.readBytes()).toEqual(encoder.encode(OUTPUT));
  } finally {
    await w.shutdown();
  }
});
