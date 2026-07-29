import { expect, test } from "vitest";
import { FunctionCallInvocationType } from "../proto/modal_proto/api";
import { shouldUpload } from "../src/function";

const maxObjectSize = 2 * 1024 * 1024; // 2 MiB
const maxAsyncObjectSize = 8 * 1024; // 8 KiB

test("shouldUpload sync uses the sync threshold", () => {
  const sync = FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC;
  // Below the sync threshold, even if above the async threshold.
  expect(
    shouldUpload(
      maxAsyncObjectSize + 1,
      maxObjectSize,
      maxAsyncObjectSize,
      sync,
    ),
  ).toBe(false);
  // Exactly at the threshold should not upload (strict greater-than).
  expect(
    shouldUpload(maxObjectSize, maxObjectSize, maxAsyncObjectSize, sync),
  ).toBe(false);
  // Above the sync threshold should upload.
  expect(
    shouldUpload(maxObjectSize + 1, maxObjectSize, maxAsyncObjectSize, sync),
  ).toBe(true);
});

test("shouldUpload async uses the smaller async threshold", () => {
  const async_ = FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_ASYNC;
  // Below the async threshold should not upload.
  expect(
    shouldUpload(maxAsyncObjectSize, maxObjectSize, maxAsyncObjectSize, async_),
  ).toBe(false);
  // Above the async threshold but below the sync threshold should still upload.
  expect(
    shouldUpload(
      maxAsyncObjectSize + 1,
      maxObjectSize,
      maxAsyncObjectSize,
      async_,
    ),
  ).toBe(true);
  // Above the sync threshold should upload.
  expect(
    shouldUpload(maxObjectSize + 1, maxObjectSize, maxAsyncObjectSize, async_),
  ).toBe(true);
});
