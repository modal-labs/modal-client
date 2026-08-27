import { describe, expect, test, vi } from "vitest";

import { pLimit } from "./plimit";

function deferred(): { promise: Promise<void>; resolve(): void } {
  let resolve = () => {};
  const promise = new Promise<void>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

describe("pLimit", () => {
  test("runs queued functions in FIFO order with bounded concurrency", async () => {
    const limit = pLimit(2);
    const gates = Array.from({ length: 4 }, deferred);
    const started: number[] = [];

    const tasks = gates.map((gate, index) =>
      limit(async () => {
        started.push(index);
        await gate.promise;
        return index;
      }),
    );

    await vi.waitFor(() => expect(started).toEqual([0, 1]));
    expect(limit.activeCount).toBe(2);
    expect(limit.pendingCount).toBe(2);

    gates[1].resolve();
    await vi.waitFor(() => expect(started).toEqual([0, 1, 2]));

    gates[0].resolve();
    await vi.waitFor(() => expect(started).toEqual([0, 1, 2, 3]));

    gates[2].resolve();
    gates[3].resolve();
    await expect(Promise.all(tasks)).resolves.toEqual([0, 1, 2, 3]);
    expect(limit.activeCount).toBe(0);
    expect(limit.pendingCount).toBe(0);
  });

  test("starts more queued work when concurrency increases", async () => {
    const limit = pLimit(1);
    const firstGate = deferred();
    const started: number[] = [];

    const first = limit(async () => {
      started.push(1);
      await firstGate.promise;
    });
    const second = limit(async () => {
      started.push(2);
    });

    await vi.waitFor(() => expect(started).toEqual([1]));
    limit.concurrency = 2;
    await vi.waitFor(() => expect(started).toEqual([1, 2]));

    firstGate.resolve();
    await Promise.all([first, second]);
  });

  test("does not start pending work after clearing the queue", async () => {
    const limit = pLimit(1);
    const gate = deferred();
    const running = limit(() => gate.promise);
    let pendingStarted = false;
    void limit(() => {
      pendingStarted = true;
    });

    expect(limit.pendingCount).toBe(1);
    limit.clearQueue();
    expect(limit.pendingCount).toBe(0);

    gate.resolve();
    await running;
    await new Promise<void>(queueMicrotask);
    expect(pendingStarted).toBe(false);
  });

  test("maps values with limited concurrency", async () => {
    const limit = pLimit(2);
    await expect(
      limit.map([1, 2, 3], async (value, index) => value + index),
    ).resolves.toEqual([1, 3, 5]);
  });

  test.each([0, -1, 1.5, Number.NaN])(
    "rejects concurrency %s",
    (concurrency) => {
      expect(() => pLimit(concurrency)).toThrow(
        "Expected concurrency to be a number from 1 and up",
      );
    },
  );
});
