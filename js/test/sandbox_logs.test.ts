import { describe, expect, test } from "vitest";
import { ModalClient, Sandbox } from "modal";
import { ClientError, Status } from "nice-grpc";
import { setTimeout as sleep } from "timers/promises";

function makeClient(cpClient: any): ModalClient {
  return new ModalClient({
    cpClient: cpClient as any,
    tokenId: "test-id",
    tokenSecret: "test-secret",
  });
}

function textItem(data: string) {
  return { data };
}

function batch(entryId: string, items: Array<{ data: string }>, eof = false) {
  return { entryId, items, eof };
}

describe("SandboxGetLogs lazy and retry behavior", () => {
  test("testSandboxGetLogsNotCalledUntilStdoutIsAccessed", async () => {
    let calls = 0;
    const cpClient = {
      // Return an empty, immediate EOF stream
      async *sandboxGetLogs(_req: any) {
        calls++;
        yield batch("1-0", [], true);
      },
    };
    const client = makeClient(cpClient);
    const sb = new Sandbox(client, "sb-123");

    // Constructor should not trigger logs
    expect(calls).toBe(0);

    // Accessing stdout doesn't start pulling until read
    const reader = sb.stdout.getReader();
    await reader.read();
    reader.releaseLock();

    expect(calls).toBe(1);
  });

  test("testSandboxGetLogsNonRetryablePropagates", async () => {
    let calls = 0;
    const cpClient = {
      sandboxGetLogs(_req: any) {
        calls++;
        throw new ClientError(
          "/modal.client.ModalClient/SandboxGetLogs",
          Status.PERMISSION_DENIED,
          "denied",
        );
      },
    };
    const client = makeClient(cpClient);
    const sb = new Sandbox(client, "sb-denied");

    let caught: unknown = null;
    try {
      await sb.stdout.readText();
    } catch (e) {
      caught = e;
    }
    expect(calls).toBe(1);
    expect(caught).toBeInstanceOf(ClientError);
    expect((caught as ClientError).code).toBe(Status.PERMISSION_DENIED);
  });

  test("testSandboxGetLogsRetryExhaustion", async () => {
    let calls = 0;
    const cpClient = {
      sandboxGetLogs(_req: any) {
        calls++;
        throw new ClientError(
          "/modal.client.ModalClient/SandboxGetLogs",
          Status.UNAVAILABLE,
          "always-unavailable",
        );
      },
    };
    const client = makeClient(cpClient);
    const sb = new Sandbox(client, "sb-retry-exhaust");

    let caught: unknown = null;
    try {
      await sb.stdout.readText();
    } catch (e) {
      caught = e;
    }
    // 1 initial attempt + 10 retries = 11 (if defaults unchanged)
    expect(calls).toBeGreaterThanOrEqual(11);
    expect(caught).toBeInstanceOf(ClientError);
    expect((caught as ClientError).code).toBe(Status.UNAVAILABLE);
  });

  test("testSandboxGetLogsResumesWithLastEntryIdAfterRetry", async () => {
    const seenLastEntryIds: string[] = [];
    let attempt = 0;
    const cpClient = {
      sandboxGetLogs(req: any) {
        seenLastEntryIds.push(req.lastEntryId);
        attempt++;
        if (attempt === 1) {
          return (async function* () {
            // Yield a batch that sets lastEntryId, then throw to trigger retry
            yield batch("1-9", [textItem("part")], false);
            throw new ClientError(
              "/modal.client.ModalClient/SandboxGetLogs",
              Status.UNAVAILABLE,
              "transient",
            );
          })();
        }
        // Second attempt should see lastEntryId "1-9" and then complete
        return (async function* () {
          yield batch("1-10", [], true);
        })();
      },
    };
    const client = makeClient(cpClient);
    const sb = new Sandbox(client, "sb-last-entry");

    const out = await sb.stdout.readText();
    expect(out).toContain("part");
    expect(seenLastEntryIds.length).toBeGreaterThanOrEqual(2);
    expect(seenLastEntryIds[0]).toBe("0-0");
    expect(seenLastEntryIds[1]).toBe("1-9");
  });
  test("testSandboxGetLogsNotCalledUntilStderrIsAccessed", async () => {
    let calls = 0;
    const cpClient = {
      async *sandboxGetLogs(_req: any) {
        calls++;
        yield batch("1-0", [], true);
      },
    };
    const client = makeClient(cpClient);
    const sb = new Sandbox(client, "sb-456");

    expect(calls).toBe(0);

    const reader = sb.stderr.getReader();
    await reader.read();
    reader.releaseLock();

    expect(calls).toBe(1);
  });

  test("testSandboxGetLogsRetriesAfterDelayOnRetriableError", async () => {
    const callTimes: number[] = [];
    let attempt = 0;
    const cpClient = {
      sandboxGetLogs(_req: any) {
        callTimes.push(Date.now());
        attempt++;
        if (attempt === 1) {
          // First attempt: retryable error
          throw new ClientError(
            "/modal.client.ModalClient/SandboxGetLogs",
            Status.UNAVAILABLE,
            "transient",
          );
        }
        // Second attempt: immediate EOF
        return (async function* () {
          yield batch("1-0", [], true);
        })();
      },
    };
    const client = makeClient(cpClient);
    const sb = new Sandbox(client, "sb-789");

    const reader = sb.stdout.getReader();
    await reader.read();
    reader.releaseLock();

    expect(callTimes.length).toBeGreaterThanOrEqual(2);
    const delta = callTimes[1] - callTimes[0];
    // Expect at least ~10ms backoff; keep upper bound generous for CI variance
    expect(delta).toBeGreaterThanOrEqual(8);
    expect(delta).toBeLessThan(500);
  });

  test("testSandboxGetLogsRetryDelayResetsAfterSuccessfulRead", async () => {
    const callTimes: number[] = [];
    let attempt = 0;
    const cpClient = {
      sandboxGetLogs(_req: any) {
        callTimes.push(Date.now());
        attempt++;
        if (attempt === 1) {
          // First: retryable error (will set next delay to 20ms internally)
          throw new ClientError(
            "/modal.client.ModalClient/SandboxGetLogs",
            Status.UNAVAILABLE,
            "transient-1",
          );
        } else if (attempt === 2) {
          // Second: successful read (resets delay back to initial)
          return (async function* () {
            yield batch("1-0", [textItem("hi")], false);
            // end of stream without eof -> outer loop will re-enter
          })();
        } else if (attempt === 3) {
          // Third: retryable error; delay should be reset to initial (~10ms)
          throw new ClientError(
            "/modal.client.ModalClient/SandboxGetLogs",
            Status.UNAVAILABLE,
            "transient-2",
          );
        }
        // Fourth: complete
        return (async function* () {
          yield batch("1-1", [], true);
        })();
      },
    };
    const client = makeClient(cpClient);
    const sb = new Sandbox(client, "sb-000");

    const reader = sb.stdout.getReader();
    // First read triggers attempt 1 (error) and attempt 2 (success with one chunk)
    await reader.read();
    // Second read drives the generator to re-enter, causing attempt 3 (error) and attempt 4 (EOF)
    await reader.read();
    reader.releaseLock();

    // We expect at least 4 invocations
    expect(callTimes.length).toBeGreaterThanOrEqual(4);
    const deltaAfterReset = callTimes[3] - callTimes[2];
    expect(deltaAfterReset).toBeGreaterThanOrEqual(8);
    expect(deltaAfterReset).toBeLessThan(500);
  });

  test("testCancellingStdoutIteratorClosesIterator", async () => {
    let cancelled = false;

    const cpClient = {
      sandboxGetLogs(_req: any, _opts?: { signal?: AbortSignal }) {
        return (async function* () {
          try {
            yield batch("1-0", [textItem("hello")], false);
            // Simulate server keeping the connection open with no more data.
            await new Promise(() => {});
          } finally {
            cancelled = true;
          }
        })();
      },
    };
    const client = makeClient(cpClient);
    const sb = new Sandbox(client, "sb-cancel");

    const reader = sb.stdout.getReader();
    const first = await reader.read(); // pull first chunk
    expect(first.done).toBe(false);
    // Cancel consumption
    await reader.cancel();

    // Give the generator a moment to run its finally block
    await sleep(20);
    expect(cancelled).toBe(true);
  });

  test("testCancelStdoutStopsPollingWithEmptyBatches", async () => {
    let batchesConsumed = 0;

    const cpClient = {
      sandboxGetLogs(_req: any, _opts?: { signal?: AbortSignal }) {
        return (async function* () {
          yield batch("1-0", [textItem("hello")], false);
          for (let i = 0; i < 100; i++) {
            await sleep(2);
            batchesConsumed++;
            yield batch("1-0", [], false);
          }
          yield batch("1-0", [], true);
        })();
      },
    };
    const client = makeClient(cpClient);
    const sb = new Sandbox(client, "sb-empty-cancel");

    const reader = sb.stdout.getReader();
    const first = await reader.read();
    expect(first.done).toBe(false);
    expect(first.value).toBe("hello");

    await Promise.race([reader.cancel(), sleep(50)]);
    const countAtCancel = batchesConsumed;
    expect(countAtCancel).toBeLessThan(100);

    await sleep(100);

    expect(batchesConsumed).toBe(countAtCancel);
  });
});
