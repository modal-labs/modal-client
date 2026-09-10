import { expect, test } from "vitest";
import { createServer } from "nice-grpc";
import {
  ModalClientDefinition,
  type SandboxGetLogsRequest,
  TaskLogsBatch,
} from "../proto/modal_proto/api";
import { ModalClient } from "modal";

/**
 * A V1 Sandbox's output comes from the control plane, not from the worker, so
 * none of the Sandbox connection's own release machinery covers it: the stream
 * is multiplexed onto the long-lived control-plane connection, which nothing
 * closes.
 *
 * The control plane here is a real server on a real connection on purpose. A
 * forgotten stream costs no socket and no handle of its own, so what a leak
 * keeps alive is the call itself — which only a server that counts its open
 * calls can see. A mock standing in for the stream has no call to leak, and a
 * test written against one passes whether or not the bug is there.
 */

/** V1-shaped, so the Sandbox reads its output through the control plane. */
const SANDBOX_ID = "sb-nGEijt9WbBMlGrsPH9FOaC";
const IDLE_TIMEOUT_MS = 400;
const LINE = "hello from the sandbox\n";

test("a pending output read stays active beyond the idle timeout", async () => {
  let sendOutput!: () => void;
  const outputGate = new Promise<void>((resolve) => {
    sendOutput = resolve;
  });
  const cp = await startFakeControlPlane(false, undefined, outputGate);
  try {
    const sandbox = await cp.client.sandboxes.fromId(SANDBOX_ID);
    const reader = sandbox.stdout.getReader();
    const pending = reader.read();
    expect(await waitFor(() => cp.live() === 1, 5000)).toBe(true);
    await new Promise((resolve) =>
      globalThis.setTimeout(resolve, 2 * IDLE_TIMEOUT_MS),
    );
    expect(cp.live()).toBe(1);
    expect(cp.started()).toBe(1);
    sendOutput();
    expect((await pending).value).toBe(LINE);
    expect(await waitFor(() => cp.live() === 0, 6 * IDLE_TIMEOUT_MS)).toBe(
      true,
    );
    await reader.cancel();
  } finally {
    sendOutput();
    await cp.shutdown();
  }
});

function mockJwt(): string {
  const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  const exp = Math.floor(Date.now() / 1000) + 3600;
  return `${header}.${btoa(JSON.stringify({ exp }))}.signature`;
}

/**
 * A control plane that serves one batch per call and then holds the call open.
 * With `silent`, it sends nothing at all, standing in for a Sandbox that has
 * not printed anything yet.
 */
async function startFakeControlPlane(
  silent = false,
  authGate?: Promise<void>,
  outputGate?: Promise<void>,
) {
  let open = 0;
  let started = 0;
  const requestedCursors: string[] = [];
  let authenticating = false;

  const unimplemented: Record<string, unknown> = {};
  for (const methodName of Object.keys(ModalClientDefinition.methods)) {
    unimplemented[methodName] = () => {
      throw new Error(`${methodName} is not implemented in this control plane`);
    };
  }

  const server = createServer();
  server.add(ModalClientDefinition, {
    ...unimplemented,
    async authTokenGet() {
      authenticating = true;
      await authGate;
      return { token: mockJwt() };
    },
    async *sandboxGetLogs(
      request: SandboxGetLogsRequest,
      context: { signal: AbortSignal },
    ) {
      started++;
      requestedCursors.push(request.lastEntryId);
      open++;
      try {
        await outputGate;
        if (!silent) {
          const firstBatch = request.lastEntryId === "0-0";
          if (!firstBatch && request.lastEntryId !== "1-0") {
            throw new Error(`unexpected log cursor: ${request.lastEntryId}`);
          }
          yield TaskLogsBatch.create({
            entryId: firstBatch ? "1-0" : "2-0",
            items: firstBatch
              ? [{ data: LINE }, { data: LINE }]
              : [{ data: "subsequent output\n" }],
            eof: false,
          });
        }
        // Held open, as the control plane does while it waits for a Sandbox to
        // print something more, until the caller goes away.
        await new Promise<void>((resolve) => {
          context.signal.addEventListener("abort", () => resolve());
        });
      } finally {
        open--;
      }
    },
  } as never);

  const port = await server.listen("127.0.0.1:0");
  process.env["MODAL_SERVER_URL"] = `http://127.0.0.1:${port}`;
  const client = new ModalClient({
    tokenId: "test-token-id",
    tokenSecret: "test-token-secret",
  });

  client.profile.sandboxChannelIdleTimeoutMs = IDLE_TIMEOUT_MS;

  return {
    client,
    live: () => open,
    started: () => started,
    requestedCursors,
    authenticating: () => authenticating,
    shutdown: async () => {
      client.close();
      server.forceShutdown();
    },
  };
}

async function waitFor(
  condition: () => boolean,
  timeoutMs: number,
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (condition()) return true;
    await new Promise((resolve) => globalThis.setTimeout(resolve, 25));
  }
  return condition();
}

test("cancelling a read that is waiting for output returns instead of hanging", async () => {
  const cp = await startFakeControlPlane(true);
  try {
    const sandbox = await cp.client.sandboxes.fromId(SANDBOX_ID);
    const reader = sandbox.stdout.getReader();

    // Nothing has been printed, so this read is parked on the wire.
    const pending = reader.read().catch(() => undefined);
    expect(await waitFor(() => cp.live() === 1, 5_000)).toBe(true);

    // Giving up has to reach the call itself. Unwinding the iterator cannot do
    // it on its own here: that waits for the read to settle, and the read is
    // waiting for output that is not coming.
    const cancelled = reader.cancel().then(() => "returned");
    const outcome = await Promise.race([
      cancelled,
      new Promise((resolve) =>
        globalThis.setTimeout(() => resolve("still waiting"), 10_000),
      ),
    ]);
    expect(outcome).toBe("returned");
    await pending;
  } finally {
    await cp.shutdown();
  }
});

test.each([false, true])(
  "V1 output remains readable after detach (partly read: %s)",
  async (partlyRead) => {
    const cp = await startFakeControlPlane();
    try {
      cp.client.profile.sandboxChannelIdleTimeoutMs = 0;
      const sandbox = await cp.client.sandboxes.fromId(SANDBOX_ID);
      let reader;
      if (partlyRead) {
        reader = sandbox.stdout.getReader();
        expect((await reader.read()).value).toBe(LINE);
      }
      sandbox.detach();
      reader ??= sandbox.stdout.getReader();
      expect((await reader.read()).value).toBe(LINE);
      expect(cp.live()).toBe(1);
      expect(cp.started()).toBe(1);
      await reader.cancel();
      expect(await waitFor(() => cp.live() === 0, 5000)).toBe(true);
    } finally {
      await cp.shutdown();
    }
  },
);

test("explicit cancellation during authentication opens no log call", async () => {
  let finishAuth!: () => void;
  const gate = new Promise<void>((resolve) => {
    finishAuth = resolve;
  });
  const cp = await startFakeControlPlane(true, gate);
  try {
    const sandbox = await cp.client.sandboxes.fromId(SANDBOX_ID);
    const reader = sandbox.stdout.getReader();
    const pending = reader.read();
    expect(await waitFor(cp.authenticating, 5000)).toBe(true);
    const cancelled = reader.cancel();
    finishAuth();
    await cancelled;
    await pending;
    expect(cp.started()).toBe(0);
  } finally {
    finishAuth();
    await cp.shutdown();
  }
});

test("reading part of a Sandbox's output and forgetting it releases the stream", async () => {
  const cp = await startFakeControlPlane();
  try {
    const sandbox = await cp.client.sandboxes.fromId(SANDBOX_ID);
    const reader = sandbox.stdout.getReader();
    expect((await reader.read()).value).toBe(LINE);
    expect(cp.live()).toBe(1);

    // Walk away: stop reading, without cancel and without detach.
    expect(await waitFor(() => cp.live() === 0, 6 * IDLE_TIMEOUT_MS)).toBe(
      true,
    );
  } finally {
    await cp.shutdown();
  }
});

test("a released output stream reopens when it is read again", async () => {
  const cp = await startFakeControlPlane();
  try {
    const sandbox = await cp.client.sandboxes.fromId(SANDBOX_ID);
    const reader = sandbox.stdout.getReader();
    expect((await reader.read()).value).toBe(LINE);

    expect(await waitFor(() => cp.live() === 0, 6 * IDLE_TIMEOUT_MS)).toBe(
      true,
    );

    // Finish the buffered batch before reopening the released call.
    expect((await reader.read()).value).toBe(LINE);
    expect((await reader.read()).value).toBe("subsequent output\n");
    expect(cp.started()).toBe(2);
    expect(cp.requestedCursors).toEqual(["0-0", "1-0"]);
    await reader.cancel();
  } finally {
    await cp.shutdown();
  }
});

/** Puts several messages in one batch, which all resume from one entry. */
async function startMultiItemControlPlane() {
  const requestedCursors: string[] = [];

  const unimplemented: Record<string, unknown> = {};
  for (const methodName of Object.keys(ModalClientDefinition.methods)) {
    unimplemented[methodName] = () => {
      throw new Error(`${methodName} is not implemented in this control plane`);
    };
  }

  const server = createServer();
  server.add(ModalClientDefinition, {
    ...unimplemented,
    async authTokenGet() {
      return { token: mockJwt() };
    },
    async *sandboxGetLogs(
      request: SandboxGetLogsRequest,
      context: { signal: AbortSignal },
    ) {
      requestedCursors.push(request.lastEntryId);
      const firstBatch = request.lastEntryId === "0-0";
      if (!firstBatch && request.lastEntryId !== "1-0") {
        throw new Error(`unexpected log cursor: ${request.lastEntryId}`);
      }
      const lines = firstBatch ? ["a\n", "b\n", "c\n"] : ["d\n"];
      yield TaskLogsBatch.create({
        entryId: firstBatch ? "1-0" : "2-0",
        items: lines.map((data) => ({ data })),
        eof: false,
      });
      await new Promise<void>((resolve) => {
        context.signal.addEventListener("abort", () => resolve());
      });
    },
  } as never);

  const port = await server.listen("127.0.0.1:0");
  process.env["MODAL_SERVER_URL"] = `http://127.0.0.1:${port}`;
  const client = new ModalClient({
    tokenId: "test-token-id",
    tokenSecret: "test-token-secret",
  });
  client.profile.sandboxChannelIdleTimeoutMs = IDLE_TIMEOUT_MS;

  return {
    client,
    requestedCursors,
    shutdown: async () => {
      client.close();
      server.forceShutdown();
    },
  };
}

test("a release part way through a batch loses nothing", async () => {
  const cp = await startMultiItemControlPlane();
  try {
    const sandbox = await cp.client.sandboxes.fromId(SANDBOX_ID);
    const reader = sandbox.stdout.getReader();
    expect((await reader.read()).value).toBe("a\n");

    // Pause past the timeout, so the stream goes back mid-batch. Every message
    // in it resumes from the same entry, so reopening would step over the rest.
    await new Promise((resolve) =>
      globalThis.setTimeout(resolve, IDLE_TIMEOUT_MS * 3),
    );

    expect((await reader.read()).value).toBe("b\n");
    expect((await reader.read()).value).toBe("c\n");
    // Only then does it carry on from after that batch.
    expect((await reader.read()).value).toBe("d\n");
    expect(cp.requestedCursors).toEqual(["0-0", "1-0"]);
    await reader.cancel();
  } finally {
    await cp.shutdown();
  }
});
