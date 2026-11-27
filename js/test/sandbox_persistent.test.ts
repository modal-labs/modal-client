import { afterEach, beforeEach, expect, test, vi } from "vitest";
import {
  App,
  ClientClosedError,
  ExperimentalOutboundPolicy,
  Image,
  InvalidError,
  OutboundPolicy,
  OutboundRoute,
  Sandbox,
  SandboxPaused,
  type SandboxCreateParams,
  TimeoutError,
} from "modal";
import { ClientError, Status } from "nice-grpc";
import { GenericResult_GenericStatus } from "../proto/modal_proto/api";
import {} from "../src/sandbox_impl_persistent";
import {
  createMockModalClients,
  type MockGrpcClient,
} from "../test-support/grpc_mock";
import { newSandboxImpl } from "../src/sandbox_common";

const V1_SANDBOX_ID = "sb-nGEijt9WbBMlGrsPH9FOaC";
const SESSION_1 = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV";
const SESSION_2 = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAW";
const PERSISTENT_ID = "sx-z1arz3ndektsv4rrffq69g5fa";

const TERMINATED = {
  status: GenericResult_GenericStatus.GENERIC_STATUS_TERMINATED,
  exitcode: 0,
  exception: "",
};

function setup() {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  return {
    mc,
    mock,
    app: new App("ap-1234"),
    image: new Image(mc, "im-123", ""),
  };
}

async function createPersistent(
  params: SandboxCreateParams = {},
): Promise<{ sb: Sandbox; mock: MockGrpcClient; createReq: any }> {
  const { mc, mock, app, image } = setup();
  let createReq: any;
  mock.handleUnary("/SxSandboxCreate", (req: any) => {
    createReq = req;
    return { sxId: PERSISTENT_ID, sandboxId: SESSION_1 };
  });
  const sb = await mc.sandboxes.create(app, image, {
    experimentalPersistent: true,
    ...params,
  });
  return { sb, mock, createReq };
}

// Sidecar create is a single RPC on the current session, which makes it a
// convenient probe for the operations that follow the persistent Sandbox to its
// running session.
const SIDECAR_ID = "sb-ctr-1";

beforeEach(() => {
  vi.stubEnv("MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE", undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
});

function expectSidecarCreate(
  mock: MockGrpcClient,
  sandboxId: string,
  err?: ClientError,
) {
  mock.handleUnary("/SandboxContainerCreateV2", (req: any) => {
    expect(req.sandboxId).toBe(sandboxId);
    if (err) throw err;
    return { containerId: SIDECAR_ID };
  });
}

async function createSidecar(sb: Sandbox): Promise<string> {
  const { mockClient } = createMockModalClients();
  const image = new Image(mockClient, "im-123", "");
  const sidecar = await sb.experimentalSidecars.create("worker", image);
  return sidecar.containerId;
}

test.each(["running", "paused", "terminated", "missing", "stale"])(
  "fromId persistent Sandbox: %s",
  async (state) => {
    const { mc, mock } = setup();
    let calls = 0;
    mock.handleUnary("/SxSandboxGet", (req: any) => {
      calls++;
      expect(req.sxId).toBe(PERSISTENT_ID);
      if (state === "missing")
        throw new ClientError("/SxSandboxGet", Status.NOT_FOUND, "unknown");
      return {
        sxId: PERSISTENT_ID,
        currentSandboxId:
          state === "running" || state === "stale" ? SESSION_1 : undefined,
        terminatedAt: state === "terminated" ? new Date(0) : undefined,
      };
    });
    const sb = await mc.sandboxes.fromId(PERSISTENT_ID);
    expect(calls).toBe(0);
    if (state === "missing") {
      await expect(sb.poll()).rejects.toThrow("unknown");
    } else {
      expect(sb.sandboxId).toBe(PERSISTENT_ID);
      if (state === "terminated") {
        expect(await sb.poll()).not.toBeNull();
        expect(calls).toBe(1);
        expect(await sb.wait()).not.toBeNull();
        await expect(createSidecar(sb)).rejects.toThrow("has been terminated");
        await expect(sb.experimentalPause()).rejects.toThrow(
          "has been terminated",
        );
        await expect(sb.experimentalResume()).rejects.toThrow(
          "has been terminated",
        );
      } else {
        expect(await sb.poll()).toBeNull();
        expect(calls).toBe(1);
        if (state === "paused") {
          mock.assertExhausted();
          mock.handleUnary("/SxSandboxPause", () => ({}));
          await sb.experimentalPause();
        }
        if (state !== "running") {
          if (state === "stale")
            expectSidecarCreate(
              mock,
              SESSION_1,
              new ClientError(
                "/SandboxContainerCreateV2",
                Status.NOT_FOUND,
                "task gone",
              ),
            );
          mock.handleUnary("/SxSandboxGet", () => ({
            currentSandboxId: undefined,
          }));
          await expect(createSidecar(sb)).rejects.toThrow(SandboxPaused);
          mock.assertExhausted();
          mock.handleUnary("/SxSandboxResume", () => ({
            sandboxId: SESSION_2,
          }));
          await sb.experimentalResume();
        }
        expectSidecarCreate(mock, state === "running" ? SESSION_1 : SESSION_2);
        expect(await createSidecar(sb)).toBe(SIDECAR_ID);
      }
    }
    mock.assertExhausted();
  },
);

test("create sends an SxSandboxCreate with the session timeouts", async () => {
  const { sb, mock, createReq } = await createPersistent();

  expect(createReq.appId).toBe("ap-1234");
  expect(createReq.definition.name).toBeFalsy();
  expect(createReq.definition.timeoutSecs).toBe(3600);
  expect(createReq.definition.idleTimeoutSecs).toBe(300);
  expect(createReq.definition.drainTimeoutSecs).toBe(60);
  expect(sb.sandboxId).toBe(PERSISTENT_ID);
  mock.handleUnary("/SxSandboxGet", () => ({
    currentSandboxId: SESSION_1,
  }));
  expect(await sb.poll()).toBeNull();
  mock.assertExhausted();
});

test("create sends the requested drain timeout", async () => {
  const { createReq } = await createPersistent({
    experimentalOptions: {
      drain_timeout_secs: 300,
      enable_exit_snapshot: true,
    },
  });
  expect(createReq.definition.drainTimeoutSecs).toBe(300);
  expect(createReq.definition.experimentalOptionsV2).toEqual({
    enable_exit_snapshot: "true",
  });
});

test("create sends the default drain timeout", async () => {
  const { createReq } = await createPersistent();
  expect(createReq.definition.drainTimeoutSecs).toBe(60);
});

test("create rejects a drain timeout for an ordinary Sandbox", async () => {
  const { mc, app, image } = setup();
  await expect(
    mc.sandboxes.create(app, image, {
      experimentalOptions: { drain_timeout_secs: 60 },
    }),
  ).rejects.toThrow(InvalidError);
});

test.each<[string, SandboxCreateParams]>([
  ["timeoutMs", { timeoutMs: 300_000 }],
  ["idleTimeoutMs", { idleTimeoutMs: 60_000 }],
  ["name", { name: "my-persistent-sandbox" }],
  ["gpu", { gpu: "any" }],
  ["pty", { pty: true }],
  ["env", { env: { A: "1" } }],
  ["tags", { tags: { a: "b" } }],
  ["experimentalEnableSnapshot", { experimentalEnableSnapshot: true }],
  ["a zero drain timeout", { experimentalOptions: { drain_timeout_secs: 0 } }],
  [
    "a drain timeout over 10 minutes",
    { experimentalOptions: { drain_timeout_secs: 601 } },
  ],
  [
    "a fractional drain timeout",
    { experimentalOptions: { drain_timeout_secs: 1.5 } },
  ],
  [
    "a drain timeout that isn't a number",
    { experimentalOptions: { drain_timeout_secs: "60" } },
  ],
])("create rejects %s for persistent Sandboxes", async (_name, params) => {
  const { mc, app, image } = setup();
  // No RPC handlers are registered, so any request would fail with a
  // different error than the validation error expected here.
  await expect(
    mc.sandboxes.create(app, image, {
      experimentalPersistent: true,
      ...params,
    }),
  ).rejects.toThrow(InvalidError);
});

test("pause stops the session and resume starts a new one", async () => {
  const { sb, mock } = await createPersistent();

  // Pausing stops the running session, then waits for it to stop.
  mock.handleUnary("/SxSandboxPause", (req: any) => {
    expect(req.sxId).toBe(PERSISTENT_ID);
    return { sandboxId: SESSION_1 };
  });
  mock.handleUnary("/SandboxWaitV2", (req: any) => {
    expect(req.sandboxId).toBe(SESSION_1);
    return { result: TERMINATED };
  });
  await sb.experimentalPause();
  // Pausing ends a session, not the persistent Sandbox.
  mock.handleUnary("/SxSandboxGet", () => ({
    currentSandboxId: undefined,
  }));
  expect(await sb.poll()).toBeNull();

  mock.handleUnary("/SxSandboxResume", (req: any) => {
    expect(req.sxId).toBe(PERSISTENT_ID);
    return { sandboxId: SESSION_2 };
  });
  await sb.experimentalResume();

  // Operations now target the new session.
  expectSidecarCreate(mock, SESSION_2);
  expect(await createSidecar(sb)).toBe(SIDECAR_ID);

  // Resuming a running persistent Sandbox keeps its session.
  mock.handleUnary("/SxSandboxResume", () => ({
    sandboxId: SESSION_2,
  }));
  await sb.experimentalResume();
  expectSidecarCreate(mock, SESSION_2);
  expect(await createSidecar(sb)).toBe(SIDECAR_ID);

  mock.assertExhausted();
});

test("pause waits for a session that was started through another handle", async () => {
  const { sb, mock } = await createPersistent();

  // Another handle paused and resumed the Sandbox, so a different session is running.
  mock.handleUnary("/SxSandboxPause", () => ({ sandboxId: SESSION_2 }));
  mock.handleUnary("/SandboxWaitV2", (req: any) => {
    expect(req.sandboxId).toBe(SESSION_2);
    return { result: TERMINATED };
  });
  await sb.experimentalPause();
  mock.assertExhausted();
});

test("pause does not resume a paused persistent Sandbox", async () => {
  const { sb, mock } = await createPersistent();

  // Pausing a paused Sandbox stops nothing, so no other request is made.
  mock.handleUnary("/SxSandboxPause", () => ({}));
  await sb.experimentalPause();
  mock.assertExhausted();
});

test("pause retries while a session is starting", async () => {
  const { sb, mock } = await createPersistent();

  mock.handleUnary("/SxSandboxPause", () => {
    throw new ClientError("/SxSandboxPause", Status.UNAVAILABLE, "starting");
  });
  mock.handleUnary("/SxSandboxPause", () => ({ sandboxId: SESSION_1 }));
  mock.handleUnary("/SandboxWaitV2", () => ({ result: TERMINATED }));
  await sb.experimentalPause();
  mock.assertExhausted();
});

test("the exit snapshot is only available once the persistent Sandbox is terminated", async () => {
  const { sb, mock } = await createPersistent();

  // A paused Sandbox has not exited: it is neither resumed nor asked for a snapshot.
  mock.handleUnary("/SxSandboxGet", () => ({
    currentSandboxId: undefined,
  }));
  await expect(
    sb.experimentalGetExitSnapshot({ timeoutMs: 0 }),
  ).rejects.toThrow(TimeoutError);
  mock.assertExhausted();
});

test.each(["this handle", "another handle"])(
  "the exit snapshot is read from the last session after %s terminates",
  async (terminatedBy) => {
    const clients = createMockModalClients();
    const mock = clients.mockCpClient;
    const sb = await clients.mockClient.sandboxes.fromId(PERSISTENT_ID);

    if (terminatedBy === "this handle") {
      // Terminating reports the last session, so no lookup is needed.
      mock.handleUnary("/SxSandboxTerminate", () => ({
        lastSessionSandboxId: SESSION_2,
      }));
      await sb.terminate();
    } else {
      mock.handleUnary("/SxSandboxGet", () => ({
        currentSandboxId: SESSION_2,
        terminatedAt: new Date(0),
      }));
    }
    mock.handleUnary("/SandboxGetExitSnapshotV2", (req: any) => {
      expect(req.sandboxId).toBe(SESSION_2);
      return { success: { imageId: "im-exit-snapshot-123" } };
    });
    const image = await sb.experimentalGetExitSnapshot({ timeoutMs: 0 });
    expect(image.imageId).toBe("im-exit-snapshot-123");
    mock.assertExhausted();
  },
);

test("tunnels are listed for the persistent Sandbox, not its session", async () => {
  const { sb, mock } = await createPersistent();

  mock.handleUnary("/SxSandboxTunnelsList", (req: any) => {
    expect(req.sxId).toBe(PERSISTENT_ID);
    return {
      tunnels: [{ containerPort: 8080, host: "tunnel.modal.host", port: 443 }],
    };
  });
  const tunnels = await sb.tunnels();
  expect(Object.keys(tunnels)).toEqual(["8080"]);
  expect(tunnels[8080].host).toBe("tunnel.modal.host");
  mock.assertExhausted();
});

test("connect tokens are created for the persistent Sandbox, not its session", async () => {
  const { sb, mock } = await createPersistent();

  mock.handleUnary("/SxSandboxCreateConnectToken", (req: any) => {
    expect(req.sxId).toBe(PERSISTENT_ID);
    expect(req.port).toBe(3000);
    expect(req.userMetadata).toBe("alice");
    return { url: "https://a-1-abcde.modal.host", token: "tok" };
  });
  expect(
    await sb.createConnectToken({ port: 3000, userMetadata: "alice" }),
  ).toEqual({ url: "https://a-1-abcde.modal.host", token: "tok" });
  mock.assertExhausted();
});

test("a failed operation on a stopped session requires explicit resume", async () => {
  const { sb, mock } = await createPersistent();

  expectSidecarCreate(
    mock,
    SESSION_1,
    new ClientError("/SandboxContainerCreateV2", Status.NOT_FOUND, "stopped"),
  );
  mock.handleUnary("/SxSandboxGet", (req: any) => {
    expect(req.sxId).toBe(PERSISTENT_ID);
    return {
      sxId: PERSISTENT_ID,
      currentSandboxId: SESSION_1,
    };
  });
  mock.handleUnary("/SandboxWaitV2", () => ({ result: TERMINATED }));

  await expect(createSidecar(sb)).rejects.toThrow(SandboxPaused);
  mock.assertExhausted();

  mock.handleUnary("/SxSandboxResume", () => ({
    sandboxId: SESSION_2,
  }));
  expectSidecarCreate(mock, SESSION_2);

  await sb.experimentalResume();
  expect(await createSidecar(sb)).toBe(SIDECAR_ID);
  mock.assertExhausted();
});

test("a stale handle requires explicit resume even if another session is running", async () => {
  const { sb, mock } = await createPersistent();

  expectSidecarCreate(
    mock,
    SESSION_1,
    new ClientError("/SandboxContainerCreateV2", Status.NOT_FOUND, "stopped"),
  );
  mock.handleUnary("/SxSandboxGet", () => ({
    sxId: PERSISTENT_ID,
    currentSandboxId: SESSION_2,
  }));
  await expect(createSidecar(sb)).rejects.toThrow(SandboxPaused);
  mock.assertExhausted();
});

test("a failed operation is not retried while its session is still running", async () => {
  const { sb, mock } = await createPersistent();

  const err = new ClientError(
    "/SandboxContainerCreateV2",
    Status.NOT_FOUND,
    "no tunnels",
  );
  expectSidecarCreate(mock, SESSION_1, err);
  mock.handleUnary("/SxSandboxGet", () => ({
    sxId: PERSISTENT_ID,
    currentSandboxId: SESSION_1,
  }));
  mock.handleUnary("/SandboxWaitV2", () => ({}));

  await expect(createSidecar(sb)).rejects.toBe(err);
  mock.assertExhausted();
});

test("resume retries while the previous session is still pausing", async () => {
  const { sb, mock } = await createPersistent();

  mock.handleUnary("/SxSandboxResume", () => {
    throw new ClientError("/SxSandboxResume", Status.UNAVAILABLE, "pausing");
  });
  mock.handleUnary("/SxSandboxResume", () => ({
    sandboxId: SESSION_2,
  }));
  await sb.experimentalResume();

  expectSidecarCreate(mock, SESSION_2);
  expect(await createSidecar(sb)).toBe(SIDECAR_ID);
  mock.assertExhausted();
});

test("terminate stops the persistent Sandbox for good", async () => {
  const { sb, mock } = await createPersistent();

  mock.handleUnary("/SxSandboxTerminate", (req: any) => {
    expect(req.sxId).toBe(PERSISTENT_ID);
    return {};
  });
  await sb.terminate();
  expect(await sb.poll()).toBe(137);

  await expect(sb.experimentalPause()).rejects.toThrow(/has been terminated/);
  await expect(sb.experimentalResume()).rejects.toThrow(/has been terminated/);

  // Operations fail without a request to the last session.
  await expect(createSidecar(sb)).rejects.toThrow(/has been terminated/);
  mock.assertExhausted();
});

test("wait resolves once the persistent Sandbox is terminated", async () => {
  const { sb, mock } = await createPersistent();
  mock.handleUnary("/SxSandboxTerminate", () => ({}));
  mock.handleUnary("/SxSandboxGet", () => ({
    currentSandboxId: SESSION_1,
  }));

  const waited = sb.wait();
  await sb.terminate();
  expect(await waited).toBe(137);
  mock.assertExhausted();
});

test.each(["running", "paused"])(
  "reattached %s handle polls for external termination",
  async (state) => {
    const { mc, mock } = setup();
    const response = {
      currentSandboxId: state === "running" ? SESSION_1 : undefined,
    };
    const sandboxA = await mc.sandboxes.fromId(PERSISTENT_ID);
    const sandboxB = await mc.sandboxes.fromId(PERSISTENT_ID);
    mock.handleUnary("/SxSandboxGet", () => response);
    expect(await sandboxB.poll()).toBeNull();
    mock.handleUnary("/SxSandboxTerminate", () => ({}));
    await sandboxA.terminate();
    mock.handleUnary("/SxSandboxGet", () => ({
      terminatedAt: new Date(0),
    }));
    expect(await sandboxB.poll()).toBe(137);
    expect(await sandboxB.poll()).toBe(137);
    mock.assertExhausted();
  },
);

test.each(["running", "paused"])(
  "reattached %s handle waits for external termination",
  async (state) => {
    const { mc, mock } = setup();
    const response = {
      currentSandboxId: state === "running" ? SESSION_1 : undefined,
    };
    const sandboxA = await mc.sandboxes.fromId(PERSISTENT_ID);
    const sandboxB = await mc.sandboxes.fromId(PERSISTENT_ID);
    mock.handleUnary("/SxSandboxTerminate", () => ({}));
    mock.handleUnary("/SxSandboxGet", async () => {
      await sandboxA.terminate();
      return response;
    });
    mock.handleUnary("/SxSandboxGet", () => ({
      terminatedAt: new Date(0),
    }));
    expect(await sandboxB.wait()).toBe(137);
    mock.assertExhausted();
  },
);

test.each(["poll", "wait"] as const)(
  "%s propagates persistent lookup errors",
  async (operation) => {
    const { sb, mock } = await createPersistent();
    const error = new ClientError("/SxSandboxGet", Status.NOT_FOUND, "unknown");
    mock.handleUnary("/SxSandboxGet", () => {
      throw error;
    });
    await expect(sb[operation]()).rejects.toBe(error);
    mock.assertExhausted();
  },
);

test.each(["stdin", "stdout", "stderr"] as const)(
  "detached sessionless persistent %s does not resume",
  async (streamName) => {
    const { mc, mock } = setup();
    const sandbox = await mc.sandboxes.fromId(PERSISTENT_ID);
    if (streamName === "stdin") {
      const stream = sandbox.stdin;
      sandbox.detach();
      await expect(stream.writeText("hello")).rejects.toBeInstanceOf(
        ClientClosedError,
      );
    } else {
      const stream = sandbox[streamName];
      sandbox.detach();
      await expect(stream.readText()).rejects.toBeInstanceOf(ClientClosedError);
    }
    mock.assertExhausted();
  },
);

test("operations that only apply to a session are rejected", async () => {
  const { sb, mock } = await createPersistent();

  await expect(sb.getTags()).rejects.toThrow(/Tags/);
  await expect(sb.setTags({ a: "b" })).rejects.toThrow(/Tags/);
  await expect(sb.experimentalSetName("other")).rejects.toThrow(
    /can't be changed/,
  );
  await expect(sb.experimentalSnapshot()).rejects.toThrow(/Memory snapshots/);
  await expect(
    // eslint-disable-next-line @typescript-eslint/no-deprecated -- tests the deprecated method
    sb.updateNetworkPolicy({
      outboundCidrAllowlist: ["10.0.0.0/8"],
      outboundDomainAllowlist: [],
    }),
  ).rejects.toThrow(/network access/);
  await expect(
    // eslint-disable-next-line @typescript-eslint/no-deprecated -- tests the deprecated method
    sb.experimentalUpdateOutboundPolicy(new ExperimentalOutboundPolicy()),
  ).rejects.toThrow(/outbound policy/);
  await expect(
    sb.setOutboundPolicy(new OutboundPolicy({ routes: [OutboundRoute.any()] })),
  ).rejects.toThrow(/Setting the outbound policy/);
  await expect(sb.mountImage("/mnt")).rejects.toThrow(/Mounting images/);
  await expect(sb.unmountImage("/mnt")).rejects.toThrow(/Mounting images/);
  mock.assertExhausted();
});

test.each([
  ["V1", V1_SANDBOX_ID],
  ["V2", SESSION_1],
])("%s Sandboxes don't support pause and resume", async (_name, sandboxId) => {
  const { mc } = setup();
  const sb = new Sandbox(newSandboxImpl(mc, sandboxId));
  await expect(sb.experimentalPause()).rejects.toThrow(
    /requires a persistent Sandbox/,
  );
  await expect(sb.experimentalResume()).rejects.toThrow(
    /requires a persistent Sandbox/,
  );
});

test("an exec reports a stopped session whose command router it cannot reach", async () => {
  const { sb, mock } = await createPersistent();

  const refuseRouterAccess = (sandboxId: string, err: ClientError) => {
    mock.handleUnary("/SandboxGetTaskIdV2", (req: any) => {
      expect(req.sandboxId).toBe(sandboxId);
      return { taskId: `ta-${sandboxId}` };
    });
    mock.handleUnary("/SandboxGetCommandRouterAccess", (req: any) => {
      expect(req.sandboxId).toBe(sandboxId);
      throw err;
    });
  };
  refuseRouterAccess(
    SESSION_1,
    new ClientError(
      "/SandboxGetCommandRouterAccess",
      Status.FAILED_PRECONDITION,
      "Sandbox has finished",
    ),
  );
  mock.handleUnary("/SxSandboxGet", () => ({
    sxId: PERSISTENT_ID,
    currentSandboxId: SESSION_1,
  }));
  mock.handleUnary("/SandboxWaitV2", () => ({ result: TERMINATED }));

  await expect(sb.exec(["echo", "hello"])).rejects.toThrow(SandboxPaused);
  mock.assertExhausted();
});
