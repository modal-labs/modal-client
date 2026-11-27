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
  type SandboxCreateParams,
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
const PERSISTENT_ID = "sp-1";

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
  mock.handleUnary("/PersistentSandboxCreate", (req: any) => {
    createReq = req;
    return { persistentSandboxId: PERSISTENT_ID, sandboxId: SESSION_1 };
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

test("create sends a PersistentSandboxCreate with the session timeouts", async () => {
  const { sb, mock, createReq } = await createPersistent();

  expect(createReq.appId).toBe("ap-1234");
  expect(createReq.definition.name).toBeFalsy();
  expect(createReq.definition.timeoutSecs).toBe(3600);
  expect(createReq.definition.idleTimeoutSecs).toBe(300);
  expect(createReq.definition.drainTimeoutSecs).toBe(60);
  expect(sb.sandboxId).toBe(SESSION_1);
  mock.handleUnary("/PersistentSandboxGet", () => ({ sandboxId: SESSION_1 }));
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

  mock.handleUnary("/SandboxTerminateV2", (req: any) => {
    expect(req.sandboxId).toBe(SESSION_1);
    return {};
  });
  mock.handleUnary("/SandboxWaitV2", (req: any) => {
    expect(req.sandboxId).toBe(SESSION_1);
    return { result: TERMINATED };
  });
  await sb.experimentalPause();
  // Pausing ends a session, not the persistent Sandbox.
  mock.handleUnary("/PersistentSandboxGet", () => ({ sandboxId: "" }));
  expect(await sb.poll()).toBeNull();

  mock.handleUnary("/PersistentSandboxResume", (req: any) => {
    expect(req.persistentSandboxId).toBe(PERSISTENT_ID);
    return { sandboxId: SESSION_2 };
  });
  await sb.experimentalResume();

  // Operations now target the new session.
  expectSidecarCreate(mock, SESSION_2);
  expect(await createSidecar(sb)).toBe(SIDECAR_ID);

  // Resuming a running persistent Sandbox keeps its session.
  mock.handleUnary("/PersistentSandboxResume", () => ({
    sandboxId: SESSION_2,
  }));
  await sb.experimentalResume();
  expectSidecarCreate(mock, SESSION_2);
  expect(await createSidecar(sb)).toBe(SIDECAR_ID);

  mock.assertExhausted();
});

test("tunnels are listed for the persistent Sandbox, not its session", async () => {
  const { sb, mock } = await createPersistent();

  mock.handleUnary("/PersistentSandboxTunnelsList", (req: any) => {
    expect(req.persistentSandboxId).toBe(PERSISTENT_ID);
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

  mock.handleUnary("/PersistentSandboxCreateConnectToken", (req: any) => {
    expect(req.persistentSandboxId).toBe(PERSISTENT_ID);
    expect(req.port).toBe(3000);
    expect(req.userMetadata).toBe("alice");
    return { url: "https://a-1-abcde.modal.host", token: "tok" };
  });
  expect(
    await sb.createConnectToken({ port: 3000, userMetadata: "alice" }),
  ).toEqual({ url: "https://a-1-abcde.modal.host", token: "tok" });
  mock.assertExhausted();
});

test("a failed operation resumes the persistent Sandbox and is retried", async () => {
  const { sb, mock } = await createPersistent();

  expectSidecarCreate(
    mock,
    SESSION_1,
    new ClientError("/SandboxContainerCreateV2", Status.NOT_FOUND, "stopped"),
  );
  mock.handleUnary("/PersistentSandboxGet", (req: any) => {
    expect(req.persistentSandboxId).toBe(PERSISTENT_ID);
    return { persistentSandboxId: PERSISTENT_ID, sandboxId: "" };
  });
  mock.handleUnary("/PersistentSandboxResume", () => ({
    sandboxId: SESSION_2,
  }));
  expectSidecarCreate(mock, SESSION_2);

  expect(await createSidecar(sb)).toBe(SIDECAR_ID);
  mock.assertExhausted();
});

test("a failed operation switches to an already-running session without resuming", async () => {
  const { sb, mock } = await createPersistent();

  expectSidecarCreate(
    mock,
    SESSION_1,
    new ClientError("/SandboxContainerCreateV2", Status.NOT_FOUND, "stopped"),
  );
  mock.handleUnary("/PersistentSandboxGet", () => ({
    persistentSandboxId: PERSISTENT_ID,
    sandboxId: SESSION_2,
  }));
  expectSidecarCreate(mock, SESSION_2);

  expect(await createSidecar(sb)).toBe(SIDECAR_ID);
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
  mock.handleUnary("/PersistentSandboxGet", () => ({
    persistentSandboxId: PERSISTENT_ID,
    sandboxId: SESSION_1,
  }));

  await expect(createSidecar(sb)).rejects.toBe(err);
  mock.assertExhausted();
});

test("resume retries while the previous session is still pausing", async () => {
  const { sb, mock } = await createPersistent();

  mock.handleUnary("/PersistentSandboxResume", () => {
    throw new ClientError(
      "/PersistentSandboxResume",
      Status.UNAVAILABLE,
      "pausing",
    );
  });
  mock.handleUnary("/PersistentSandboxResume", () => ({
    sandboxId: SESSION_2,
  }));
  await sb.experimentalResume();

  expectSidecarCreate(mock, SESSION_2);
  expect(await createSidecar(sb)).toBe(SIDECAR_ID);
  mock.assertExhausted();
});

test("terminate stops the persistent Sandbox for good", async () => {
  const { sb, mock } = await createPersistent();

  mock.handleUnary("/PersistentSandboxTerminate", (req: any) => {
    expect(req.persistentSandboxId).toBe(PERSISTENT_ID);
    return {};
  });
  await sb.terminate();
  expect(await sb.poll()).toBe(137);

  await expect(sb.experimentalPause()).rejects.toThrow(/has been terminated/);
  await expect(sb.experimentalResume()).rejects.toThrow(/has been terminated/);

  // Failed operations are no longer retried on a new session.
  const err = new ClientError(
    "/SandboxContainerCreateV2",
    Status.NOT_FOUND,
    "stopped",
  );
  expectSidecarCreate(mock, SESSION_1, err);
  await expect(createSidecar(sb)).rejects.toBe(err);
  mock.assertExhausted();
});

test("wait resolves once the persistent Sandbox is terminated", async () => {
  const { sb, mock } = await createPersistent();
  mock.handleUnary("/PersistentSandboxTerminate", () => ({}));

  mock.handleUnary("/PersistentSandboxGet", () => ({ sandboxId: SESSION_1 }));
  const waited = sb.wait();
  await sb.terminate();
  expect(await waited).toBe(137);
  mock.assertExhausted();
});

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

test.each(["running", "paused"])(
  "existing %s handle polls for external termination",
  async (state) => {
    const { sb, mock } = await createPersistent();
    mock.handleUnary("/PersistentSandboxGet", () => ({
      sandboxId: state === "running" ? SESSION_1 : "",
    }));
    expect(await sb.poll()).toBeNull();
    mock.handleUnary("/PersistentSandboxGet", () => ({
      terminatedAt: new Date(0),
    }));
    expect(await sb.poll()).toBe(137);
    expect(await sb.poll()).toBe(137);
    mock.assertExhausted();
  },
);

test.each(["running", "paused"])(
  "existing %s handle waits for external termination",
  async (state) => {
    const { sb, mock } = await createPersistent();
    mock.handleUnary("/PersistentSandboxGet", () => ({
      sandboxId: state === "running" ? SESSION_1 : "",
    }));
    mock.handleUnary("/PersistentSandboxGet", () => ({
      terminatedAt: new Date(0),
    }));
    expect(await sb.wait()).toBe(137);
    mock.assertExhausted();
  },
);

test.each(["poll", "wait"] as const)(
  "%s propagates persistent lookup errors",
  async (operation) => {
    const { sb, mock } = await createPersistent();
    const error = new ClientError(
      "/PersistentSandboxGet",
      Status.NOT_FOUND,
      "unknown",
    );
    mock.handleUnary("/PersistentSandboxGet", () => {
      throw error;
    });
    await expect(sb[operation]()).rejects.toBe(error);
    mock.assertExhausted();
  },
);

test.each(["stdin", "stdout", "stderr"] as const)(
  "detached persistent %s does not resume",
  async (streamName) => {
    const { sb: sandbox, mock } = await createPersistent();
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

test("an exec moves off a session whose command router it cannot reach", async () => {
  const { sb, mock } = await createPersistent();

  // The old session's command router can't be reached, so no command was
  // sent and the exec moves to a resumed session. That one refuses too, and
  // its error is what the caller gets.
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
  mock.handleUnary("/PersistentSandboxGet", () => ({
    persistentSandboxId: PERSISTENT_ID,
    sandboxId: "",
  }));
  mock.handleUnary("/PersistentSandboxResume", () => ({
    sandboxId: SESSION_2,
  }));
  refuseRouterAccess(
    SESSION_2,
    new ClientError(
      "/SandboxGetCommandRouterAccess",
      Status.PERMISSION_DENIED,
      "second session",
    ),
  );

  await expect(sb.exec(["echo", "hello"])).rejects.toThrow("second session");
  mock.assertExhausted();
});
