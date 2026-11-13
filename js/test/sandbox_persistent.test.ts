import { expect, test } from "vitest";
import {
  App,
  ExperimentalOutboundPolicy,
  Image,
  InvalidError,
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
import { newSandboxImpl } from "../src/sandbox";

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

function expectConnectToken(
  mock: MockGrpcClient,
  sandboxId: string,
  err?: ClientError,
) {
  mock.handleUnary("/SandboxCreateConnectTokenV2", (req: any) => {
    expect(req.sandboxId).toBe(sandboxId);
    if (err) throw err;
    return { url: "https://connect", token: "tok" };
  });
}

test("create sends a PersistentSandboxCreate with the session timeouts", async () => {
  const { sb, mock, createReq } = await createPersistent();

  expect(createReq.appId).toBe("ap-1234");
  expect(createReq.definition.name).toBeFalsy();
  expect(createReq.definition.timeoutSecs).toBe(3600);
  expect(createReq.definition.idleTimeoutSecs).toBe(300);
  expect(sb.sandboxId).toBe(SESSION_1);
  expect(await sb.poll()).toBeNull();
  mock.assertExhausted();
});

test("experimentalCreate also creates a persistent Sandbox", async () => {
  const { mc, mock, app, image } = setup();
  mock.handleUnary("/PersistentSandboxCreate", () => ({
    persistentSandboxId: PERSISTENT_ID,
    sandboxId: SESSION_1,
  }));
  const sb = await mc.sandboxes.experimentalCreate(app, image, {
    experimentalPersistent: true,
  });
  expect(sb.sandboxId).toBe(SESSION_1);
  mock.assertExhausted();
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
  expect(await sb.poll()).toBeNull();

  mock.handleUnary("/PersistentSandboxResume", (req: any) => {
    expect(req.persistentSandboxId).toBe(PERSISTENT_ID);
    return { sandboxId: SESSION_2 };
  });
  await sb.experimentalResume();

  // Operations now target the new session.
  mock.handleUnary("/SandboxCreateConnectTokenV2", (req: any) => {
    expect(req.sandboxId).toBe(SESSION_2);
    return { url: "https://connect", token: "tok" };
  });
  expect(await sb.createConnectToken()).toEqual({
    url: "https://connect",
    token: "tok",
  });

  // Resuming a running persistent Sandbox keeps its session.
  mock.handleUnary("/PersistentSandboxResume", () => ({
    sandboxId: SESSION_2,
  }));
  await sb.experimentalResume();
  expectConnectToken(mock, SESSION_2);
  expect((await sb.createConnectToken()).token).toBe("tok");

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

test("a failed operation resumes the persistent Sandbox and is retried", async () => {
  const { sb, mock } = await createPersistent();

  expectConnectToken(
    mock,
    SESSION_1,
    new ClientError(
      "/SandboxCreateConnectTokenV2",
      Status.NOT_FOUND,
      "stopped",
    ),
  );
  mock.handleUnary("/PersistentSandboxGet", (req: any) => {
    expect(req.persistentSandboxId).toBe(PERSISTENT_ID);
    return { persistentSandboxId: PERSISTENT_ID, sandboxId: "" };
  });
  mock.handleUnary("/PersistentSandboxResume", () => ({
    sandboxId: SESSION_2,
  }));
  expectConnectToken(mock, SESSION_2);

  expect((await sb.createConnectToken()).token).toBe("tok");
  mock.assertExhausted();
});

test("a failed operation switches to an already-running session without resuming", async () => {
  const { sb, mock } = await createPersistent();

  expectConnectToken(
    mock,
    SESSION_1,
    new ClientError(
      "/SandboxCreateConnectTokenV2",
      Status.NOT_FOUND,
      "stopped",
    ),
  );
  mock.handleUnary("/PersistentSandboxGet", () => ({
    persistentSandboxId: PERSISTENT_ID,
    sandboxId: SESSION_2,
  }));
  expectConnectToken(mock, SESSION_2);

  expect((await sb.createConnectToken()).token).toBe("tok");
  mock.assertExhausted();
});

test("a failed operation is not retried while its session is still running", async () => {
  const { sb, mock } = await createPersistent();

  const err = new ClientError(
    "/SandboxCreateConnectTokenV2",
    Status.NOT_FOUND,
    "no tunnels",
  );
  expectConnectToken(mock, SESSION_1, err);
  mock.handleUnary("/PersistentSandboxGet", () => ({
    persistentSandboxId: PERSISTENT_ID,
    sandboxId: SESSION_1,
  }));

  await expect(sb.createConnectToken()).rejects.toBe(err);
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

  expectConnectToken(mock, SESSION_2);
  expect((await sb.createConnectToken()).token).toBe("tok");
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
    "/SandboxCreateConnectTokenV2",
    Status.NOT_FOUND,
    "stopped",
  );
  expectConnectToken(mock, SESSION_1, err);
  await expect(sb.createConnectToken()).rejects.toBe(err);
  mock.assertExhausted();
});

test("wait resolves once the persistent Sandbox is terminated", async () => {
  const { sb, mock } = await createPersistent();
  mock.handleUnary("/PersistentSandboxTerminate", () => ({}));

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
    sb.updateNetworkPolicy({
      outboundCidrAllowlist: ["10.0.0.0/8"],
      outboundDomainAllowlist: [],
    }),
  ).rejects.toThrow(/network access/);
  await expect(
    sb.experimentalUpdateOutboundPolicy(new ExperimentalOutboundPolicy()),
  ).rejects.toThrow(/outbound policy/);
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
