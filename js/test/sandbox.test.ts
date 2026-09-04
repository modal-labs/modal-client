import { tc } from "../test-support/test-client";
import { parseGpuConfig } from "../src/app";
import {
  buildOutboundNetworkAccess,
  buildSandboxCreateRequestProto,
  buildSandboxCreateV2RequestProto,
  buildTaskMountDirectoryRequestProto,
  buildTaskExecStartRequestProto,
  buildTaskSnapshotDirectoryRequestProto,
  getSandboxVersion,
  SandboxVersion,
  validateExperimentalEncryptionKey,
  validateExecArgs,
  Probe,
  Sandbox,
  ContainerProcess,
} from "../src/sandbox";
import { expect, test, onTestFinished, vi } from "vitest";
import {
  FileDescriptor,
  GPUConfig,
  PTYInfo_PTYType,
  NetworkAccess_NetworkAccessType,
  GenericResult_GenericStatus,
  ImageGetOrCreateResponse,
  AppGetOrCreateResponse,
  SandboxCreateResponse,
  SandboxCreateV2Response,
  SandboxGetExitSnapshotResponse_ErrorCode,
  SandboxRestoreV2Response,
  SandboxSnapshotGetResponse,
} from "../proto/modal_proto/api";
import { createMockModalClients } from "../test-support/grpc_mock";
import { TaskCommandRouterClientImpl } from "../src/task_command_router_client";
import { SandboxSnapshot } from "../src/sandbox_snapshot";
import {
  SandboxStdioReadV2Response,
  TaskExecStdioReadResponse,
  TaskSetNetworkAccessRequest,
  TaskSnapshotFilesystemRequest,
  TaskSnapshotMemoryRequest,
} from "../proto/modal_proto/task_command_router";
import {
  AlreadyExistsError,
  ConflictError,
  ExecutionError,
  Image,
  InvalidError,
  NotFoundError,
  SnapshotCreationError,
  TimeoutError,
} from "modal";
import { ClientError, Status } from "nice-grpc";

const V1_SANDBOX_ID = "sb-nGEijt9WbBMlGrsPH9FOaC";
const V2_SANDBOX_ID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV";

function mockCommandRouter(methods: Record<string, unknown>): void {
  const tryInit = vi
    .spyOn(TaskCommandRouterClientImpl, "tryInit")
    .mockResolvedValue({
      close: vi.fn(),
      ...methods,
    } as unknown as TaskCommandRouterClientImpl);
  onTestFinished(() => tryInit.mockRestore());
}

test("CreateOneSandbox", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  // The readiness probe gives the test something to wait on, so that
  // terminating the Sandbox does not race with its startup.
  const sb = await tc.sandboxes.create(app, image, {
    readinessProbe: Probe.withExec(["true"]),
  });
  expect(sb.sandboxId).toBeTruthy();
  await sb.waitUntilReady();

  expect(await sb.terminate({ wait: true })).toBe(137);
}, 30000); // fixme(ayush): this probably shouldn't take > 20s

test("CreateOneSandboxTerminateWaitWorks", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  // The readiness probe gives the test something to wait on, so that
  // terminating the Sandbox does not race with its startup.
  const sb = await tc.sandboxes.create(app, image, {
    readinessProbe: Probe.withExec(["true"]),
  });
  expect(sb.sandboxId).toBeTruthy();
  await sb.waitUntilReady();

  await sb.terminate();
  expect(await sb.wait()).toBe(137);
}, 30000); // fixme(ayush): this probably shouldn't take > 20s

test("PassCatToStdin", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image, { command: ["cat"] });
  onTestFinished(async () => await sb.terminate());

  await sb.stdin.writeText("this is input that should be mirrored by cat");
  await sb.stdin.close();
  expect(await sb.stdout.readText()).toBe(
    "this is input that should be mirrored by cat",
  );
});

test("SandboxStdoutStaysReadableAfterTerminate", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image, {
    command: ["sh", "-c", "echo line-one; echo line-two; sleep 60"],
  });
  onTestFinished(async () => await sb.terminate());

  // Reading a first chunk blocks until the Sandbox is up and has written, so
  // both lines exist before the Sandbox is stopped.
  const reader = sb.stdout.getReader();
  const first = await reader.read();
  expect(first.done).toBe(false);
  reader.releaseLock();

  await sb.terminate();

  // terminate() leaves the Sandbox attached, so the stream keeps working and
  // none of the output is lost.
  const output = (first.value ?? "") + (await sb.stdout.readText());
  expect(output).toContain("line-one");
  expect(output).toContain("line-two");
});

test("SandboxFirstStdoutReadAfterTerminate", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  // The readiness probe gives the test something to wait on, so that
  // terminating the Sandbox does not race with its startup.
  const sb = await tc.sandboxes.create(app, image, {
    command: ["sh", "-c", "echo only-line; sleep 60"],
    readinessProbe: Probe.withExec(["true"]),
  });
  await sb.waitUntilReady();

  await sb.terminate();

  // Nothing read this Sandbox's output before it was stopped, so the first read
  // has to open the stream against a Sandbox that has already finished.
  expect(await sb.stdout.readText()).toContain("only-line");
});

// The same for a Sandbox created with experimentalCreate.
test("SandboxFirstStdoutReadAfterTerminateV2", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.experimentalCreate(app, image, {
    command: ["sh", "-c", "echo only-line; sleep 60"],
    readinessProbe: Probe.withExec(["true"]),
  });
  await sb.waitUntilReady();

  await sb.terminate();

  expect(await sb.stdout.readText()).toContain("only-line");
});

test("IgnoreLargeStdout", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("python:3.13-alpine");

  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const p = await sb.exec(["python", "-c", `print("a" * 1_000_000)`], {
    stdout: "ignore",
  });
  expect(await p.stdout.readText()).toBe(""); // Stdout is ignored
  // Stdout should be consumed after cancel, without blocking the process.
  expect(await p.wait()).toBe(0);
});

test("SandboxCreateOptions", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image, {
    command: ["echo", "hello, params"],
    cloud: "aws",
    regions: ["us-east-1", "us-west-2"],
    verbose: true,
  });
  onTestFinished(async () => await sb.terminate());

  expect(sb.sandboxId).toMatch(/^sb-/);

  const exitCode = await sb.wait();
  expect(exitCode).toBe(0);

  await expect(
    tc.sandboxes.create(app, image, {
      cloud: "invalid-cloud",
    }),
  ).rejects.toThrow("INVALID_ARGUMENT");

  await expect(
    tc.sandboxes.create(app, image, {
      regions: ["invalid-region"],
    }),
  ).rejects.toThrow("INVALID_ARGUMENT");
});

test("SandboxExecOptions", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());
  const p = await sb.exec(["pwd"], {
    workdir: "/tmp",
    timeoutMs: 5000,
  });

  expect(await p.stdout.readText()).toBe("/tmp\n");
  expect(await p.wait()).toBe(0);
});

test("parseGpuConfig", () => {
  expect(parseGpuConfig(undefined)).toEqual(GPUConfig.create({}));
  expect(parseGpuConfig("T4")).toEqual({
    type: 0,
    count: 1,
    gpuType: "T4",
  });
  expect(parseGpuConfig("A10G")).toEqual({
    type: 0,
    count: 1,
    gpuType: "A10G",
  });
  expect(parseGpuConfig("A100-80GB")).toEqual({
    type: 0,
    count: 1,
    gpuType: "A100-80GB",
  });
  expect(parseGpuConfig("A100-80GB:3")).toEqual({
    type: 0,
    count: 3,
    gpuType: "A100-80GB",
  });
  expect(parseGpuConfig("T4:2")).toEqual({
    type: 0,
    count: 2,
    gpuType: "T4",
  });
  expect(parseGpuConfig("a100:4")).toEqual({
    type: 0,
    count: 4,
    gpuType: "A100",
  });

  expect(() => parseGpuConfig("T4:invalid")).toThrow(
    "Invalid GPU count: invalid. Value must be a positive integer.",
  );
  expect(() => parseGpuConfig("T4:")).toThrow(
    "Invalid GPU count: . Value must be a positive integer.",
  );
  expect(() => parseGpuConfig("T4:0")).toThrow(
    "Invalid GPU count: 0. Value must be a positive integer.",
  );
  expect(() => parseGpuConfig("T4:-1")).toThrow(
    "Invalid GPU count: -1. Value must be a positive integer.",
  );
});

test("SandboxWithVolume", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const volume = await tc.volumes.fromName("libmodal-test-sandbox-volume", {
    createIfMissing: true,
  });

  const sb = await tc.sandboxes.create(app, image, {
    command: ["echo", "volume test"],
    volumes: { "/mnt/test": volume },
  });
  onTestFinished(async () => await sb.terminate());

  expect(sb.sandboxId).toMatch(/^sb-/);

  const exitCode = await sb.wait();
  expect(exitCode).toBe(0);
});

test("SandboxWithReadOnlyVolume", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const volume = await tc.volumes.fromName("libmodal-test-sandbox-volume", {
    createIfMissing: true,
  });

  const readOnlyVolume = volume.withMountOptions({ readOnly: true });

  const sb = await tc.sandboxes.create(app, image, {
    command: ["sh", "-c", "echo 'test' > /mnt/test/test.txt"],
    volumes: { "/mnt/test": readOnlyVolume },
  });
  onTestFinished(async () => await sb.terminate());

  expect(await sb.wait()).toBe(1);
  expect(await sb.stderr.readText()).toContain("Read-only file system");
});

test("SandboxWithSubPathVolume", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const volume = await tc.volumes.ephemeral();
  onTestFinished(() => volume.closeEphemeral());

  const subPath = "/scoped";
  const subPathVolume = volume.withMountOptions({ subPath });

  // Write a marker file into the sub-path-mounted volume.
  const writer = await tc.sandboxes.create(app, image, {
    command: ["sh", "-c", "echo subpath-works > /mnt/sub/marker.txt"],
    volumes: { "/mnt/sub": subPathVolume },
  });
  onTestFinished(async () => await writer.terminate());
  expect(await writer.wait()).toBe(0);

  // Mount the same volume at the root and verify the file landed under the sub-path.
  const reader = await tc.sandboxes.create(app, image, {
    command: ["cat", "/mnt/full/scoped/marker.txt"],
    volumes: { "/mnt/full": volume },
  });
  onTestFinished(async () => await reader.terminate());
  expect(await reader.wait()).toBe(0);
  expect((await reader.stdout.readText()).trim()).toBe("subpath-works");
});

test("SandboxReloadVolumes", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const volume = await tc.volumes.ephemeral();
  onTestFinished(() => volume.closeEphemeral());

  const ls = async (sb: Sandbox): Promise<string[]> => {
    const p = await sb.exec(["ls", "-1", "/volume"]);
    const out = await p.stdout.readText();
    expect(await p.wait()).toBe(0);
    return out
      .split(/\s+/)
      .filter((name) => name)
      .sort();
  };

  const sb1 = await tc.sandboxes.create(app, image, {
    command: ["sleep", "120"],
    volumes: { "/volume": volume },
  });
  onTestFinished(async () => await sb1.terminate());
  const sb2 = await tc.sandboxes.create(app, image, {
    command: ["sleep", "120"],
    volumes: { "/volume": volume },
  });
  onTestFinished(async () => await sb2.terminate());

  expect(await ls(sb1)).toEqual([]);
  expect(await ls(sb2)).toEqual([]);

  // Touch a file from sb1.
  const touch = await sb1.exec(["touch", "/volume/test.txt"]);
  expect(await touch.wait()).toBe(0);
  expect(await ls(sb1)).toEqual(["test.txt"]);
  expect(await ls(sb2)).toEqual([]);

  // Reloading sb1 commits its write to the volume; sb2 is unaffected
  // until it reloads too. The reload is synchronous, so the committed
  // state is observable as soon as the call returns.
  await sb1.reloadVolumes();
  expect(await ls(sb1)).toEqual(["test.txt"]);
  expect(await ls(sb2)).toEqual([]);

  // sb2 sees the file only after it reloads.
  await sb2.reloadVolumes();
  expect(await ls(sb1)).toEqual(["test.txt"]);
  expect(await ls(sb2)).toEqual(["test.txt"]);
});

test("SandboxWithTunnels", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image, {
    command: ["cat"],
    encryptedPorts: [8443],
    unencryptedPorts: [8080],
  });
  onTestFinished(async () => await sb.terminate());

  expect(sb.sandboxId).toMatch(/^sb-/);

  const tunnels = await sb.tunnels();
  expect(Object.keys(tunnels)).toHaveLength(2);

  // Test encrypted tunnel (port 8443)
  const encryptedTunnel = tunnels[8443];
  expect(encryptedTunnel.host).toMatch(/\.modal\.host$/);
  expect(encryptedTunnel.port).toBe(443);
  expect(encryptedTunnel.url).toMatch(/^https:\/\//);
  expect(encryptedTunnel.tlsSocket).toEqual([
    encryptedTunnel.host,
    encryptedTunnel.port,
  ]);

  // Test unencrypted tunnel (port 8080)
  const unencryptedTunnel = tunnels[8080];
  expect(unencryptedTunnel.unencryptedHost).toMatch(/\.modal\.host$/);
  expect(typeof unencryptedTunnel.unencryptedPort).toBe("number");
  expect(unencryptedTunnel.tcpSocket).toEqual([
    unencryptedTunnel.unencryptedHost,
    unencryptedTunnel.unencryptedPort,
  ]);
});

test("CreateSandboxWithSecrets", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const secret = await tc.secrets.fromName("libmodal-test-secret", {
    requiredKeys: ["c"],
  });

  const sb = await tc.sandboxes.create(app, image, {
    command: ["printenv", "c"],
    secrets: [secret],
  });
  onTestFinished(async () => await sb.terminate());

  const result = await sb.stdout.readText();
  expect(result).toBe("hello world\n");
});

test("CreateSandboxWithNetworkAccessParams", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image, {
    command: ["echo", "hello, network access"],
    blockNetwork: false,
    outboundCidrAllowlist: ["10.0.0.0/8", "192.168.0.0/16"],
  });
  onTestFinished(async () => await sb.terminate());

  expect(sb.sandboxId).toMatch(/^sb-/);

  const exitCode = await sb.wait();
  expect(exitCode).toBe(0);

  await expect(
    tc.sandboxes.create(app, image, {
      blockNetwork: false,
      outboundCidrAllowlist: ["not-an-ip/8"],
    }),
  ).rejects.toThrow("Invalid CIDR: not-an-ip/8");

  await expect(
    tc.sandboxes.create(app, image, {
      blockNetwork: true,
      outboundCidrAllowlist: ["10.0.0.0/8"],
    }),
  ).rejects.toThrow(
    "outboundCidrAllowlist cannot be used when blockNetwork is enabled",
  );
});

test("CreateSandboxWithInboundCidrAllowlist", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  // Verify proto is correctly populated.
  const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
    inboundCidrAllowlist: ["10.0.0.0/8", "192.168.0.0/16"],
  });
  expect(req.definition?.inboundCidrAllowlist).toEqual([
    "10.0.0.0/8",
    "192.168.0.0/16",
  ]);

  // Default: empty list (all IPs allowed).
  const req2 = await buildSandboxCreateRequestProto("app-123", "img-456", {});
  expect(req2.definition?.inboundCidrAllowlist).toEqual([]);

  // Cannot be combined with blockNetwork.
  await expect(
    buildSandboxCreateRequestProto("app-123", "img-456", {
      blockNetwork: true,
      inboundCidrAllowlist: ["10.0.0.0/8"],
    }),
  ).rejects.toThrow(
    "inboundCidrAllowlist cannot be used when blockNetwork is enabled",
  );

  // End-to-end: sandbox is created successfully with the param.
  const sb = await tc.sandboxes.create(app, image, {
    command: ["echo", "hello, inbound cidrs"],
    inboundCidrAllowlist: ["10.0.0.0/8"],
  });
  onTestFinished(async () => await sb.terminate());
  expect(await sb.wait()).toBe(0);
});

test("CreateSandboxWithDomainAllowlist", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  // Domain-only allowlist: ALLOWLIST with allowedDomains populated.
  const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
    outboundDomainAllowlist: ["example.com", "*.github.com"],
  });
  expect(req.definition?.networkAccess?.networkAccessType).toBe(
    NetworkAccess_NetworkAccessType.ALLOWLIST,
  );
  expect(req.definition?.networkAccess?.allowedDomains).toEqual([
    "example.com",
    "*.github.com",
  ]);
  expect(req.definition?.networkAccess?.allowedCidrs).toEqual([]);

  // Domain + CIDR combined: both lists are populated.
  const req2 = await buildSandboxCreateRequestProto("app-123", "img-456", {
    outboundDomainAllowlist: ["api.example.com"],
    outboundCidrAllowlist: ["10.0.0.0/8"],
  });
  expect(req2.definition?.networkAccess?.networkAccessType).toBe(
    NetworkAccess_NetworkAccessType.ALLOWLIST,
  );
  expect(req2.definition?.networkAccess?.allowedDomains).toEqual([
    "api.example.com",
  ]);
  expect(req2.definition?.networkAccess?.allowedCidrs).toEqual(["10.0.0.0/8"]);

  // Cannot be combined with blockNetwork.
  await expect(
    buildSandboxCreateRequestProto("app-123", "img-456", {
      blockNetwork: true,
      outboundDomainAllowlist: ["example.com"],
    }),
  ).rejects.toThrow(
    "outboundDomainAllowlist cannot be used when blockNetwork is enabled",
  );

  // Invalid domain triggers server-side validation error.
  await expect(
    tc.sandboxes.create(app, image, {
      outboundDomainAllowlist: ["not a valid domain!"],
    }),
  ).rejects.toThrow();

  // End-to-end: sandbox is created successfully with the param.
  const sb = await tc.sandboxes.create(app, image, {
    command: ["echo", "hello, domain allowlist"],
    outboundDomainAllowlist: ["example.com", "*.github.com"],
  });
  onTestFinished(async () => await sb.terminate());
  expect(await sb.wait()).toBe(0);
});

test("buildSandboxCreateRequestProto rejects removed cidrAllowlist", async () => {
  // The deprecated `cidrAllowlist` must throw rather than be silently
  // ignored, which would downgrade ALLOWLIST to OPEN network access.
  await expect(
    buildSandboxCreateRequestProto("app-123", "img-456", {
      cidrAllowlist: ["10.0.0.0/8"],
    } as any),
  ).rejects.toThrow(
    "Parameter 'cidrAllowlist' has been renamed to 'outboundCidrAllowlist'.",
  );
});

test("buildSandboxCreateRequestProto sets i6pn", async () => {
  const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
    i6pn: true,
  });
  expect(req.definition?.i6pnEnabled).toBe(true);

  const req2 = await buildSandboxCreateRequestProto("app-123", "img-456", {});
  expect(req2.definition?.i6pnEnabled).toBe(false);

  await expect(
    buildSandboxCreateRequestProto("app-123", "img-456", {
      blockNetwork: true,
      i6pn: true,
    }),
  ).rejects.toThrow("blockNetwork disables all networking, including i6pn");
});

test("buildOutboundNetworkAccess sidecar rules", () => {
  // No allowlist (the sidecar default) maps to open access, independent of the
  // main container. Sidecars never pass blockNetwork=true.
  const open = buildOutboundNetworkAccess(false, undefined, undefined);
  expect(open.networkAccessType).toBe(NetworkAccess_NetworkAccessType.OPEN);

  // CIDR allowlist.
  const cidr = buildOutboundNetworkAccess(false, ["10.0.0.0/8"], undefined);
  expect(cidr.networkAccessType).toBe(
    NetworkAccess_NetworkAccessType.ALLOWLIST,
  );
  expect(cidr.allowedCidrs).toEqual(["10.0.0.0/8"]);
  expect(cidr.allowedDomains).toEqual([]);

  // Domain allowlist.
  const domain = buildOutboundNetworkAccess(false, undefined, [
    "*.example.com",
  ]);
  expect(domain.networkAccessType).toBe(
    NetworkAccess_NetworkAccessType.ALLOWLIST,
  );
  expect(domain.allowedDomains).toEqual(["*.example.com"]);
  expect(domain.allowedCidrs).toEqual([]);

  // An empty allowlist blocks external egress while keeping main-container
  // connectivity; it must remain ALLOWLIST, not BLOCKED.
  const empty = buildOutboundNetworkAccess(false, [], undefined);
  expect(empty.networkAccessType).toBe(
    NetworkAccess_NetworkAccessType.ALLOWLIST,
  );
  expect(empty.allowedCidrs).toEqual([]);
  expect(empty.allowedDomains).toEqual([]);
});

test("SandboxPollAndReturnCode", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image, { command: ["cat"] });
  onTestFinished(async () => await sb.terminate());

  expect(await sb.poll()).toBeNull();

  // Send input to make the cat command complete
  await sb.stdin.writeText("hello, Sandbox");
  await sb.stdin.close();

  expect(await sb.wait()).toBe(0);
  expect(await sb.poll()).toBe(0);
});

test("SandboxPollAfterFailure", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image, {
    command: ["sh", "-c", "exit 42"],
  });
  onTestFinished(async () => await sb.terminate());

  expect(await sb.wait()).toBe(42);
  expect(await sb.poll()).toBe(42);
});

test("SandboxExecSecret", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const secret = await tc.secrets.fromName("libmodal-test-secret", {
    requiredKeys: ["c"],
  });
  const secret2 = await tc.secrets.fromObject({ d: "3" });
  const printSecret = await sb.exec(["printenv", "c", "d"], {
    stdout: "pipe",
    secrets: [secret, secret2],
  });
  const secretText = await printSecret.stdout.readText();
  expect(secretText).toBe("hello world\n3\n");
});

test("SandboxModalIdentityTokenUnsetByDefault", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image, {
    command: ["sh", "-c", "echo ${MODAL_IDENTITY_TOKEN:-UNSET}"],
  });
  onTestFinished(async () => await sb.terminate());

  expect((await sb.stdout.readText()).trim()).toBe("UNSET");
  expect(await sb.wait()).toBe(0);
});

test("SandboxIncludeOidcIdentityTokenSetsModalIdentityTokenEnv", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image, {
    command: ["sh", "-c", "echo ${MODAL_IDENTITY_TOKEN:-UNSET}"],
    includeOidcIdentityToken: true,
  });
  onTestFinished(async () => await sb.terminate());

  const token = (await sb.stdout.readText()).trim();
  expect(token).not.toBe("UNSET");
  expect(token.length).toBeGreaterThan(0);
  expect(await sb.wait()).toBe(0);
});

test("SandboxFromId", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const sbFromId = await tc.sandboxes.fromId(sb.sandboxId);
  expect(sbFromId.sandboxId).toBe(sb.sandboxId);
});

test("SandboxWithWorkdir", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image, {
    command: ["pwd"],
    workdir: "/tmp",
  });
  onTestFinished(async () => await sb.terminate());

  expect(await sb.stdout.readText()).toBe("/tmp\n");
});

test("SandboxWithWorkdirValidation", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  await expect(
    tc.sandboxes.create(app, image, {
      workdir: "relative/path",
    }),
  ).rejects.toThrow("workdir must be an absolute path, got: relative/path");
});

test("SandboxSetTagsAndList", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const unique = `${Math.random()}`;

  const foundBefore: string[] = [];
  for await (const s of tc.sandboxes.list({ tags: { "test-key": unique } })) {
    foundBefore.push(s.sandboxId);
  }
  expect(foundBefore.length).toBe(0);

  await sb.setTags({ "test-key": unique });

  const foundAfter: string[] = [];
  for await (const s of tc.sandboxes.list({ tags: { "test-key": unique } })) {
    foundAfter.push(s.sandboxId);
  }
  expect(foundAfter).toEqual([sb.sandboxId]);
});

test("SandboxSetMultipleTagsAndList", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const tagA = `A-${Math.random()}`;
  const tagB = `B-${Math.random()}`;
  const tagC = `C-${Math.random()}`;

  expect(await sb.getTags()).toEqual({});

  await sb.setTags({ "key-a": tagA, "key-b": tagB, "key-c": tagC });

  expect(await sb.getTags()).toEqual({
    "key-a": tagA,
    "key-b": tagB,
    "key-c": tagC,
  });

  let ids: string[] = [];
  for await (const s of tc.sandboxes.list({ tags: { "key-a": tagA } })) {
    ids.push(s.sandboxId);
  }
  expect(ids).toEqual([sb.sandboxId]);

  ids = [];
  for await (const s of tc.sandboxes.list({
    tags: { "key-a": tagA, "key-b": tagB },
  })) {
    ids.push(s.sandboxId);
  }
  expect(ids).toEqual([sb.sandboxId]);

  ids = [];
  for await (const s of tc.sandboxes.list({
    tags: { "key-a": tagA, "key-b": tagB, "key-d": "not-set" },
  })) {
    ids.push(s.sandboxId);
  }
  expect(ids.length).toBe(0);
});

test("SandboxListByAppId", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  let count = 0;
  for await (const s of tc.sandboxes.list({ appId: app.appId })) {
    expect(s.sandboxId).toMatch(/^sb-/);
    count++;
    if (count > 0) break;
  }
  expect(count).toBeGreaterThan(0);
});

test("NamedSandbox", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sandboxName = `test-sandbox-${Math.random().toString().substring(2, 10)}`;

  const sb = await tc.sandboxes.create(app, image, {
    name: sandboxName,
    command: ["sleep", "60"],
  });
  onTestFinished(async () => await sb.terminate());

  const sb1FromName = await tc.sandboxes.fromName("libmodal-test", sandboxName);
  expect(sb1FromName.sandboxId).toBe(sb.sandboxId);
  const sb2FromName = await tc.sandboxes.fromName("libmodal-test", sandboxName);
  expect(sb2FromName.sandboxId).toBe(sb1FromName.sandboxId);

  await expect(
    tc.sandboxes.create(app, image, {
      name: sandboxName,
      command: ["sleep", "60"],
    }),
  ).rejects.toThrow("already exists");
});

test("NamedSandboxNotFound", async () => {
  await expect(
    tc.sandboxes.fromName("libmodal-test", "non-existent-sandbox"),
  ).rejects.toThrow("not found");
});

test("buildSandboxCreateRequestProto without PTY", async () => {
  const req = await buildSandboxCreateRequestProto("app-123", "img-456");

  const definition = req.definition!;
  expect(definition.ptyInfo).toBeUndefined();
});

test("buildSandboxCreateRequestProto with PTY", async () => {
  const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
    pty: true,
  });

  const definition = req.definition!;
  const ptyInfo = definition.ptyInfo!;
  expect(ptyInfo.enabled).toBe(true);
  expect(ptyInfo.winszRows).toBe(24);
  expect(ptyInfo.winszCols).toBe(80);
  expect(ptyInfo.envTerm).toBe("xterm-256color");
  expect(ptyInfo.envColorterm).toBe("truecolor");
  expect(ptyInfo.ptyType).toBe(PTYInfo_PTYType.PTY_TYPE_SHELL);
});

test("Probe.withTcp invalid values", () => {
  expect(() => Probe.withTcp("8080" as any)).toThrow("expects an integer");
  expect(() => Probe.withTcp(0)).toThrow("expects `port` in [1, 65535]");
  expect(() => Probe.withTcp(65536)).toThrow("expects `port` in [1, 65535]");
  expect(() => Probe.withTcp(8080, { intervalMs: "100" as any })).toThrow(
    "expects an integer `intervalMs`",
  );
  expect(() => Probe.withTcp(8080, { intervalMs: 0 })).toThrow(
    "expects `intervalMs` > 0",
  );
});

test("Probe.withExec invalid values", () => {
  expect(() => Probe.withExec([])).toThrow("requires at least one argument");
  expect(() => Probe.withExec(["echo", 1 as any])).toThrow(
    "expects all arguments to be strings",
  );
  expect(() => Probe.withExec(["echo"], { intervalMs: "100" as any })).toThrow(
    "expects an integer `intervalMs`",
  );
  expect(() => Probe.withExec(["echo"], { intervalMs: 0 })).toThrow(
    "expects `intervalMs` > 0",
  );
});

test("buildSandboxCreateRequestProto with TCP readiness probe", async () => {
  const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
    readinessProbe: Probe.withTcp(8080, { intervalMs: 250 }),
  });
  expect(req.definition?.readinessProbe?.tcpPort).toBe(8080);
  expect(req.definition?.readinessProbe?.intervalMs).toBe(250);
});

test("buildSandboxCreateRequestProto with exec readiness probe", async () => {
  const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
    readinessProbe: Probe.withExec(["sh", "-c", "echo ok"], {
      intervalMs: 300,
    }),
  });
  expect(req.definition?.readinessProbe?.execCommand?.argv).toEqual([
    "sh",
    "-c",
    "echo ok",
  ]);
  expect(req.definition?.readinessProbe?.intervalMs).toBe(300);
});

test("buildSandboxCreateRequestProto with CPU and CPULimit", async () => {
  const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
    cpu: 2.0,
    cpuLimit: 4.5,
  });

  const resources = req.definition!.resources!;
  expect(resources.milliCpu).toBe(2000);
  expect(resources.milliCpuMax).toBe(4500);
});

test("buildSandboxCreateRequestProto CPULimit lower than CPU", async () => {
  await expect(
    buildSandboxCreateRequestProto("app-123", "img-456", {
      cpu: 4.0,
      cpuLimit: 2.0,
    }),
  ).rejects.toThrow("cpu (4) cannot be higher than cpuLimit (2)");
});

test("buildSandboxCreateRequestProto CPULimit without CPU", async () => {
  await expect(
    buildSandboxCreateRequestProto("app-123", "img-456", {
      cpuLimit: 4.0,
    }),
  ).rejects.toThrow("must also specify cpu when cpuLimit is specified");
});

test("buildSandboxCreateRequestProto with Memory and MemoryLimit", async () => {
  const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
    memoryMiB: 1024,
    memoryLimitMiB: 2048,
  });

  const resources = req.definition!.resources!;
  expect(resources.memoryMb).toBe(1024);
  expect(resources.memoryMbMax).toBe(2048);
});

test("buildSandboxCreateRequestProto MemoryLimit lower than Memory", async () => {
  await expect(
    buildSandboxCreateRequestProto("app-123", "img-456", {
      memoryMiB: 2048,
      memoryLimitMiB: 1024,
    }),
  ).rejects.toThrow(
    "the memoryMiB request (2048) cannot be higher than memoryLimitMiB (1024)",
  );
});

test("buildSandboxCreateRequestProto MemoryLimit without Memory", async () => {
  await expect(
    buildSandboxCreateRequestProto("app-123", "img-456", {
      memoryLimitMiB: 2048,
    }),
  ).rejects.toThrow(
    "must also specify memoryMiB when memoryLimitMiB is specified",
  );
});

test("buildSandboxCreateRequestProto negative CPU", async () => {
  await expect(
    buildSandboxCreateRequestProto("app-123", "img-456", {
      cpu: -1.0,
    }),
  ).rejects.toThrow("must be a positive number");
});

test("buildSandboxCreateRequestProto negative Memory", async () => {
  await expect(
    buildSandboxCreateRequestProto("app-123", "img-456", {
      memoryMiB: -100,
    }),
  ).rejects.toThrow("must be a positive number");
});

test("buildSandboxCreateRequestProto includeOidcIdentityToken", async () => {
  const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
    includeOidcIdentityToken: true,
  });
  expect(req.definition!.includeOidcIdentityToken).toBe(true);
});

test("buildSandboxCreateRequestProto with tags", async () => {
  const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
    tags: { env: "prod", team: "infra" },
  });
  const got: Record<string, string> = {};
  for (const tag of req.tags) {
    got[tag.tagName] = tag.tagValue;
  }
  expect(got).toEqual({ env: "prod", team: "infra" });
});

test("ConnectToken", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("python:3.12-alpine");

  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => {
    await sb.terminate();
  });

  const creds = await sb.createConnectToken({ userMetadata: "abc" });
  expect(creds.token).toBeTruthy();
  expect(creds.url).toBeTruthy();
});

test("createConnectToken sends port", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxCreateConnectToken", (req: any) => {
    expect(req.sandboxId).toBe(V1_SANDBOX_ID);
    expect(req.port).toBe(9000);
    return { token: "token-9000" };
  });

  const sb = await mc.sandboxes.fromId(V1_SANDBOX_ID);
  const creds = await sb.createConnectToken({ port: 9000 });
  expect(creds.token).toBe("token-9000");

  mock.assertExhausted();
});

test("createConnectToken defaults to port 8080", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxCreateConnectToken", (req: any) => {
    expect(req.port).toBe(8080);
    return { token: "token" };
  });

  const sb = await mc.sandboxes.fromId(V1_SANDBOX_ID);
  const creds = await sb.createConnectToken();
  expect(creds.token).toBe("token");

  mock.assertExhausted();
});

test.each([0, -1, 65536, 8080.5, NaN])(
  "createConnectToken rejects invalid port %s",
  async (port) => {
    const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

    const sb = await mc.sandboxes.fromId(V1_SANDBOX_ID);
    await expect(sb.createConnectToken({ port })).rejects.toThrow(
      "expects `port` in [1, 65535]",
    );

    mock.assertExhausted();
  },
);

test("createConnectToken routes V2 sandboxes to the V2 RPC", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxCreateConnectTokenV2", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    expect(req.userMetadata).toBe("abc");
    expect(req.port).toBe(9000);
    return {
      url: "https://sandbox.modal.host/connect/v2",
      token: "v2token-9000",
    };
  });

  const sb = new Sandbox(mc, V2_SANDBOX_ID, {
    taskId: "ta-v2-123",
  });
  const creds = await sb.createConnectToken({
    userMetadata: "abc",
    port: 9000,
  });
  expect(creds.url).toBe("https://sandbox.modal.host/connect/v2");
  expect(creds.token).toBe("v2token-9000");

  mock.assertExhausted();
});

test("buildSandboxCreateRequestProto_defaults", async () => {
  const req = await buildSandboxCreateRequestProto("app-123", "img-456");
  const def = req.definition!;

  expect(def.timeoutSecs).toBe(300);
  expect(def.entrypointArgs).toEqual([]);
  expect(def.networkAccess?.networkAccessType).toBe(
    NetworkAccess_NetworkAccessType.OPEN,
  );
  expect(def.networkAccess?.allowedCidrs).toEqual([]);
  expect(def.verbose).toBe(false);
  expect(def.cloudProviderStr).toBe("");
  expect(def.resources?.milliCpu).toBe(0);
  expect(def.resources?.memoryMb).toBe(0);
  expect(def.ptyInfo).toBeUndefined();
  expect(def.idleTimeoutSecs).toBeUndefined();
  expect(def.workdir).toBeUndefined();
  expect(def.schedulerPlacement).toBeUndefined();
  expect(def.proxyId).toBeUndefined();
  expect(def.volumeMounts).toEqual([]);
  expect(def.cloudBucketMounts).toEqual([]);
  expect(def.secretIds).toEqual([]);
  expect(def.openPorts?.ports).toEqual([]);
  expect(def.name).toBeUndefined();
  expect(def.includeOidcIdentityToken).toBe(false);
  expect(req.tags).toEqual([]);
});

test("sandboxInvalidTimeouts", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  await expect(
    tc.sandboxes.create(app, image, { timeoutMs: 0 }),
  ).rejects.toThrow(/timeoutMs must be positive/);

  await expect(
    tc.sandboxes.create(app, image, { timeoutMs: -1000 }),
  ).rejects.toThrow(/timeoutMs must be positive/);

  await expect(
    tc.sandboxes.create(app, image, { timeoutMs: 1500 }),
  ).rejects.toThrow(/timeoutMs must be a multiple of 1000ms/);

  await expect(
    tc.sandboxes.create(app, image, { idleTimeoutMs: 0 }),
  ).rejects.toThrow(/idleTimeoutMs must be positive/);

  await expect(
    tc.sandboxes.create(app, image, { idleTimeoutMs: -2000 }),
  ).rejects.toThrow(/idleTimeoutMs must be positive/);

  await expect(
    tc.sandboxes.create(app, image, { idleTimeoutMs: 2500 }),
  ).rejects.toThrow(/idleTimeoutMs must be a multiple of 1000ms/);

  const sandbox = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sandbox.terminate());

  await expect(
    sandbox.exec(["echo", "test"], { timeoutMs: 0 }),
  ).rejects.toThrow(/timeoutMs must be positive/);

  await expect(
    sandbox.exec(["echo", "test"], { timeoutMs: -5000 }),
  ).rejects.toThrow(/timeoutMs must be positive/);

  await expect(
    sandbox.exec(["echo", "test"], { timeoutMs: 1500 }),
  ).rejects.toThrow(/timeoutMs must be a multiple of 1000ms/);
});

test("testSandboxExperimentalDocker", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  // With experimental option should include /var/lib/docker
  const sb = await tc.sandboxes.create(app, image, {
    experimentalOptions: { enable_docker: true },
  });
  onTestFinished(async () => {
    await sb.terminate();
  });

  const p = await sb.exec(["test", "-d", "/var/lib/docker"]);
  expect(await p.wait()).toBe(0);

  // Without experimental option should **not** include /var/lib/docker
  const sbDefault = await tc.sandboxes.create(app, image);
  onTestFinished(async () => {
    await sbDefault.terminate();
  });
  const pDefault = await sbDefault.exec(["test", "-d", "/var/lib/docker"]);
  expect(await pDefault.wait()).toBe(1);
});

test("testSandboxExperimentalDockerNotBool", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  await expect(
    tc.sandboxes.create(app, image, {
      experimentalOptions: { enable_docker: 42 },
    }),
  ).rejects.toThrow("must be a boolean or string");
});

test("testSandboxExperimentalDockerMock", async () => {
  const origImageBuilderVersion = process.env["MODAL_IMAGE_BUILDER_VERSION"];
  delete process.env["MODAL_IMAGE_BUILDER_VERSION"];
  onTestFinished(() => {
    if (origImageBuilderVersion !== undefined) {
      process.env["MODAL_IMAGE_BUILDER_VERSION"] = origImageBuilderVersion;
    } else {
      delete process.env["MODAL_IMAGE_BUILDER_VERSION"];
    }
  });
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  const options = { enable_docker: true };
  mock.handleUnary("/SandboxCreate", (req: any): SandboxCreateResponse => {
    expect(req.definition?.experimentalOptionsV2).toMatchObject({
      enable_docker: "true",
    });
    return {
      sandboxId: "sb-1234",
      metadata: { result: undefined, appId: "app-123" },
    };
  });

  mock.handleUnary("/AppGetOrCreate", (_: any): AppGetOrCreateResponse => {
    return AppGetOrCreateResponse.create({ appId: "ap-1234" });
  });

  const app = await mc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });

  mock.handleUnary("ImageGetOrCreate", (_: any): ImageGetOrCreateResponse => {
    return {
      imageId: "im-123",
      result: {
        status: GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS,
        exception: "",
        exitcode: 0,
        traceback: "",
        serializedTb: new Uint8Array(0),
        tbLineCache: new Uint8Array(0),
        propagationReason: "",
      },
      metadata: undefined,
    };
  });

  mock.handleUnary("/EnvironmentGetOrCreate", () => {
    return {
      environmentId: "en-main-123",
      metadata: {
        name: "main",
        settings: {
          imageBuilderVersion: "2025.06",
          webhookSuffix: "modal.run",
        },
      },
    };
  });

  const image = mc.images.fromRegistry("alpine:3.21");

  const sb = await mc.sandboxes.create(app, image, {
    experimentalOptions: options,
  });
  expect(sb.sandboxId).toEqual("sb-1234");

  mock.assertExhausted();
});

test("create deduces V2 from the returned Sandbox ID shape", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/AppGetOrCreate", (): AppGetOrCreateResponse => {
    return AppGetOrCreateResponse.create({ appId: "ap-1234" });
  });
  mock.handleUnary("/EnvironmentGetOrCreate", () => {
    return {
      environmentId: "en-main-123",
      metadata: {
        name: "main",
        settings: {
          imageBuilderVersion: "2025.06",
          webhookSuffix: "modal.run",
        },
      },
    };
  });
  mock.handleUnary("/ImageGetOrCreate", (): ImageGetOrCreateResponse => {
    return {
      imageId: "im-123",
      result: {
        status: GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS,
        exception: "",
        exitcode: 0,
        traceback: "",
        serializedTb: new Uint8Array(0),
        tbLineCache: new Uint8Array(0),
        propagationReason: "",
      },
      metadata: undefined,
    };
  });
  mock.handleUnary("/SandboxCreate", (): SandboxCreateResponse => {
    return {
      sandboxId: V2_SANDBOX_ID,
      metadata: { result: undefined, appId: "ap-1234" },
    };
  });
  mock.handleUnary("/SandboxTerminateV2", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    return {};
  });

  const app = await mc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = mc.images.fromRegistry("alpine:3.21");
  const sb = await mc.sandboxes.create(app, image);
  expect(sb.sandboxId).toBe(V2_SANDBOX_ID);
  await sb.terminate();

  mock.assertExhausted();
});

test("fromName deduces V2 from the returned Sandbox ID shape", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetFromName", (req: any) => {
    expect(req.sandboxName).toBe("my-sandbox");
    return { sandboxId: V2_SANDBOX_ID };
  });
  mock.handleUnary("/SandboxTerminateV2", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    return {};
  });

  const sb = await mc.sandboxes.fromName("libmodal-test", "my-sandbox");
  await sb.terminate();

  mock.assertExhausted();
});

test("experimentalList routes mirrored V1 sandboxes to the V1 backend", async () => {
  // V1 Sandboxes are mirrored into the V2 store during the V1->V2 migration,
  // so SandboxListV2 can return V1 Sandboxes.
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxListV2", () => {
    return { sandboxes: [{ id: V1_SANDBOX_ID, createdAt: 1 }] };
  });
  mock.handleUnary("/SandboxListV2", () => {
    return { sandboxes: [] };
  });
  mock.handleUnary("/SandboxTerminate", (req: any) => {
    expect(req.sandboxId).toBe(V1_SANDBOX_ID);
    return {};
  });

  const sandboxes = [];
  for await (const sb of mc.sandboxes.experimentalList({ appId: "ap-123" })) {
    sandboxes.push(sb);
  }
  expect(sandboxes).toHaveLength(1);
  await sandboxes[0].terminate();

  mock.assertExhausted();
});

test("experimentalFromName routes a mirrored V1 sandbox to the V1 backend", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetFromNameV2", () => {
    return { sandboxId: V1_SANDBOX_ID };
  });
  mock.handleUnary("/SandboxTerminate", (req: any) => {
    expect(req.sandboxId).toBe(V1_SANDBOX_ID);
    return {};
  });

  const sb = await mc.sandboxes.experimentalFromName(
    "libmodal-test",
    "mirrored-v1",
  );
  await sb.terminate();

  mock.assertExhausted();
});

test("list deduces V2 from returned Sandbox ID shapes", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxList", () => {
    return { sandboxes: [{ id: V2_SANDBOX_ID, createdAt: 1 }] };
  });
  mock.handleUnary("/SandboxList", () => {
    return { sandboxes: [] };
  });
  mock.handleUnary("/SandboxTerminateV2", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    return {};
  });

  const sandboxes = [];
  for await (const sb of mc.sandboxes.list()) {
    sandboxes.push(sb);
  }
  expect(sandboxes).toHaveLength(1);
  await sandboxes[0].terminate();

  mock.assertExhausted();
});

test("testSandboxExperimentalOptionsAcceptsStringValues", async () => {
  const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
    experimentalOptions: { proxy_traffic_via_sidecar: "mitm" },
  });

  expect(req.definition?.experimentalOptionsV2).toEqual({
    proxy_traffic_via_sidecar: "mitm",
  });
});

test("testSandboxExperimentalOptionsAcceptsBoolValues", async () => {
  const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
    experimentalOptions: { enable_docker: true },
  });

  expect(req.definition?.experimentalOptionsV2).toEqual({
    enable_docker: "true",
  });
});

test("testSandboxExperimentalOptionsAcceptsMixedStringAndBoolValues", async () => {
  const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
    experimentalOptions: {
      enable_docker: true,
      proxy_traffic_via_sidecar: "mitm",
    },
  });

  expect(req.definition?.experimentalOptionsV2).toEqual({
    enable_docker: "true",
    proxy_traffic_via_sidecar: "mitm",
  });
});

test("buildSandboxCreateV2RequestProto", async () => {
  const req = await buildSandboxCreateV2RequestProto("app-123", "img-456", {
    command: ["sleep", "60"],
    timeoutMs: 600_000,
  });

  expect(req.appId).toBe("app-123");
  expect(req.definition?.imageId).toBe("img-456");
  expect(req.definition?.entrypointArgs).toEqual(["sleep", "60"]);
  expect(req.definition?.timeoutSecs).toBe(600);
});

test("buildSandboxCreateV2RequestProto supports a proxy", async () => {
  const req = await buildSandboxCreateV2RequestProto("app-123", "img-456", {
    proxy: { proxyId: "pr-123" } as any,
  });

  expect(req.definition?.proxyId).toBe("pr-123");
});

test("buildSandboxCreateV2RequestProto supports tags", async () => {
  const req = await buildSandboxCreateV2RequestProto("app-123", "img-456", {
    tags: { env: "prod", team: "infra" },
  });

  const got: Record<string, string> = {};
  for (const tag of req.tags) {
    got[tag.tagName] = tag.tagValue;
  }
  expect(got).toEqual({ env: "prod", team: "infra" });
});

test("buildSandboxCreateV2RequestProto supports experimental options", async () => {
  const req = await buildSandboxCreateV2RequestProto("app-123", "img-456", {
    experimentalOptions: { enable_docker: true },
  });

  expect(req.definition?.experimentalOptionsV2).toMatchObject({
    enable_docker: "true",
  });
});

test("buildSandboxCreateV2RequestProto rejects non-boolean experimental options", async () => {
  await expect(
    buildSandboxCreateV2RequestProto("app-123", "img-456", {
      experimentalOptions: { enable_docker: 42 as any },
    }),
  ).rejects.toThrow("must be a boolean or string");
});

test.each([["gpu", { gpu: "A10G" }, "GPUs are not supported"]])(
  "buildSandboxCreateV2RequestProto rejects unsupported option %s",
  async (_name, params, expectedError) => {
    await expect(
      buildSandboxCreateV2RequestProto("app-123", "img-456", params),
    ).rejects.toThrow(expectedError);
  },
);

test("buildSandboxCreateV2RequestProto supports custom domains", async () => {
  const req = await buildSandboxCreateV2RequestProto("app-123", "img-456", {
    customDomain: "sandboxes.example.com",
  });
  expect(req.definition?.customDomain).toBe("sandboxes.example.com");
});

test("buildSandboxCreateV2RequestProto supports volumes and cloud bucket mounts", async () => {
  const cbm = tc.cloudBucketMounts.create("my-bucket");
  const req = await buildSandboxCreateV2RequestProto("app-123", "img-456", {
    volumes: { "/mnt/vol": { volumeId: "vo-123" } as any },
    cloudBucketMounts: { "/mnt/s3": cbm },
  });

  expect(req.definition?.volumeMounts).toHaveLength(1);
  expect(req.definition?.volumeMounts?.[0].mountPath).toBe("/mnt/vol");
  expect(req.definition?.volumeMounts?.[0].volumeId).toBe("vo-123");

  expect(req.definition?.cloudBucketMounts).toHaveLength(1);
  expect(req.definition?.cloudBucketMounts?.[0].mountPath).toBe("/mnt/s3");
  expect(req.definition?.cloudBucketMounts?.[0].bucketName).toBe("my-bucket");
});

test("buildSandboxCreateV2RequestProto supports OIDC identity tokens", async () => {
  const cbm = tc.cloudBucketMounts.create("my-bucket", {
    oidcAuthRoleArn: "arn:aws:iam::123:role/r",
  });
  const req = await buildSandboxCreateV2RequestProto("app-123", "img-456", {
    includeOidcIdentityToken: true,
    cloudBucketMounts: { "/mnt/s3": cbm },
  });

  expect(req.definition?.includeOidcIdentityToken).toBe(true);
  expect(req.definition?.cloudBucketMounts?.[0].oidcAuthRoleArn).toBe(
    "arn:aws:iam::123:role/r",
  );
});

test("ExperimentalCreate routes lifecycle calls to V2 RPCs", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/AppGetOrCreate", (_: any): AppGetOrCreateResponse => {
    return AppGetOrCreateResponse.create({ appId: "ap-1234" });
  });
  mock.handleUnary("ImageGetOrCreate", (_: any): ImageGetOrCreateResponse => {
    return {
      imageId: "im-123",
      result: {
        status: GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS,
        exception: "",
        exitcode: 0,
        traceback: "",
        serializedTb: new Uint8Array(0),
        tbLineCache: new Uint8Array(0),
        propagationReason: "",
      },
      metadata: undefined,
    };
  });
  mock.handleUnary("/SandboxCreateV2", (req: any): SandboxCreateV2Response => {
    expect(req.appId).toBe("ap-1234");
    return {
      sandboxId: V2_SANDBOX_ID,
      taskId: "ta-v2-123",
      tunnels: [],
      metadata: { result: undefined, appId: "ap-1234" },
      commandRouterAccess: undefined,
    };
  });
  mock.handleUnary("/SandboxWaitV2", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    return {
      result: {
        status: GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS,
        exitcode: 0,
      },
    };
  });
  mock.handleUnary("/SandboxWaitV2", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    expect(req.timeout).toBe(0);
    return {};
  });
  mock.handleUnary("/SandboxGetTunnelsV2", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    return { tunnels: [] };
  });
  mock.handleUnary("/SandboxTerminateV2", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    return {};
  });
  mock.handleUnary("/EnvironmentGetOrCreate", () => {
    return {
      environmentId: "en-main-123",
      metadata: {
        name: "main",
        settings: {
          imageBuilderVersion: "2025.06",
          webhookSuffix: "modal.run",
        },
      },
    };
  });

  const app = await mc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = mc.images.fromRegistry("alpine:3.21");

  const sb = await mc.sandboxes.experimentalCreate(app, image);
  expect(sb.sandboxId).toBe(V2_SANDBOX_ID);
  expect(await sb.wait()).toBe(0);
  expect(await sb.poll()).toBeNull();
  expect(await sb.tunnels()).toEqual({});
  await sb.terminate();

  mock.assertExhausted();
});

test("ExperimentalCreate caches encrypted-only tunnels from the create response", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/AppGetOrCreate", (_: any): AppGetOrCreateResponse => {
    return AppGetOrCreateResponse.create({ appId: "ap-1234" });
  });
  mock.handleUnary("ImageGetOrCreate", (_: any): ImageGetOrCreateResponse => {
    return {
      imageId: "im-123",
      result: {
        status: GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS,
        exception: "",
        exitcode: 0,
        traceback: "",
        serializedTb: new Uint8Array(0),
        tbLineCache: new Uint8Array(0),
        propagationReason: "",
      },
      metadata: undefined,
    };
  });
  mock.handleUnary("/SandboxCreateV2", (req: any): SandboxCreateV2Response => {
    expect(req.definition?.openPorts?.ports).toHaveLength(1);
    return {
      sandboxId: V2_SANDBOX_ID,
      taskId: "ta-v2-123",
      tunnels: [
        { host: "sb-v2-123-8080.modal.host", port: 443, containerPort: 8080 },
      ],
      metadata: { result: undefined, appId: "ap-1234" },
    } as any;
  });
  mock.handleUnary("/EnvironmentGetOrCreate", () => {
    return {
      environmentId: "en-main-123",
      metadata: {
        name: "main",
        settings: {
          imageBuilderVersion: "2025.06",
          webhookSuffix: "modal.run",
        },
      },
    };
  });

  const app = await mc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = mc.images.fromRegistry("alpine:3.21");

  const sb = await mc.sandboxes.experimentalCreate(app, image, {
    encryptedPorts: [8080],
  });

  const tunnels = await sb.tunnels();
  expect(Object.keys(tunnels)).toHaveLength(1);
  expect(tunnels[8080].host).toBe("sb-v2-123-8080.modal.host");
  expect(tunnels[8080].port).toBe(443);

  mock.assertExhausted();
});

test("ExperimentalCreate uses command router access from the create response", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/AppGetOrCreate", (_: any): AppGetOrCreateResponse => {
    return AppGetOrCreateResponse.create({ appId: "ap-1234" });
  });
  mock.handleUnary("ImageGetOrCreate", (_: any): ImageGetOrCreateResponse => {
    return {
      imageId: "im-123",
      result: {
        status: GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS,
        exception: "",
        exitcode: 0,
        traceback: "",
        serializedTb: new Uint8Array(0),
        tbLineCache: new Uint8Array(0),
        propagationReason: "",
      },
      metadata: undefined,
    };
  });
  mock.handleUnary("/SandboxCreateV2", (_: any): SandboxCreateV2Response => {
    return SandboxCreateV2Response.create({
      sandboxId: V2_SANDBOX_ID,
      taskId: "ta-v2-123",
      metadata: { appId: "ap-1234" },
      commandRouterAccess: {
        url: "https://task-abc123.modal.test",
        jwt: "seeded-jwt",
      },
    });
  });
  mock.handleUnary("/EnvironmentGetOrCreate", () => {
    return {
      environmentId: "en-main-123",
      metadata: {
        name: "main",
        settings: {
          imageBuilderVersion: "2025.06",
          webhookSuffix: "modal.run",
        },
      },
    };
  });

  const setNetworkAccess = vi.fn().mockResolvedValue(undefined);
  // Mock out the task command router so the test doesn't touch real infra.
  const tryInit = vi
    .spyOn(TaskCommandRouterClientImpl, "tryInit")
    .mockResolvedValue({
      setNetworkAccess,
      close: vi.fn(),
    } as unknown as TaskCommandRouterClientImpl);
  onTestFinished(() => tryInit.mockRestore());

  const app = await mc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = mc.images.fromRegistry("alpine:3.21");
  const sb = await mc.sandboxes.experimentalCreate(app, image);

  await sb.updateNetworkPolicy({
    outboundCidrAllowlist: ["10.0.0.0/8"],
    outboundDomainAllowlist: [],
  });

  // tryInit is mocked out here, so this covers the plumbing only: the access from
  // the create response is threaded through to it. The Python tests cover that a
  // seeded access actually suppresses the SandboxGetCommandRouterAccess call.
  expect(tryInit).toHaveBeenCalledTimes(1);
  expect(tryInit).toHaveBeenCalledWith(
    expect.anything(),
    "ta-v2-123",
    V2_SANDBOX_ID,
    true,
    { url: "https://task-abc123.modal.test", jwt: "seeded-jwt" },
    expect.anything(),
    expect.anything(),
  );

  mock.assertExhausted();
});

test("experimentalFromSnapshot uses command router access from the restore response", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxRestoreV2", (): SandboxRestoreV2Response => {
    return SandboxRestoreV2Response.create({
      sandboxId: V2_SANDBOX_ID,
      taskId: "ta-restored-v2-123",
      commandRouterAccess: {
        url: "https://task-abc123.modal.test",
        jwt: "seeded-jwt",
      },
    });
  });

  const setNetworkAccess = vi.fn().mockResolvedValue(undefined);
  const tryInit = vi
    .spyOn(TaskCommandRouterClientImpl, "tryInit")
    .mockResolvedValue({
      setNetworkAccess,
      close: vi.fn(),
    } as unknown as TaskCommandRouterClientImpl);
  onTestFinished(() => tryInit.mockRestore());

  const snapshot = new SandboxSnapshot(mc, "sn-01BX5ZZKBKACTAV9WEVGEMMVRY", {
    isV2: true,
  });
  const sb = await mc.sandboxes.experimentalFromSnapshot(snapshot);

  await sb.updateNetworkPolicy({
    outboundCidrAllowlist: ["10.0.0.0/8"],
    outboundDomainAllowlist: [],
  });

  expect(tryInit).toHaveBeenCalledTimes(1);
  expect(tryInit).toHaveBeenCalledWith(
    expect.anything(),
    "ta-restored-v2-123",
    V2_SANDBOX_ID,
    true,
    { url: "https://task-abc123.modal.test", jwt: "seeded-jwt" },
    expect.anything(),
    expect.anything(),
  );

  mock.assertExhausted();
});

test("ExperimentalCreate fetches unencrypted tunnels missing from the create response", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/AppGetOrCreate", (_: any): AppGetOrCreateResponse => {
    return AppGetOrCreateResponse.create({ appId: "ap-1234" });
  });
  mock.handleUnary("ImageGetOrCreate", (_: any): ImageGetOrCreateResponse => {
    return {
      imageId: "im-123",
      result: {
        status: GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS,
        exception: "",
        exitcode: 0,
        traceback: "",
        serializedTb: new Uint8Array(0),
        tbLineCache: new Uint8Array(0),
        propagationReason: "",
      },
      metadata: undefined,
    };
  });
  mock.handleUnary("/SandboxCreateV2", (req: any): SandboxCreateV2Response => {
    expect(req.definition?.openPorts?.ports).toHaveLength(2);
    return {
      sandboxId: V2_SANDBOX_ID,
      taskId: "ta-v2-123",
      tunnels: [
        { host: "sb-v2-123-8080.modal.host", port: 443, containerPort: 8080 },
      ],
      metadata: { result: undefined, appId: "ap-1234" },
    } as any;
  });
  mock.handleUnary("/SandboxGetTunnelsV2", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    return {
      tunnels: [
        { host: "sb-v2-123-8080.modal.host", port: 443, containerPort: 8080 },
        {
          host: "r1.modal.host",
          port: 443,
          unencryptedHost: "r1.modal.host",
          unencryptedPort: 39000,
          containerPort: 9000,
        },
      ],
    };
  });
  mock.handleUnary("/EnvironmentGetOrCreate", () => {
    return {
      environmentId: "en-main-123",
      metadata: {
        name: "main",
        settings: {
          imageBuilderVersion: "2025.06",
          webhookSuffix: "modal.run",
        },
      },
    };
  });

  const app = await mc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = mc.images.fromRegistry("alpine:3.21");

  const sb = await mc.sandboxes.experimentalCreate(app, image, {
    encryptedPorts: [8080],
    unencryptedPorts: [9000],
  });

  // Unencrypted tunnels are missing from the create response. tunnels()
  // fetches all of them.
  const tunnels = await sb.tunnels();
  expect(Object.keys(tunnels)).toHaveLength(2);
  expect(tunnels[8080].host).toBe("sb-v2-123-8080.modal.host");
  expect(tunnels[9000].unencryptedHost).toBe("r1.modal.host");
  expect(tunnels[9000].unencryptedPort).toBe(39000);

  mock.assertExhausted();
});

test("ExperimentalList yields V2 Sandboxes and paginates", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  // First batch returns a Sandbox; the empty second batch terminates the loop.
  mock.handleUnary("/SandboxListV2", (req: any) => {
    expect(req.appId).toBe("ap-1234");
    expect(req.includeFinished).toBe(false);
    expect(req.beforeTimestamp).toBeFalsy();
    return { sandboxes: [{ id: V2_SANDBOX_ID, createdAt: 100 }] };
  });
  mock.handleUnary("/SandboxListV2", (req: any) => {
    expect(req.appId).toBe("ap-1234");
    expect(req.beforeTimestamp).toBe(100);
    return { sandboxes: [] };
  });

  const ids: string[] = [];
  for await (const sb of mc.sandboxes.experimentalList({ appId: "ap-1234" })) {
    expect(sb.sandboxId).toBe(V2_SANDBOX_ID);
    ids.push(sb.sandboxId);
  }

  expect(ids).toEqual([V2_SANDBOX_ID]);
  mock.assertExhausted();
});

test("ExperimentalList lists environment-scoped without an appId", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  // Without an appId the query is environment-scoped. The empty second batch
  // terminates the loop.
  mock.handleUnary("/SandboxListV2", (req: any) => {
    expect(req.appId).toBeFalsy();
    expect(req.environmentName).toBe("my-env");
    return { sandboxes: [{ id: V2_SANDBOX_ID, createdAt: 100 }] };
  });
  mock.handleUnary("/SandboxListV2", () => {
    return { sandboxes: [] };
  });

  const ids: string[] = [];
  for await (const sb of mc.sandboxes.experimentalList({
    environment: "my-env",
  })) {
    ids.push(sb.sandboxId);
  }
  expect(ids).toEqual([V2_SANDBOX_ID]);
  mock.assertExhausted();
});

test("ExperimentalList forwards tag filters", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxListV2", (req: any) => {
    expect(req.tags).toEqual([
      { tagName: "env", tagValue: "prod" },
      { tagName: "team", tagValue: "infra" },
    ]);
    return { sandboxes: [] };
  });

  const ids: string[] = [];
  for await (const sb of mc.sandboxes.experimentalList({
    appId: "ap-1234",
    tags: { env: "prod", team: "infra" },
  })) {
    ids.push(sb.sandboxId);
  }
  expect(ids).toEqual([]);
  mock.assertExhausted();
});

test("ExperimentalFromName routes to V2 RPCs", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetFromNameV2", (req: any) => {
    expect(req.sandboxName).toBe("my-sandbox");
    expect(req.appName).toBe("libmodal-test");
    return { sandboxId: V2_SANDBOX_ID };
  });
  // A subsequent lifecycle call routes through the V2 RPC, confirming the
  // returned Sandbox is marked V2.
  mock.handleUnary("/SandboxTerminateV2", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    return {};
  });

  const sb = await mc.sandboxes.experimentalFromName(
    "libmodal-test",
    "my-sandbox",
  );
  expect(sb.sandboxId).toBe(V2_SANDBOX_ID);

  await sb.terminate();

  mock.assertExhausted();
});

test("experimentalSetName routes to V2 RPC", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, {
    taskId: "ta-v2-123",
  });

  mock.handleUnary("/SandboxSetName", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    expect(req.name).toBe("my-sandbox");
    return {};
  });

  await sb.experimentalSetName("my-sandbox");

  mock.assertExhausted();
});

test("experimentalSetName rejects V1 sandboxes", async () => {
  const { mockClient: mc } = createMockModalClients();
  const sb = new Sandbox(mc, V1_SANDBOX_ID, {});

  await expect(sb.experimentalSetName("my-sandbox")).rejects.toThrow(
    "only supported for V2 sandboxes",
  );
});

test("experimentalSetName rejects invalid names", async () => {
  const { mockClient: mc } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, {
    taskId: "ta-v2-123",
  });

  // Validation happens client-side, so no RPC is sent (no handler registered).
  await expect(sb.experimentalSetName("bad name")).rejects.toThrow(
    "Invalid Sandbox name",
  );
});

test.each([
  [Status.ALREADY_EXISTS, AlreadyExistsError],
  [Status.INVALID_ARGUMENT, InvalidError],
  [Status.FAILED_PRECONDITION, ConflictError],
])(
  "experimentalSetName maps gRPC status %s to a typed error",
  async (status, errorClass) => {
    const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
    const sb = new Sandbox(mc, V2_SANDBOX_ID, {
      taskId: "ta-v2-123",
    });

    mock.handleUnary("/SandboxSetName", () => {
      throw new ClientError(
        "/modal.client.ModalClient/SandboxSetName",
        status,
        "server rejected the name",
      );
    });

    await expect(sb.experimentalSetName("my-sandbox")).rejects.toBeInstanceOf(
      errorClass,
    );

    mock.assertExhausted();
  },
);

test("V2 Sandbox reads stdio of a Sandbox that has already finished", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  // No cached task ID, so the read has to resolve one for a Sandbox that ran
  // and then exited. The response carries both a task ID and a result, and the
  // task ID is what makes the buffered output reachable.
  const sb = new Sandbox(mc, V2_SANDBOX_ID, {});
  mock.handleUnary("/SandboxGetTaskIdV2", () => ({
    taskId: "ta-finished",
    taskResult: {
      status: GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS,
      exitcode: 0,
    },
  }));

  const sandboxStdioReadV2 = vi.fn(() =>
    (async function* () {
      yield SandboxStdioReadV2Response.create({
        data: new TextEncoder().encode("after exit\n"),
      });
    })(),
  );
  mockCommandRouter({ sandboxStdioReadV2 });

  expect(await sb.stdout.readText()).toBe("after exit\n");
  expect(sandboxStdioReadV2).toHaveBeenCalledWith(
    "ta-finished",
    FileDescriptor.FILE_DESCRIPTOR_STDOUT,
    expect.any(AbortSignal),
  );
  mock.assertExhausted();
});

test("V2 Sandbox cancelling stdout ends a pending read", async () => {
  const { mockClient: mc } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, { taskId: "ta-v2-123" });

  const sandboxStdioReadV2 = vi.fn((_taskId, _fd, signal?: AbortSignal) =>
    (async function* () {
      await new Promise((_resolve, reject) => {
        signal?.addEventListener(
          "abort",
          () => reject(new ClientError("/test", Status.CANCELLED, "cancelled")),
          { once: true },
        );
      });
      yield SandboxStdioReadV2Response.create({});
    })(),
  );
  mockCommandRouter({ sandboxStdioReadV2 });

  const reader = sb.stdout.getReader();
  const read = reader.read().catch(() => undefined);
  await new Promise((resolve) => globalThis.setTimeout(resolve, 50));
  expect(sandboxStdioReadV2).toHaveBeenCalled();

  const outcome = await Promise.race([
    reader.cancel().then(() => "cancelled"),
    new Promise((resolve) =>
      globalThis.setTimeout(() => resolve("still waiting"), 2000),
    ),
  ]);
  expect(outcome).toBe("cancelled");
  await read;
});

test("V2 Sandbox cancelling stdout aborts a pending task lookup", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, {});

  for (let i = 0; i < 40; i++) {
    mock.handleUnary("/SandboxGetTaskIdV2", () => ({}));
  }
  mockCommandRouter({});

  const reader = sb.stdout.getReader();
  const read = reader.read().catch(() => undefined);
  // Let the first lookup land before giving up on it.
  await new Promise((resolve) => globalThis.setTimeout(resolve, 50));

  const outcome = await Promise.race([
    reader.cancel().then(() => "cancelled"),
    new Promise((resolve) =>
      globalThis.setTimeout(() => resolve("still waiting"), 2000),
    ),
  ]);
  expect(outcome).toBe("cancelled");
  await read;
});

test("V2 Sandbox streams stdout and stderr through the command router", async () => {
  const { mockClient: mc } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, { taskId: "ta-v2-123" });

  const sandboxStdioReadV2 = vi.fn((_taskId: string, fd: FileDescriptor) =>
    (async function* () {
      const text =
        fd === FileDescriptor.FILE_DESCRIPTOR_STDOUT
          ? "hello stdout\n"
          : "oops stderr\n";
      yield SandboxStdioReadV2Response.create({
        data: new TextEncoder().encode(text),
      });
    })(),
  );
  mockCommandRouter({ sandboxStdioReadV2 });

  expect(await sb.stdout.readText()).toBe("hello stdout\n");
  expect(await sb.stderr.readText()).toBe("oops stderr\n");
  expect(sandboxStdioReadV2.mock.calls).toEqual([
    [
      "ta-v2-123",
      FileDescriptor.FILE_DESCRIPTOR_STDOUT,
      expect.any(AbortSignal),
    ],
    [
      "ta-v2-123",
      FileDescriptor.FILE_DESCRIPTOR_STDERR,
      expect.any(AbortSignal),
    ],
  ]);
});

test("V2 Sandbox warns when the worker dropped buffered output", async () => {
  const { mockClient: mc } = createMockModalClients();
  const warn = vi.spyOn(mc.logger, "warn").mockImplementation(() => {});
  onTestFinished(() => warn.mockRestore());

  const sb = new Sandbox(mc, V2_SANDBOX_ID, { taskId: "ta-v2-123" });
  mockCommandRouter({
    sandboxStdioReadV2: () =>
      (async function* () {
        yield SandboxStdioReadV2Response.create({
          data: new TextEncoder().encode("tail\n"),
          startingOffset: 1500,
        });
      })(),
  });

  expect(await sb.stdout.readText()).toBe("tail\n");
  expect(warn).toHaveBeenCalledWith(
    expect.stringContaining("dropped bytes"),
    "sandbox_id",
    V2_SANDBOX_ID,
    "dropped_bytes",
    1500,
  );
});

test("V2 Sandbox stdin writes at increasing offsets and closes with EOF", async () => {
  const { mockClient: mc } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, { taskId: "ta-v2-123" });

  const sandboxStdinWriteV2 = vi.fn().mockResolvedValue({});
  mockCommandRouter({ sandboxStdinWriteV2 });

  await sb.stdin.writeText("hello ");
  await sb.stdin.writeText("world");
  await sb.stdin.close();

  expect(sandboxStdinWriteV2.mock.calls).toEqual([
    ["ta-v2-123", 0, new TextEncoder().encode("hello "), false],
    ["ta-v2-123", 6, new TextEncoder().encode("world"), false],
    ["ta-v2-123", 11, new Uint8Array(0), true],
  ]);
});

test("V2 Sandbox setTags/getTags route to V2 RPCs", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, {
    taskId: "ta-v2-123",
  });

  let setReq: any;
  mock.handleUnary("/SandboxTagsSetV2", (req: any) => {
    setReq = req;
    return {};
  });
  mock.handleUnary("/SandboxTagsGetV2", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    return { tags: [{ tagName: "env", tagValue: "prod" }] };
  });

  await sb.setTags({ env: "prod" });
  expect(setReq.sandboxId).toBe(V2_SANDBOX_ID);
  expect(setReq.tags).toEqual([{ tagName: "env", tagValue: "prod" }]);
  expect(await sb.getTags()).toEqual({ env: "prod" });
  mock.assertExhausted();
});

test("V2 Sandbox supports filesystem", () => {
  const { mockClient: mc } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, {
    taskId: "ta-v2-123",
  });
  expect(() => sb.filesystem).not.toThrow();
  expect(sb.filesystem).toBeDefined();
});

test.each([
  [V1_SANDBOX_ID, SandboxVersion.V1],
  [V2_SANDBOX_ID, SandboxVersion.V2],
  // IDs that match neither known shape are assumed to be V2, so that future
  // ID formats route to the newer backend instead of erroring.
  ["sb-123", SandboxVersion.V2],
  ["sb-nGEijt9WbBMlGrsPH9FOa_", SandboxVersion.V2],
  ["sb-81ARZ3NDEKTSV4RRFFQ69G5FAV", SandboxVersion.V2],
  ["sb-01arz3ndektsv4rrffq69g5fav", SandboxVersion.V2],
  ["sb-01ARZ3NDEKTSV4RRFFQ69G5FAVXY", SandboxVersion.V2],
  ["fu-01ARZ3NDEKTSV4RRFFQ69G5FAV", SandboxVersion.V2],
  ["sb-foo-bar", SandboxVersion.V2],
  ["not-a-sandbox-id", SandboxVersion.V2],
])("getSandboxVersion classifies %s", (sandboxId, expectedVersion) => {
  expect(getSandboxVersion(sandboxId)).toBe(expectedVersion);
});

test("client.sandboxes.fromId routes V1 IDs to SandboxWait", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  const sandboxId = V1_SANDBOX_ID;

  mock.handleUnary("/SandboxWait", (req: any) => {
    expect(req.sandboxId).toBe(sandboxId);
    expect(req.timeout).toBe(0);
    return {
      result: {
        status: GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS,
        exitcode: 0,
      },
    };
  });

  const sb = await mc.sandboxes.fromId(sandboxId);
  expect(await sb.poll()).toBe(0);

  mock.assertExhausted();
});

test("client.sandboxes.fromId routes V2 IDs to SandboxWaitV2", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  const sandboxId = V2_SANDBOX_ID;

  mock.handleUnary("/SandboxTerminateV2", (req: any) => {
    expect(req.sandboxId).toBe(sandboxId);
    return {};
  });
  mock.handleUnary("/SandboxWaitV2", (req: any) => {
    expect(req.sandboxId).toBe(sandboxId);
    expect(req.timeout).toBe(10);
    return {
      result: {
        status: GenericResult_GenericStatus.GENERIC_STATUS_TERMINATED,
      },
    };
  });

  const sb = await mc.sandboxes.fromId(sandboxId);
  expect(await sb.terminate({ wait: true })).toBe(137);

  mock.assertExhausted();
});

test("client.sandboxes.fromId routes unrecognized IDs to the V2 backend", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  // An ID that matches neither known shape (e.g. a format introduced after
  // this client version) routes to the V2 backend instead of erroring.
  const sandboxId = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAVXY";

  mock.handleUnary("/SandboxWaitV2", (req: any) => {
    expect(req.sandboxId).toBe(sandboxId);
    return {
      result: {
        status: GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS,
        exitcode: 0,
      },
    };
  });

  const sb = await mc.sandboxes.fromId(sandboxId);
  expect(await sb.poll()).toBe(0);

  mock.assertExhausted();
});

test("SandboxGetTaskIdPolling", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetTaskId", () => ({}));
  mock.handleUnary("/SandboxGetTaskId", () => ({ taskId: "ta-123" }));

  const sb = await mc.sandboxes.fromId(V1_SANDBOX_ID);
  await expect(sb.filesystem.stat("/test")).rejects.toThrow();

  mock.assertExhausted();
});

test("SandboxGetTaskIdTerminated", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetTaskId", () => ({
    taskResult: { status: 3 },
  }));

  const sb = await mc.sandboxes.fromId(V1_SANDBOX_ID);
  await expect(sb.exec(["echo", "hello"])).rejects.toThrow(/already completed/);

  mock.assertExhausted();
});

test("SandboxWaitUntilReady", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("python:3.13-alpine");

  const sb = await tc.sandboxes.create(app, image, {
    command: ["python", "-m", "http.server", "8080"],
    readinessProbe: Probe.withTcp(8080),
  });
  onTestFinished(async () => await sb.terminate());

  await sb.waitUntilReady(60_000);
}, 60_000);

test("SandboxWaitUntilReady times out", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("python:3.13-alpine");

  const sb = await tc.sandboxes.create(app, image, {
    command: ["python", "-m", "http.server", "8080"],
    // A readiness probe that always fails, so the sandbox never becomes ready.
    readinessProbe: Probe.withExec(["sh", "-c", "exit 1"]),
  });
  onTestFinished(async () => await sb.terminate());

  await expect(sb.waitUntilReady(5_000)).rejects.toThrow(TimeoutError);
}, 60_000);

test("validateExecArgs with args within limit", () => {
  expect(() => validateExecArgs(["echo", "hello"])).not.toThrow();

  expect(() => validateExecArgs(["a".repeat(2 ** 16 - 10)])).not.toThrow();
});

test("validateExecArgs with args exceeding ARG_MAX", () => {
  const longArg = "a".repeat(2 ** 16 + 1);
  const args = [longArg];
  expect(() => validateExecArgs(args)).toThrow(
    "Total length of CMD arguments must be less than",
  );
});

test("validateExperimentalEncryptionKey", () => {
  const key = new Uint8Array(16).fill(7);
  expect(validateExperimentalEncryptionKey(undefined)).toBeUndefined();
  expect(validateExperimentalEncryptionKey(key)).toBe(key);

  expect(() => validateExperimentalEncryptionKey("not bytes" as any)).toThrow(
    TypeError,
  );
  expect(() => validateExperimentalEncryptionKey(new Uint8Array(0))).toThrow(
    "between 16 and 512 bytes",
  );
  expect(() => validateExperimentalEncryptionKey(new Uint8Array(15))).toThrow(
    "between 16 and 512 bytes",
  );
  expect(() => validateExperimentalEncryptionKey(new Uint8Array(513))).toThrow(
    "between 16 and 512 bytes",
  );
});

test("TaskMountDirectoryRequest carries experimental encryption key", () => {
  const key = new Uint8Array(16).fill(1);

  const req = buildTaskMountDirectoryRequestProto(
    "ta-123",
    "/mnt/data",
    "im-123",
    {
      experimentalEncryptionKey: key,
    },
  );
  expect(req.taskId).toBe("ta-123");
  expect(req.path).toEqual(new TextEncoder().encode("/mnt/data"));
  expect(req.imageId).toBe("im-123");
  expect(req.customerSuppliedEncryptionKey).toEqual(key);

  const reqWithoutKey = buildTaskMountDirectoryRequestProto(
    "ta-123",
    "/mnt/data",
    "im-123",
  );
  expect(reqWithoutKey.customerSuppliedEncryptionKey).toBeUndefined();
});

test("TaskSnapshotDirectoryRequest carries experimental encryption key", () => {
  const key = new Uint8Array(32).fill(2);

  const req = buildTaskSnapshotDirectoryRequestProto(
    "ta-123",
    "/mnt/data",
    "snapshot-123",
    3600,
    { experimentalEncryptionKey: key },
  );
  expect(req.taskId).toBe("ta-123");
  expect(req.path).toEqual(new TextEncoder().encode("/mnt/data"));
  expect(req.snapshotId).toBe("snapshot-123");
  expect(req.ttlSeconds).toBe(3600);
  expect(req.customerSuppliedEncryptionKey).toEqual(key);

  const reqWithoutKey = buildTaskSnapshotDirectoryRequestProto(
    "ta-123",
    "/mnt/data",
    "snapshot-123",
    3600,
  );
  expect(reqWithoutKey.customerSuppliedEncryptionKey).toBeUndefined();
});

test("buildTaskExecStartRequestProto defaults", () => {
  const req = buildTaskExecStartRequestProto("task-123", "exec-456", ["bash"]);

  expect(req.taskId).toBe("task-123");
  expect(req.execId).toBe("exec-456");
  expect(req.commandArgs).toEqual(["bash"]);
  expect(req.stdoutConfig).toBe(1); // TASK_EXEC_STDOUT_CONFIG_PIPE
  expect(req.stderrConfig).toBe(1); // TASK_EXEC_STDERR_CONFIG_PIPE
  expect(req.timeoutSecs).toBeUndefined();
  expect(req.workdir).toBeUndefined();
  expect(req.secretIds).toEqual([]);
  expect(req.env).toEqual({});
  expect(req.ptyInfo).toBeUndefined();
  expect(req.runtimeDebug).toBe(false);
});

test("buildTaskExecStartRequestProto with stdout ignore", () => {
  const req = buildTaskExecStartRequestProto("task-123", "exec-456", ["bash"], {
    stdout: "ignore",
    stderr: "ignore",
  });

  expect(req.stdoutConfig).toBe(0); // TASK_EXEC_STDOUT_CONFIG_DEVNULL
  expect(req.stderrConfig).toBe(0); // TASK_EXEC_STDERR_CONFIG_DEVNULL
});

test("buildTaskExecStartRequestProto with PTY", () => {
  const req = buildTaskExecStartRequestProto("task-123", "exec-456", ["bash"], {
    pty: true,
  });

  const ptyInfo = req.ptyInfo!;
  expect(ptyInfo.enabled).toBe(true);
  expect(ptyInfo.winszRows).toBe(24);
  expect(ptyInfo.winszCols).toBe(80);
  expect(ptyInfo.envTerm).toBe("xterm-256color");
  expect(ptyInfo.envColorterm).toBe("truecolor");
  expect(ptyInfo.ptyType).toBe(PTYInfo_PTYType.PTY_TYPE_SHELL);
});

test("buildTaskExecStartRequestProto with workdir", () => {
  const req = buildTaskExecStartRequestProto("task-123", "exec-456", ["pwd"], {
    workdir: "/tmp",
  });

  expect(req.workdir).toBe("/tmp");
});

test("buildTaskExecStartRequestProto rejects relative workdir", () => {
  expect(() =>
    buildTaskExecStartRequestProto("task-123", "exec-456", ["pwd"], {
      workdir: "tmp",
    }),
  ).toThrow("workdir must be an absolute path");
});

test("buildTaskExecStartRequestProto rejects empty workdir", () => {
  expect(() =>
    buildTaskExecStartRequestProto("task-123", "exec-456", ["pwd"], {
      workdir: "",
    }),
  ).toThrow("workdir must be an absolute path");
});

test("buildTaskExecStartRequestProto with containerId", () => {
  const req = buildTaskExecStartRequestProto(
    "task-123",
    "exec-456",
    ["pwd"],
    undefined,
    "ctr-123",
  );

  expect(req.containerId).toBe("ctr-123");
});

test("buildTaskExecStartRequestProto with timeoutMs", () => {
  const req = buildTaskExecStartRequestProto(
    "task-123",
    "exec-456",
    ["sleep", "10"],
    { timeoutMs: 5000 },
  );

  expect(req.timeoutSecs).toBe(5);
});

test("buildTaskExecStartRequestProto with env", () => {
  const req = buildTaskExecStartRequestProto("task-123", "exec-456", ["env"], {
    env: { FOO: "bar" },
  });

  expect(req.env).toEqual({ FOO: "bar" });
});

test.each([
  [0, "timeoutMs must be positive"],
  [-1000, "timeoutMs must be positive"],
  [1500, "timeoutMs must be a multiple of 1000ms"],
])(
  "buildTaskExecStartRequestProto invalid timeoutMs %d",
  (timeoutMs, expectedError) => {
    expect(() =>
      buildTaskExecStartRequestProto("task-123", "exec-456", ["bash"], {
        timeoutMs,
      }),
    ).toThrow(expectedError);
  },
);

test("SandboxExecStdinStdout", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const p = await sb.exec(["sh", "-c", "while read line; do echo $line; done"]);
  await p.stdin.writeText("foo\n");
  await p.stdin.writeText("bar\n");
  await p.stdin.close();
  expect(await p.stdout.readText()).toBe("foo\nbar\n");
});

test("SandboxExecWaitExitCode", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const p = await sb.exec(["sh", "-c", "exit 42"]);
  expect(await p.wait()).toBe(42);
});

test("SandboxExecWaitSignal", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  // The shell kills itself with SIGKILL (9); wait() should return 128 + 9 = 137.
  const p = await sb.exec(["sh", "-c", "kill -9 $$"]);
  expect(await p.wait()).toBe(128 + 9);
});

test("ContainerProcess cancelling stdout ends a pending read", async () => {
  const execStdioRead = vi.fn(
    (_taskId, _execId, _fd, _deadline, signal?: AbortSignal) =>
      (async function* () {
        await new Promise((_resolve, reject) => {
          signal?.addEventListener(
            "abort",
            () =>
              reject(new ClientError("/test", Status.CANCELLED, "cancelled")),
            { once: true },
          );
        });
        yield TaskExecStdioReadResponse.create({});
      })(),
  );
  const p = new ContainerProcess("ta-1", "ex-1", {
    execStdioRead,
    execStdinWrite: vi.fn(),
  } as unknown as TaskCommandRouterClientImpl);

  const reader = p.stdout.getReader();
  const read = reader.read().catch(() => undefined);
  await new Promise((resolve) => globalThis.setTimeout(resolve, 50));
  expect(execStdioRead).toHaveBeenCalled();

  const outcome = await Promise.race([
    reader.cancel().then(() => "cancelled"),
    new Promise((resolve) =>
      globalThis.setTimeout(() => resolve("still waiting"), 2000),
    ),
  ]);
  expect(outcome).toBe("cancelled");
  await read;
});

test("SandboxExecDoubleRead", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const p = await sb.exec(["echo", "hello"]);
  expect(await p.stdout.readText()).toBe("hello\n");
  expect(await p.stdout.readText()).toBe("");
  expect(await p.wait()).toBe(0);
});

test("SandboxExecBinaryMode", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const p = await sb.exec(["printf", "\\x01\\x02\\x03"], { mode: "binary" });
  const bytes = await p.stdout.readBytes();
  expect(bytes).toEqual(new Uint8Array([0x01, 0x02, 0x03]));
  expect(await p.wait()).toBe(0);
});

test("SandboxExecWithPty", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const p = await sb.exec(["echo", "hello"], { pty: true });
  expect(await p.wait()).toBe(0);
});

test("SandboxExecWaitTimeout", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const p = await sb.exec(["sleep", "999"], { timeoutMs: 1000 });
  const exitCode = await p.wait();
  expect(exitCode).toBe(128 + 9);
});

test("SandboxExecOutputTimeout", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const p = await sb.exec(["sh", "-c", "echo hi; sleep 999"], {
    timeoutMs: 1000,
  });

  // The deadline can be observed either while draining stdout or while waiting
  // for the exit status, depending on when the command router reports EOF.
  const stdoutResult = await p.stdout.readText().then(
    (output) => ({ ok: true as const, output }),
    (error) => ({ ok: false as const, error: String(error) }),
  );

  if (!stdoutResult.ok) {
    expect(stdoutResult.error).toMatch(/Deadline exceeded/);
    return;
  }

  expect(stdoutResult.output).toBe("hi\n");

  const waitResult = await p.wait().then(
    (exitCode) => ({ ok: true as const, exitCode }),
    (error) => ({ ok: false as const, error: String(error) }),
  );

  if (waitResult.ok) {
    expect(waitResult.exitCode).toBe(137);
  } else {
    expect(waitResult.error).toMatch(/Deadline exceeded/);
  }
});

test("SandboxDetachIsNonDestructive", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image);
  const sandboxId = sb.sandboxId;

  sb.detach();

  const sbFromId = await tc.sandboxes.fromId(sandboxId);
  onTestFinished(async () => await sbFromId.terminate());
  expect(sbFromId.sandboxId).toBe(sandboxId);

  const p = await sbFromId.exec(["echo", "still running"]);
  expect(await p.wait()).toBe(0);
});

test("SandboxDetachIsIdempotent", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image);
  const sbFromId = await tc.sandboxes.fromId(sb.sandboxId);
  onTestFinished(async () => await sbFromId.terminate());

  // Multiple calls should not throw
  sb.detach();
  sb.detach();
  sb.detach();
});

test("SandboxTerminateThenDetach", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image);

  await sb.terminate();
  sb.detach(); // Should not throw
});

test("SandboxDetachForbidsAllOperations", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image);
  const sbFromId = await tc.sandboxes.fromId(sb.sandboxId);
  onTestFinished(async () => await sbFromId.terminate());

  sb.detach();

  const errorMsg = "Unable to perform operation on a detached sandbox";

  await expect(sb.exec(["echo", "hello"])).rejects.toThrow(errorMsg);
  await expect(sb.createConnectToken()).rejects.toThrow(errorMsg);
  await expect(sb.terminate()).rejects.toThrow(errorMsg);
  await expect(sb.tunnels()).rejects.toThrow(errorMsg);
  await expect(sb.snapshotFilesystem()).rejects.toThrow(errorMsg);
  await expect(sb.mountImage("/abc")).rejects.toThrow(errorMsg);
  await expect(
    sb.updateNetworkPolicy({
      outboundCidrAllowlist: [],
      outboundDomainAllowlist: [],
    }),
  ).rejects.toThrow(errorMsg);
  await expect(sb.snapshotDirectory("/abc")).rejects.toThrow(errorMsg);
  await expect(sb.poll()).rejects.toThrow(errorMsg);
  await expect(sb.setTags({})).rejects.toThrow(errorMsg);
  await expect(sb.getTags()).rejects.toThrow(errorMsg);
  await expect(sb.waitUntilReady()).rejects.toThrow(errorMsg);
});

test("updateNetworkPolicy sends correct request via mocked command router", async () => {
  const { mockClient: mc } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, {
    taskId: "ta-v2-123",
  });

  const setNetworkAccess = vi.fn().mockResolvedValue(undefined);
  // Mock out the task command router so the test doesn't touch real infra.
  const tryInit = vi
    .spyOn(TaskCommandRouterClientImpl, "tryInit")
    .mockResolvedValue({
      setNetworkAccess,
      close: vi.fn(),
    } as unknown as TaskCommandRouterClientImpl);
  onTestFinished(() => tryInit.mockRestore());

  await sb.updateNetworkPolicy({
    outboundCidrAllowlist: ["10.0.0.0/8"],
    outboundDomainAllowlist: ["example.com"],
  });

  expect(setNetworkAccess).toHaveBeenCalledTimes(1);
  const request = setNetworkAccess.mock
    .calls[0][0] as TaskSetNetworkAccessRequest;
  expect(request.taskId).toBe("ta-v2-123");
  expect(request.networkAccess?.networkAccessType).toBe(
    NetworkAccess_NetworkAccessType.ALLOWLIST,
  );
  expect(request.networkAccess?.allowedCidrs).toEqual(["10.0.0.0/8"]);
  expect(request.networkAccess?.allowedDomains).toEqual(["example.com"]);
});

test("sidecar snapshotFilesystem targets its container", async () => {
  const { mockClient: mc } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, { taskId: "ta-v2-123" });

  const containerCreate = vi.fn().mockResolvedValue({
    containerId: "sb-test-ctr-SIDECAR123",
    containerName: "worker",
  });
  const snapshotFilesystem = vi
    .fn()
    .mockResolvedValue({ imageId: "im-sidecar-snapshot" });
  const tryInit = vi
    .spyOn(TaskCommandRouterClientImpl, "tryInit")
    .mockResolvedValue({
      containerCreate,
      snapshotFilesystem,
      close: vi.fn(),
    } as unknown as TaskCommandRouterClientImpl);
  onTestFinished(() => tryInit.mockRestore());

  const sidecar = await sb.experimentalSidecars.create(
    "worker",
    new Image(mc, "im-built", ""),
    { command: ["sleep", "infinity"] },
  );
  const image = await sidecar.snapshotFilesystem({ ttlMs: null });

  expect(image.imageId).toBe("im-sidecar-snapshot");
  expect(snapshotFilesystem).toHaveBeenCalledTimes(1);
  const request = snapshotFilesystem.mock
    .calls[0][0] as TaskSnapshotFilesystemRequest;
  expect(request.taskId).toBe("ta-v2-123");
  expect(request.containerId).toBe(sidecar.containerId);
  expect(request.ttlSeconds).toBe(-1);
  expect(request.snapshotId).toBeTruthy();
});

test("updateNetworkPolicy rejects when a dimension is missing", async () => {
  const { mockClient: mc } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, {
    taskId: "ta-v2-123",
  });

  await expect(
    sb.updateNetworkPolicy({ outboundDomainAllowlist: ["example.com"] }),
  ).rejects.toThrow("both outboundCidrAllowlist and outboundDomainAllowlist");

  await expect(
    sb.updateNetworkPolicy({ outboundCidrAllowlist: ["10.0.0.0/8"] }),
  ).rejects.toThrow("both outboundCidrAllowlist and outboundDomainAllowlist");

  await expect(sb.updateNetworkPolicy({})).rejects.toThrow(
    "both outboundCidrAllowlist and outboundDomainAllowlist",
  );
});

test("experimentalSnapshot takes a memory snapshot via the command router for V2 sandboxes", async () => {
  const { mockClient: mc } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, {
    taskId: "ta-v2-123",
  });

  const snapshotMemory = vi
    .fn()
    .mockResolvedValue({ snapshotId: "sn-01BX5ZZKBKACTAV9WEVGEMMVRY" });
  const tryInit = vi
    .spyOn(TaskCommandRouterClientImpl, "tryInit")
    .mockResolvedValue({
      snapshotMemory,
      close: vi.fn(),
    } as unknown as TaskCommandRouterClientImpl);
  onTestFinished(() => tryInit.mockRestore());

  const snapshot = await sb.experimentalSnapshot();

  expect(snapshot.snapshotId).toBe("sn-01BX5ZZKBKACTAV9WEVGEMMVRY");
  expect(snapshotMemory).toHaveBeenCalledTimes(1);
  const request = snapshotMemory.mock.calls[0][0] as TaskSnapshotMemoryRequest;
  expect(request.taskId).toBe("ta-v2-123");
  expect(request.idempotencyKey).toBeTruthy();
});

test("experimentalFromSnapshot restores a V2 snapshot via SandboxRestoreV2", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary(
    "/SandboxRestoreV2",
    (req: any): SandboxRestoreV2Response => {
      expect(req.snapshotId).toBe("sn-01BX5ZZKBKACTAV9WEVGEMMVRY");
      return SandboxRestoreV2Response.create({
        sandboxId: V2_SANDBOX_ID,
        taskId: "ta-restored-v2-123",
      });
    },
  );

  const snapshot = new SandboxSnapshot(mc, "sn-01BX5ZZKBKACTAV9WEVGEMMVRY", {
    isV2: true,
  });
  const sb = await mc.sandboxes.experimentalFromSnapshot(snapshot);
  expect(sb.sandboxId).toBe(V2_SANDBOX_ID);
});

test("experimentalFromSnapshot fetches the snapshot version when unknown", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary(
    "/SandboxSnapshotGet",
    (req: any): SandboxSnapshotGetResponse => {
      expect(req.snapshotId).toBe("sn-01BX5ZZKBKACTAV9WEVGEMMVRY");
      return {
        snapshotId: "sn-01BX5ZZKBKACTAV9WEVGEMMVRY",
        handleMetadata: { isV2: true },
      };
    },
  );
  mock.handleUnary("/SandboxRestoreV2", (): SandboxRestoreV2Response => {
    return SandboxRestoreV2Response.create({
      sandboxId: V2_SANDBOX_ID,
      taskId: "ta-restored-v2-123",
    });
  });

  const snapshot = await mc.sandboxSnapshots.fromId(
    "sn-01BX5ZZKBKACTAV9WEVGEMMVRY",
  );
  const sb = await mc.sandboxes.experimentalFromSnapshot(snapshot);
  expect(sb.sandboxId).toBe(V2_SANDBOX_ID);
});

test("experimentalGetExitSnapshot routes V1 Sandboxes to SandboxGetExitSnapshot", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetExitSnapshot", (req: any) => {
    expect(req.sandboxId).toBe(V1_SANDBOX_ID);
    expect(req.timeout).toBe(0);
    return { success: { imageId: "im-exit-snapshot-123" } };
  });

  const sb = await mc.sandboxes.fromId(V1_SANDBOX_ID);
  const image = await sb.experimentalGetExitSnapshot({ timeoutMs: 0 });
  expect(image.imageId).toBe("im-exit-snapshot-123");
  mock.assertExhausted();
});

test("experimentalGetExitSnapshot routes V2 Sandboxes to SandboxGetExitSnapshotV2", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetExitSnapshotV2", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    expect(req.timeout).toBe(0);
    return { success: { imageId: "im-exit-snapshot-123" } };
  });

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  const image = await sb.experimentalGetExitSnapshot({ timeoutMs: 0 });
  expect(image.imageId).toBe("im-exit-snapshot-123");
  mock.assertExhausted();
});

test("experimentalGetExitSnapshot polls until the snapshot is ready", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetExitSnapshotV2", (req: any) => {
    expect(req.timeout).toBe(10);
    return { pending: {} };
  });
  mock.handleUnary("/SandboxGetExitSnapshotV2", (req: any) => {
    expect(req.timeout).toBe(10);
    return { success: { imageId: "im-exit-snapshot-123" } };
  });

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  const image = await sb.experimentalGetExitSnapshot();
  expect(image.imageId).toBe("im-exit-snapshot-123");
  mock.assertExhausted();
});

test("experimentalGetExitSnapshot works after detach", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetExitSnapshotV2", (req: any) => {
    expect(req.sandboxId).toBe(V2_SANDBOX_ID);
    return { success: { imageId: "im-exit-snapshot-123" } };
  });

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  sb.detach();

  const image = await sb.experimentalGetExitSnapshot({ timeoutMs: 0 });
  expect(image.imageId).toBe("im-exit-snapshot-123");
  mock.assertExhausted();
});

test("experimentalGetExitSnapshot repeats long polls", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  for (let i = 0; i < 3; i++) {
    mock.handleUnary("/SandboxGetExitSnapshotV2", (req: any) => {
      expect(req.timeout).toBeGreaterThan(9);
      expect(req.timeout).toBeLessThanOrEqual(10);
      return { pending: {} };
    });
  }
  mock.handleUnary("/SandboxGetExitSnapshotV2", () => ({
    success: { imageId: "im-exit-snapshot-123" },
  }));

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  const image = await sb.experimentalGetExitSnapshot({ timeoutMs: 30_000 });
  expect(image.imageId).toBe("im-exit-snapshot-123");
  mock.assertExhausted();
});

test("experimentalGetExitSnapshot enforces the aggregate deadline", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  for (let i = 0; i < 3; i++) {
    mock.handleUnary("/SandboxGetExitSnapshotV2", async (req: any) => {
      expect(req.timeout).toBeGreaterThanOrEqual(0);
      expect(req.timeout).toBeLessThanOrEqual(0.05);
      await new Promise((resolve) => setTimeout(resolve, req.timeout * 1000));
      return { pending: {} };
    });
  }

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  await expect(
    sb.experimentalGetExitSnapshot({ timeoutMs: 50 }),
  ).rejects.toThrow(TimeoutError);
});

test("experimentalGetExitSnapshot maps not-enabled to InvalidError", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetExitSnapshotV2", () => {
    throw new ClientError(
      "/modal.client.ModalClient/SandboxGetExitSnapshotV2",
      Status.INVALID_ARGUMENT,
      "Exit snapshot is not enabled for this sandbox",
    );
  });

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  await expect(
    sb.experimentalGetExitSnapshot({ timeoutMs: 0 }),
  ).rejects.toThrow(InvalidError);
  mock.assertExhausted();
});

test("experimentalGetExitSnapshot maps a missing Sandbox to NotFoundError", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetExitSnapshotV2", () => {
    throw new ClientError(
      "/modal.client.ModalClient/SandboxGetExitSnapshotV2",
      Status.NOT_FOUND,
      "Sandbox not found",
    );
  });

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  await expect(
    sb.experimentalGetExitSnapshot({ timeoutMs: 0 }),
  ).rejects.toThrow(NotFoundError);
  mock.assertExhausted();
});

test("experimentalGetExitSnapshot surfaces an internal error outcome", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetExitSnapshotV2", () => ({
    error: {
      errorCode: SandboxGetExitSnapshotResponse_ErrorCode.ERROR_CODE_INTERNAL,
      message: "malformed snapshot result",
    },
  }));

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  const err = await sb
    .experimentalGetExitSnapshot({ timeoutMs: 0 })
    .catch((e: unknown) => e);
  expect(err).toBeInstanceOf(ExecutionError);
  expect((err as Error).message).toBe("malformed snapshot result");
  mock.assertExhausted();
});

test("experimentalGetExitSnapshot times out on an immediate check while pending", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetExitSnapshotV2", (req: any) => {
    expect(req.timeout).toBe(0);
    return { pending: {} };
  });

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  await expect(
    sb.experimentalGetExitSnapshot({ timeoutMs: 0 }),
  ).rejects.toThrow(TimeoutError);
  mock.assertExhausted();
});

test("experimentalGetExitSnapshot maps a failed snapshot to SnapshotCreationError", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetExitSnapshotV2", () => ({
    error: {
      errorCode: SandboxGetExitSnapshotResponse_ErrorCode.ERROR_CODE_TIMEOUT,
      message: "no exit snapshot",
    },
  }));

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  await expect(
    sb.experimentalGetExitSnapshot({ timeoutMs: 0 }),
  ).rejects.toThrow(SnapshotCreationError);
  mock.assertExhausted();
});

test("experimentalGetExitSnapshot rejects a negative timeout", async () => {
  const { mockClient: mc } = createMockModalClients();

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  await expect(
    sb.experimentalGetExitSnapshot({ timeoutMs: -1 }),
  ).rejects.toThrow(InvalidError);
});

test("experimentalGetExitSnapshot absorbs transient poll failures", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  for (const code of [Status.UNAVAILABLE, Status.INTERNAL]) {
    mock.handleUnary("/SandboxGetExitSnapshotV2", () => {
      throw new ClientError(
        "/modal.client.ModalClient/SandboxGetExitSnapshotV2",
        code,
        "server hiccup",
      );
    });
  }
  mock.handleUnary("/SandboxGetExitSnapshotV2", () => ({
    success: { imageId: "im-exit-snapshot-123" },
  }));

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  const image = await sb.experimentalGetExitSnapshot();
  expect(image.imageId).toBe("im-exit-snapshot-123");
  // Transient failures below the consecutive limit are absorbed by re-polling.
  mock.assertExhausted();
});

test("experimentalGetExitSnapshot surfaces repeated poll failures", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  for (let i = 0; i < 3; i++) {
    mock.handleUnary("/SandboxGetExitSnapshotV2", () => {
      throw new ClientError(
        "/modal.client.ModalClient/SandboxGetExitSnapshotV2",
        Status.UNAVAILABLE,
        "server restarting",
      );
    });
  }

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  const err = await sb.experimentalGetExitSnapshot().catch((e: unknown) => e);
  expect(err).toBeInstanceOf(ClientError);
  expect((err as ClientError).code).toBe(Status.UNAVAILABLE);
  mock.assertExhausted();
});

test("experimentalGetExitSnapshot maps a poll failure after the deadline to TimeoutError", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  for (let i = 0; i < 2; i++) {
    mock.handleUnary("/SandboxGetExitSnapshotV2", () => {
      throw new ClientError(
        "/modal.client.ModalClient/SandboxGetExitSnapshotV2",
        Status.UNAVAILABLE,
        "server restarting",
      );
    });
  }

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  // Once the caller's own budget is gone, a failed poll is a caller timeout.
  await expect(
    sb.experimentalGetExitSnapshot({ timeoutMs: 50 }),
  ).rejects.toThrow(TimeoutError);
  mock.assertExhausted();
});

test("experimentalGetExitSnapshot surfaces a rate limit without a retry policy", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/SandboxGetExitSnapshotV2", () => {
    throw new ClientError(
      "/modal.client.ModalClient/SandboxGetExitSnapshotV2",
      Status.RESOURCE_EXHAUSTED,
      "rate limit exceeded",
    );
  });

  const sb = await mc.sandboxes.fromId(V2_SANDBOX_ID);
  const err = await sb.experimentalGetExitSnapshot().catch((e: unknown) => e);
  expect(err).toBeInstanceOf(ClientError);
  expect((err as ClientError).code).toBe(Status.RESOURCE_EXHAUSTED);
  mock.assertExhausted();
});
