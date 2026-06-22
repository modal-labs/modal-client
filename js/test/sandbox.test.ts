import { tc } from "../test-support/test-client";
import { parseGpuConfig } from "../src/app";
import {
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
} from "../src/sandbox";
import { expect, test, onTestFinished } from "vitest";
import {
  GPUConfig,
  PTYInfo_PTYType,
  NetworkAccess_NetworkAccessType,
  GenericResult_GenericStatus,
  ImageGetOrCreateResponse,
  AppGetOrCreateResponse,
  SandboxCreateResponse,
  SandboxCreateV2Response,
} from "../proto/modal_proto/api";
import { createMockModalClients } from "../test-support/grpc_mock";
import { TimeoutError } from "modal";

const V1_SANDBOX_ID = "sb-nGEijt9WbBMlGrsPH9FOaC";
const V2_SANDBOX_ID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV";

test("CreateOneSandbox", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image);
  expect(sb.sandboxId).toBeTruthy();
  expect(await sb.terminate({ wait: true })).toBe(137);
}, 30000); // fixme(ayush): this probably shouldn't take > 20s

test("CreateOneSandboxTerminateWaitWorks", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");

  const sb = await tc.sandboxes.create(app, image);
  expect(sb.sandboxId).toBeTruthy();
  await sb.terminate();
  expect(await sb.wait()).toBe(137);
});

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
      experimentalOptions: { enable_docker: "not-a-bool" },
    }),
  ).rejects.toThrow("must be a bool");
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
    expect(req.definition?.experimentalOptions).toMatchObject(options);
    return { sandboxId: "sb-1234" };
  });

  mock.handleUnary("/AppGetOrCreate", (_: any): AppGetOrCreateResponse => {
    return { appId: "ap-1234" };
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

test.each([
  ["tags", { tags: { key: "value" } }, "tags are not supported"],
  ["gpu", { gpu: "A10G" }, "GPUs are not supported"],
  [
    "custom domain",
    { customDomain: "example.com" },
    "custom domains are not supported",
  ],
  [
    "proxy",
    { proxy: { proxyId: "pr-123" } as any },
    "proxies are not supported",
  ],
  [
    "includeOidcIdentityToken",
    { includeOidcIdentityToken: true },
    "includeOidcIdentityToken is not supported",
  ],
  [
    "cloud bucket mount with oidcAuthRoleArn",
    {
      cloudBucketMounts: {
        "/bucket": {
          bucketName: "bucket",
          oidcAuthRoleArn: "arn:aws:iam::123:role/r",
        } as any,
      },
    },
    "CloudBucketMount with oidcAuthRoleArn is not supported",
  ],
])(
  "buildSandboxCreateV2RequestProto rejects unsupported option %s",
  async (_name, params, expectedError) => {
    await expect(
      buildSandboxCreateV2RequestProto("app-123", "img-456", params),
    ).rejects.toThrow(expectedError);
  },
);

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

test("ExperimentalCreate routes lifecycle calls to V2 RPCs", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/AppGetOrCreate", (_: any): AppGetOrCreateResponse => {
    return { appId: "ap-1234" };
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
    return { sandboxId: V2_SANDBOX_ID, taskId: "ta-v2-123", tunnels: [] };
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

test("V2 Sandbox rejects V1-only runtime methods", async () => {
  const { mockClient: mc } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, {
    isV2: true,
    taskId: "ta-v2-123",
  });
  const expectedError = "not supported for V2 sandboxes";

  expect(() => sb.stdin).toThrow(expectedError);
  expect(() => sb.stdout).toThrow(expectedError);
  expect(() => sb.stderr).toThrow(expectedError);
  await expect(sb.setTags({})).rejects.toThrow(expectedError);
  await expect(sb.getTags()).rejects.toThrow(expectedError);
  await expect(sb.createConnectToken()).rejects.toThrow(expectedError);
});

test.each([
  [V1_SANDBOX_ID, SandboxVersion.V1],
  [V2_SANDBOX_ID, SandboxVersion.V2],
])("getSandboxVersion classifies %s", (sandboxId, expectedVersion) => {
  expect(getSandboxVersion(sandboxId)).toBe(expectedVersion);
});

test.each([
  "sb-123",
  "sb-nGEijt9WbBMlGrsPH9FOa_",
  "sb-81ARZ3NDEKTSV4RRFFQ69G5FAV",
  "sb-01arz3ndektsv4rrffq69g5fav",
  "fu-01ARZ3NDEKTSV4RRFFQ69G5FAV",
  "sb-foo-bar",
  "not-a-sandbox-id",
])("getSandboxVersion rejects invalid ID %s", (sandboxId) => {
  expect(() => getSandboxVersion(sandboxId)).toThrow("Invalid Sandbox ID");
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

test("client.sandboxes.fromId rejects invalid IDs before RPC", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  await expect(mc.sandboxes.fromId("sb-123")).rejects.toThrow(
    "Invalid Sandbox ID",
  );

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
  await expect(sb.snapshotDirectory("/abc")).rejects.toThrow(errorMsg);
  await expect(sb.poll()).rejects.toThrow(errorMsg);
  await expect(sb.setTags({})).rejects.toThrow(errorMsg);
  await expect(sb.getTags()).rejects.toThrow(errorMsg);
  await expect(sb.waitUntilReady()).rejects.toThrow(errorMsg);
});
