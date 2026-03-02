import { App, Image, Sandbox, Secret, Volume } from "modal";
import { parseGpuConfig } from "../../src/app";
import { buildSandboxCreateRequestProto } from "../../src/sandbox";
import { expect, test, onTestFinished } from "vitest";
import { GPUConfig, PTYInfo_PTYType } from "../../proto/modal_proto/api";

test("CreateOneSandbox", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  expect(app.appId).toBeTruthy();
  expect(app.name).toBe("libmodal-test");

  const image = await app.imageFromRegistry("alpine:3.21");
  expect(image.imageId).toBeTruthy();

  const sb = await app.createSandbox(image);
  expect(sb.sandboxId).toBeTruthy();
  expect(await sb.terminate({ wait: true })).toBe(137);
});

test("PassCatToStdin", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  // Spawn a sandbox running the "cat" command.
  const sb = await app.createSandbox(image, { command: ["cat"] });

  // Write to the sandbox's stdin and read from its stdout.
  await sb.stdin.writeText("this is input that should be mirrored by cat");
  await sb.stdin.close();
  expect(await sb.stdout.readText()).toBe(
    "this is input that should be mirrored by cat",
  );

  // Terminate the sandbox.
  await sb.terminate();
});

test("IgnoreLargeStdout", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("python:3.13-alpine");

  const sb = await app.createSandbox(image);
  try {
    const p = await sb.exec(["python", "-c", `print("a" * 1_000_000)`], {
      stdout: "ignore",
    });
    expect(await p.stdout.readText()).toBe(""); // Stdout is ignored
    // Stdout should be consumed after cancel, without blocking the process.
    expect(await p.wait()).toBe(0);
  } finally {
    await sb.terminate();
  }
});

test("SandboxCreateOptions", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const sandbox = await app.createSandbox(image, {
    command: ["echo", "hello, params"],
    cloud: "aws",
    regions: ["us-east-1", "us-west-2"],
    verbose: true,
  });

  onTestFinished(async () => {
    await sandbox.terminate();
  });

  expect(sandbox).toBeDefined();
  expect(sandbox.sandboxId).toMatch(/^sb-/);

  const exitCode = await sandbox.wait();
  expect(exitCode).toBe(0);

  await expect(
    app.createSandbox(image, {
      cloud: "invalid-cloud",
    }),
  ).rejects.toThrow("INVALID_ARGUMENT");

  await expect(
    app.createSandbox(image, {
      regions: ["invalid-region"],
    }),
  ).rejects.toThrow("INVALID_ARGUMENT");
});

test("SandboxExecOptions", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const sb = await app.createSandbox(image);
  try {
    // Test with a custom working directory and timeout.
    const p = await sb.exec(["pwd"], {
      workdir: "/tmp",
      timeoutMs: 5000,
    });

    expect(await p.stdout.readText()).toBe("/tmp\n");
    expect(await p.wait()).toBe(0);
  } finally {
    await sb.terminate();
  }
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
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const volume = await Volume.fromName("libmodal-test-sandbox-volume", {
    createIfMissing: true,
  });

  const sandbox = await app.createSandbox(image, {
    command: ["echo", "volume test"],
    volumes: { "/mnt/test": volume },
  });

  expect(sandbox).toBeDefined();
  expect(sandbox.sandboxId).toMatch(/^sb-/);

  const exitCode = await sandbox.wait();
  expect(exitCode).toBe(0);
});

test("SandboxWithReadOnlyVolume", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = Image.fromRegistry("alpine:3.21");

  const volume = await Volume.fromName("libmodal-test-sandbox-volume", {
    createIfMissing: true,
  });

  const readOnlyVolume = volume.readOnly();
  expect(readOnlyVolume.isReadOnly).toBe(true);

  const sb = await app.createSandbox(image, {
    command: ["sh", "-c", "echo 'test' > /mnt/test/test.txt"],
    volumes: { "/mnt/test": readOnlyVolume },
  });

  expect(await sb.wait()).toBe(1);
  expect(await sb.stderr.readText()).toContain("Read-only file system");

  await sb.terminate();
});

test("SandboxWithTunnels", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const sandbox = await app.createSandbox(image, {
    command: ["cat"],
    encryptedPorts: [8443],
    unencryptedPorts: [8080],
  });

  expect(sandbox).toBeDefined();
  expect(sandbox.sandboxId).toMatch(/^sb-/);

  const tunnels = await sandbox.tunnels();
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

  await sandbox.terminate();
});

test("CreateSandboxWithSecrets", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const secret = await Secret.fromName("libmodal-test-secret", {
    requiredKeys: ["c"],
  });
  expect(secret).toBeDefined();

  const sandbox = await app.createSandbox(image, {
    command: ["printenv", "c"],
    secrets: [secret],
  });
  expect(sandbox).toBeDefined();

  const result = await sandbox.stdout.readText();
  expect(result).toBe("hello world\n");
});

test("CreateSandboxWithNetworkAccessParams", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const sb = await app.createSandbox(image, {
    command: ["echo", "hello, network access"],
    blockNetwork: false,
    cidrAllowlist: ["10.0.0.0/8", "192.168.0.0/16"],
  });

  onTestFinished(async () => {
    await sb.terminate();
  });

  expect(sb).toBeDefined();
  expect(sb.sandboxId).toMatch(/^sb-/);

  const exitCode = await sb.wait();
  expect(exitCode).toBe(0);

  await expect(
    app.createSandbox(image, {
      blockNetwork: false,
      cidrAllowlist: ["not-an-ip/8"],
    }),
  ).rejects.toThrow("Invalid CIDR: not-an-ip/8");

  await expect(
    app.createSandbox(image, {
      blockNetwork: true,
      cidrAllowlist: ["10.0.0.0/8"],
    }),
  ).rejects.toThrow(
    "cidrAllowlist cannot be used when blockNetwork is enabled",
  );
});

test("SandboxPollAndReturnCode", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const sandbox = await app.createSandbox(image, { command: ["cat"] });

  expect(await sandbox.poll()).toBeNull();

  // Send input to make the cat command complete
  await sandbox.stdin.writeText("hello, sandbox");
  await sandbox.stdin.close();

  expect(await sandbox.wait()).toBe(0);
  expect(await sandbox.poll()).toBe(0);
});

test("SandboxPollAfterFailure", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const sandbox = await app.createSandbox(image, {
    command: ["sh", "-c", "exit 42"],
  });

  expect(await sandbox.wait()).toBe(42);
  expect(await sandbox.poll()).toBe(42);
});

test("SandboxExecSecret", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const sb = await app.createSandbox(image);
  expect(sb.sandboxId).toBeTruthy();

  onTestFinished(async () => {
    await sb.terminate();
  });

  const secret = await Secret.fromName("libmodal-test-secret", {
    requiredKeys: ["c"],
  });
  const secret2 = await Secret.fromObject({ d: "3" });
  const printSecret = await sb.exec(["printenv", "c", "d"], {
    stdout: "pipe",
    secrets: [secret, secret2],
  });
  const secretText = await printSecret.stdout.readText();
  expect(secretText).toBe("hello world\n3\n");
});

test("SandboxFromId", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const sb = await app.createSandbox(image);
  onTestFinished(async () => {
    await sb.terminate();
  });
  const sbFromId = await Sandbox.fromId(sb.sandboxId);
  expect(sbFromId.sandboxId).toBe(sb.sandboxId);
});

test("SandboxWithWorkdir", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const sb = await app.createSandbox(image, {
    command: ["pwd"],
    workdir: "/tmp",
  });

  onTestFinished(async () => {
    await sb.terminate();
  });

  expect(await sb.stdout.readText()).toBe("/tmp\n");
});

test("SandboxWithWorkdirValidation", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  await expect(
    app.createSandbox(image, {
      workdir: "relative/path",
    }),
  ).rejects.toThrow("workdir must be an absolute path, got: relative/path");
});

test("SandboxSetTagsAndList", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const sb = await app.createSandbox(image);
  onTestFinished(async () => {
    await sb.terminate();
  });

  const unique = `${Math.random()}`;

  const foundBefore: string[] = [];
  for await (const s of Sandbox.list({ tags: { "test-key": unique } })) {
    foundBefore.push(s.sandboxId);
  }
  expect(foundBefore.length).toBe(0);

  await sb.setTags({ "test-key": unique });

  const foundAfter: string[] = [];
  for await (const s of Sandbox.list({ tags: { "test-key": unique } })) {
    foundAfter.push(s.sandboxId);
  }
  expect(foundAfter).toEqual([sb.sandboxId]);
});

test("SandboxSetMultipleTagsAndList", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const sb = await app.createSandbox(image);
  onTestFinished(async () => {
    await sb.terminate();
  });

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
  for await (const s of Sandbox.list({ tags: { "key-a": tagA } })) {
    ids.push(s.sandboxId);
  }
  expect(ids).toEqual([sb.sandboxId]);

  ids = [];
  for await (const s of Sandbox.list({
    tags: { "key-a": tagA, "key-b": tagB },
  })) {
    ids.push(s.sandboxId);
  }
  expect(ids).toEqual([sb.sandboxId]);

  ids = [];
  for await (const s of Sandbox.list({
    tags: { "key-a": tagA, "key-b": tagB, "key-d": "not-set" },
  })) {
    ids.push(s.sandboxId);
  }
  expect(ids.length).toBe(0);
});

test("SandboxListByAppId", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const sb = await app.createSandbox(image);
  onTestFinished(async () => {
    await sb.terminate();
  });

  let count = 0;
  for await (const s of Sandbox.list({ appId: app.appId })) {
    expect(s.sandboxId).toMatch(/^sb-/);
    count++;
    if (count > 0) break;
  }
  expect(count).toBeGreaterThan(0);
});

test("NamedSandbox", async () => {
  const app = await App.lookup("libmodal-test", { createIfMissing: true });
  const image = await app.imageFromRegistry("alpine:3.21");

  const sandboxName = `test-sandbox-${Math.random().toString().substring(2, 10)}`;

  const sb = await app.createSandbox(image, {
    name: sandboxName,
    command: ["sleep", "60"],
  });

  onTestFinished(async () => {
    await sb.terminate();
  });

  const sb1FromName = await Sandbox.fromName("libmodal-test", sandboxName);
  expect(sb1FromName.sandboxId).toBe(sb.sandboxId);
  const sb2FromName = await Sandbox.fromName("libmodal-test", sandboxName);
  expect(sb2FromName.sandboxId).toBe(sb1FromName.sandboxId);

  await expect(
    app.createSandbox(image, {
      name: sandboxName,
      command: ["sleep", "60"],
    }),
  ).rejects.toThrow("already exists");
});

test("NamedSandboxNotFound", async () => {
  await expect(
    Sandbox.fromName("libmodal-test", "non-existent-sandbox"),
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
