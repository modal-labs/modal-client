import { expect, onTestFinished, test, vi } from "vitest";
import { ClientError, Status } from "nice-grpc";

import { tc } from "../test-support/test-client";
import { createMockModalClients } from "../test-support/grpc_mock";
import {
  AlreadyExistsError,
  ClientClosedError,
  InvalidError,
  NotFoundError,
  SandboxFilesystemNotFoundError,
} from "../src/errors";
import { Image } from "../src/image";
import { Sandbox } from "../src/sandbox";
import {
  NetworkAccess_NetworkAccessType,
  SandboxContainerCreateV2Request,
  SandboxContainerCreateV2Response,
} from "../proto/modal_proto/api";
import { TaskContainerCreateRequest } from "../proto/modal_proto/task_command_router";
import { TaskCommandRouterClientImpl } from "../src/task_command_router_client";
import { Volume } from "../src/volume";

const V2_SANDBOX_ID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV";
const V1_SANDBOX_ID = "sb-nGEijt9WbBMlGrsPH9FOaC";

async function createSandbox(): Promise<Sandbox> {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = tc.images.fromRegistry("alpine:3.21");
  const sb = await tc.sandboxes.create(app, image, {
    command: ["sleep", "infinity"],
  });
  onTestFinished(async () => await sb.terminate());
  return sb;
}

async function buildAlpineImage(): Promise<Image> {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = await tc.images.fromRegistry("alpine:3.21").build(app);
  expect(image.imageId).toBeTruthy();
  return image;
}

test("SidecarBasicLifecycle", { timeout: 60_000 }, async () => {
  const sb = await createSandbox();
  const image = await buildAlpineImage();

  const container = await sb.experimentalSidecars.create("worker", image, {
    command: ["sleep", "100"],
  });
  expect(container.containerId).toBeTruthy();
  expect(container.containerName).toBe("worker");

  expect(await container.poll()).toBeNull();

  expect(await container.terminate({ wait: true })).toBe(137);
  expect(await container.wait()).toBe(137);
  expect(await container.poll()).toBe(137);

  const terminated = await sb.experimentalSidecars.list({
    includeTerminated: true,
  });
  expect(terminated.map((c) => c.containerName)).toEqual(["worker"]);
});

test("SidecarWaitAfterNaturalExit", async () => {
  const sb = await createSandbox();
  const image = await buildAlpineImage();

  const container = await sb.experimentalSidecars.create("oneshot", image, {
    command: ["sh", "-c", "exit 42"],
  });

  expect(await container.wait()).toBe(42);
  expect(await container.wait()).toBe(42);

  await expect(sb.experimentalSidecars.get("oneshot")).rejects.toThrow(
    NotFoundError,
  );

  const got = await sb.experimentalSidecars.get("oneshot", {
    includeTerminated: true,
  });
  expect(got.containerId).toBe(container.containerId);

  const replacement = await sb.experimentalSidecars.create("oneshot", image, {
    command: ["sleep", "100"],
  });
  expect(replacement.containerId).not.toBe(container.containerId);

  const listed = await sb.experimentalSidecars.list({
    includeTerminated: true,
  });
  const ids = listed.map((c) => c.containerId);
  expect(ids).toContain(container.containerId);
  expect(ids).toContain(replacement.containerId);
});

test("SidecarCreateRejectsMainName", async () => {
  const sb = await createSandbox();
  const image = await buildAlpineImage();

  await expect(
    sb.experimentalSidecars.create("main", image, {
      command: ["sleep", "100"],
    }),
  ).rejects.toThrow(InvalidError);

  await expect(
    sb.experimentalSidecars.create("", image, { command: ["sleep", "100"] }),
  ).rejects.toThrow(InvalidError);

  await expect(sb.experimentalSidecars.get("main")).rejects.toThrow(
    InvalidError,
  );

  for (const experimentalMemoryReserveConsumeMiB of [
    0,
    -1,
    1.5,
    NaN,
    Infinity,
    -Infinity,
  ]) {
    await expect(
      sb.experimentalSidecars.create("worker", image, {
        command: ["sleep", "100"],
        experimentalMemoryReserveConsumeMiB,
      }),
    ).rejects.toThrow(
      `experimentalMemoryReserveConsumeMiB must be a positive integer number of MiB, got: ${experimentalMemoryReserveConsumeMiB}`,
    );
  }
});

test("SidecarCreateImageMustBeBuilt", async () => {
  const sb = await createSandbox();

  const unbuilt = tc.images.fromRegistry("alpine:3.21");
  expect(unbuilt.imageId).toBe("");

  await expect(
    sb.experimentalSidecars.create("worker", unbuilt, {
      command: ["sleep", "100"],
    }),
  ).rejects.toThrow(InvalidError);
});

test("SidecarCreateForwardsSecretsAndEnv", async () => {
  const secret = await tc.secrets.fromObject({ API_KEY: "secret-value" });

  const sb = await createSandbox();
  const image = await buildAlpineImage();

  const container = await sb.experimentalSidecars.create("worker", image, {
    command: ["sleep", "100"],
    env: { API_KEY: "override", PLAIN_ENV: "plain" },
    secrets: [secret],
  });

  const proc = await container.exec([
    "sh",
    "-c",
    `printf '%s:%s' "$API_KEY" "$PLAIN_ENV"`,
  ]);
  const output = await proc.stdout.readText();
  expect(await proc.wait()).toBe(0);
  expect(output).toBe("override:plain");
});

test("SidecarCreateMountsVolume", async () => {
  const volume = await tc.volumes.ephemeral();
  onTestFinished(() => volume.closeEphemeral());

  const sb = await createSandbox();
  const image = await buildAlpineImage();

  const container = await sb.experimentalSidecars.create("worker", image, {
    command: ["sleep", "100"],
    volumes: { "/mnt/data": volume },
  });

  const write = await container.exec([
    "sh",
    "-c",
    "echo volume-works > /mnt/data/marker.txt",
  ]);
  expect(await write.wait()).toBe(0);

  const read = await container.exec(["cat", "/mnt/data/marker.txt"]);
  const output = await read.stdout.readText();
  expect(await read.wait()).toBe(0);
  expect(output.trim()).toBe("volume-works");
});

test("SidecarExec", async () => {
  const sb = await createSandbox();
  const image = await buildAlpineImage();

  const container = await sb.experimentalSidecars.create("worker", image, {
    command: ["sleep", "100"],
  });

  const proc = await container.exec(["echo", "hello"]);
  expect(await proc.stdout.readText()).toBe("hello\n");
  expect(await proc.wait()).toBe(0);
});

test("SidecarFilesystem", async () => {
  const sb = await createSandbox();
  const image = await buildAlpineImage();

  const container = await sb.experimentalSidecars.create("worker", image, {
    command: ["sleep", "100"],
  });

  await container.filesystem.writeText("hi from sidecar", "/tmp/sidecar-hello");
  expect(await container.filesystem.readText("/tmp/sidecar-hello")).toBe(
    "hi from sidecar",
  );

  // The main container should not see the file in the sidecar's filesystem.
  await expect(sb.filesystem.stat("/tmp/sidecar-hello")).rejects.toThrow(
    SandboxFilesystemNotFoundError,
  );
});

test("sidecar create sends SandboxContainerCreateV2 to the control plane", async () => {
  vi.stubEnv("MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE", "1");
  onTestFinished(() => {
    vi.unstubAllEnvs();
  });
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, { taskId: "ta-v2-123" });

  let request: SandboxContainerCreateV2Request | undefined;
  mock.handleUnary("/SandboxContainerCreateV2", (req) => {
    request = req as SandboxContainerCreateV2Request;
    return SandboxContainerCreateV2Response.create({
      containerId: "sb-test-ctr-SIDECAR123",
    });
  });

  const container = await sb.experimentalSidecars.create(
    "worker",
    new Image(mc, "im-built", ""),
    {
      command: ["sleep", "100"],
      env: { PLAIN_ENV: "plain" },
      workdir: "/app",
      volumes: {
        "/mnt/data": new Volume("vo-plain"),
        "/mnt/scoped": new Volume("vo-scoped").withMountOptions({
          readOnly: true,
          subPath: "/inner",
        }),
      },
      outboundCidrAllowlist: ["10.0.0.0/8"],
      outboundDomainAllowlist: ["example.com"],
      pty: true,
    },
  );

  expect(container.containerId).toBe("sb-test-ctr-SIDECAR123");
  // The response carries no name here, so the requested name is used.
  expect(container.containerName).toBe("worker");

  expect(request?.sandboxId).toBe(V2_SANDBOX_ID);
  expect(request?.containerName).toBe("worker");
  expect(request?.definition?.imageId).toBe("im-built");
  expect(request?.definition?.entrypointArgs).toEqual(["sleep", "100"]);
  expect(request?.definition?.workdir).toBe("/app");
  expect(request?.definition?.secretIds).toEqual([]);
  expect(request?.definition?.volumeMounts).toEqual([
    {
      volumeId: "vo-plain",
      mountPath: "/mnt/data",
      allowBackgroundCommits: true,
      readOnly: false,
      subPath: undefined,
    },
    {
      volumeId: "vo-scoped",
      mountPath: "/mnt/scoped",
      allowBackgroundCommits: true,
      readOnly: true,
      subPath: "/inner",
    },
  ]);
  expect(request?.definition?.ptyInfo).toBeDefined();
  expect(request?.definition?.networkAccess?.networkAccessType).toBe(
    NetworkAccess_NetworkAccessType.ALLOWLIST,
  );
  expect(request?.definition?.networkAccess?.allowedCidrs).toEqual([
    "10.0.0.0/8",
  ]);
  expect(request?.definition?.networkAccess?.allowedDomains).toEqual([
    "example.com",
  ]);
  // Env vars travel as ephemeral secrets, not on the definition.
  expect(request?.ephemeralSecrets?.contents).toEqual({ PLAIN_ENV: "plain" });
  expect(request?.definition?.environmentVariables).toBeUndefined();

  mock.assertExhausted();
});

test("sidecar create omits ephemeral secrets when no env vars are set", async () => {
  vi.stubEnv("MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE", "1");
  onTestFinished(() => {
    vi.unstubAllEnvs();
  });
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, { taskId: "ta-v2-123" });

  let request: SandboxContainerCreateV2Request | undefined;
  mock.handleUnary("/SandboxContainerCreateV2", (req) => {
    request = req as SandboxContainerCreateV2Request;
    return SandboxContainerCreateV2Response.create({
      containerId: "sb-test-ctr-SIDECAR123",
      containerName: "worker",
    });
  });

  await sb.experimentalSidecars.create("worker", new Image(mc, "im-built", ""));

  expect(request?.ephemeralSecrets).toBeUndefined();
  expect(request?.definition?.workdir).toBeUndefined();
  expect(request?.definition?.volumeMounts).toEqual([]);
  expect(request?.definition?.ptyInfo).toBeUndefined();
  mock.assertExhausted();
});

test("sidecar create sends volume mounts over the command router", async () => {
  const { mockClient: mc } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, { taskId: "ta-v2-123" });

  const containerCreate = vi.fn().mockResolvedValue({
    containerId: "sb-test-ctr-SIDECAR123",
    containerName: "worker",
  });
  const tryInit = vi
    .spyOn(TaskCommandRouterClientImpl, "tryInit")
    .mockResolvedValue({
      containerCreate,
      close: vi.fn(),
    } as unknown as TaskCommandRouterClientImpl);
  onTestFinished(() => tryInit.mockRestore());

  await sb.experimentalSidecars.create(
    "worker",
    new Image(mc, "im-built", ""),
    {
      volumes: {
        "/mnt/data": new Volume("vo-plain"),
        "/mnt/scoped": new Volume("vo-scoped").withMountOptions({
          readOnly: true,
          subPath: "/inner",
        }),
      },
    },
  );

  expect(containerCreate).toHaveBeenCalledTimes(1);
  const request = containerCreate.mock
    .calls[0][0] as TaskContainerCreateRequest;
  expect(request.taskId).toBe("ta-v2-123");
  expect(request.volumeMounts).toEqual([
    {
      volumeId: "vo-plain",
      mountPath: "/mnt/data",
      allowBackgroundCommits: true,
      readOnly: false,
      subPath: undefined,
    },
    {
      volumeId: "vo-scoped",
      mountPath: "/mnt/scoped",
      allowBackgroundCommits: true,
      readOnly: true,
      subPath: "/inner",
    },
  ]);
});

test("sidecar create rejects invalid volume entries", async () => {
  onTestFinished(() => {
    vi.unstubAllEnvs();
  });
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, { taskId: "ta-v2-123" });
  const tryInit = vi.spyOn(TaskCommandRouterClientImpl, "tryInit");
  onTestFinished(() => tryInit.mockRestore());

  const sharedVolume = new Volume("vo-shared");
  const cases: [Record<string, Volume>, string][] = [
    [{ "/mnt/data": null as unknown as Volume }, '"/mnt/data"'],
    [
      {
        "/mnt/b": sharedVolume,
        "/mnt/a": sharedVolume.withMountOptions({ readOnly: true }),
      },
      "/mnt/a, /mnt/b",
    ],
  ];
  for (const [volumes, expectedMessage] of cases) {
    for (const useControlPlane of ["1", ""]) {
      vi.stubEnv("MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE", useControlPlane);
      const create = sb.experimentalSidecars.create(
        "worker",
        new Image(mc, "im-built", ""),
        { volumes },
      );
      await expect(create).rejects.toThrow(InvalidError);
      await expect(create).rejects.toThrow(expectedMessage);
    }
  }

  expect(tryInit).not.toHaveBeenCalled();
  mock.assertExhausted();
});

// The three tests below register a Modal server create handler they expect
// never to fire, so they deliberately skip mock.assertExhausted(). If a guard
// ever inverts, the create succeeds and usedControlPlane flips, failing the test.

test("sidecar create uses the command router by default", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, { taskId: "ta-v2-123" });

  let usedControlPlane = false;
  mock.handleUnary("/SandboxContainerCreateV2", () => {
    usedControlPlane = true;
    return SandboxContainerCreateV2Response.create({
      containerId: "sb-test-ctr-SIDECAR123",
    });
  });

  // Without the opt-in this goes over the Sandbox connection, which is not
  // mocked here; what matters is that nothing reached the Modal server.
  await expect(
    sb.experimentalSidecars.create("worker", new Image(mc, "im-built", "")),
  ).rejects.toThrow();
  expect(usedControlPlane).toBe(false);
});

test("sidecar create ignores the opt-in for a V1 sandbox", async () => {
  vi.stubEnv("MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE", "1");
  onTestFinished(() => {
    vi.unstubAllEnvs();
  });
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  const sb = new Sandbox(mc, V1_SANDBOX_ID, { taskId: "ta-v1-123" });

  let usedControlPlane = false;
  mock.handleUnary("/SandboxContainerCreateV2", () => {
    usedControlPlane = true;
    return SandboxContainerCreateV2Response.create({
      containerId: "sb-test-ctr-SIDECAR123",
    });
  });

  // The opt-in only applies to V2 Sandboxes; a V1 Sandbox always creates
  // sidecars over the Sandbox connection.
  await expect(
    sb.experimentalSidecars.create("worker", new Image(mc, "im-built", "")),
  ).rejects.toThrow();
  expect(usedControlPlane).toBe(false);
});

test("sidecar create rejects a detached sandbox", async () => {
  vi.stubEnv("MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE", "1");
  onTestFinished(() => {
    vi.unstubAllEnvs();
  });
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, { taskId: "ta-v2-123" });

  let usedControlPlane = false;
  mock.handleUnary("/SandboxContainerCreateV2", () => {
    usedControlPlane = true;
    return SandboxContainerCreateV2Response.create({
      containerId: "sb-test-ctr-SIDECAR123",
    });
  });

  // The service is obtained while attached; detaching afterwards must still
  // stop it from creating containers.
  const sidecars = sb.experimentalSidecars;
  sb.detach();

  await expect(
    sidecars.create("worker", new Image(mc, "im-built", "")),
  ).rejects.toThrow(ClientClosedError);
  expect(usedControlPlane).toBe(false);
});

test("sidecar create maps control plane errors", async () => {
  vi.stubEnv("MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE", "1");
  onTestFinished(() => {
    vi.unstubAllEnvs();
  });
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  const sb = new Sandbox(mc, V2_SANDBOX_ID, { taskId: "ta-v2-123" });

  mock.handleUnary("/SandboxContainerCreateV2", () => {
    throw new ClientError(
      "/modal.client.ModalClient/SandboxContainerCreateV2",
      Status.ALREADY_EXISTS,
      "sidecar already exists",
    );
  });
  await expect(
    sb.experimentalSidecars.create("worker", new Image(mc, "im-built", "")),
  ).rejects.toThrow(AlreadyExistsError);

  mock.handleUnary("/SandboxContainerCreateV2", () => {
    throw new ClientError(
      "/modal.client.ModalClient/SandboxContainerCreateV2",
      Status.INVALID_ARGUMENT,
      "bad definition",
    );
  });
  await expect(
    sb.experimentalSidecars.create("worker", new Image(mc, "im-built", "")),
  ).rejects.toThrow(InvalidError);

  mock.assertExhausted();
});
