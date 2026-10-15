import { tc } from "../test-support/test-client";
import { expect, onTestFinished, test } from "vitest";
import { createMockModalClients } from "../test-support/grpc_mock";
import { InvalidError, NotFoundError } from "../src/errors";
import { volumeToMountProto } from "../src/volume";
import { ClientError, Status } from "nice-grpc";
import { ObjectCreationType, VolumeFsVersion } from "../proto/modal_proto/api";

test("client.volumes.fromName", async () => {
  const volume = await tc.volumes.fromName("libmodal-test-volume", {
    createIfMissing: true,
  });
  expect(volume.volumeId).toMatch(/^vo-/);
  expect(volume.name).toBe("libmodal-test-volume");

  const promise = tc.volumes.fromName("missing-volume");
  await expect(promise).rejects.toThrow(/Volume 'missing-volume' not found/);
});

test("client.volumes.fromName version: v2 with create if missing", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/VolumeGetOrCreate", (req: any) => {
    expect(req).toMatchObject({
      version: VolumeFsVersion.VOLUME_FS_VERSION_V2,
      objectCreationType:
        ObjectCreationType.OBJECT_CREATION_TYPE_CREATE_IF_MISSING,
    });
    return {
      volumeId: "vo-v2",
      metadata: { version: VolumeFsVersion.VOLUME_FS_VERSION_V2 },
    };
  });

  const volume = await mc.volumes.fromName("test-volume", {
    version: 2,
    createIfMissing: true,
  });
  expect(volume.volumeId).toBe("vo-v2");
  mock.assertExhausted();
});

test("client.volumes.fromName version: mismatched response", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/VolumeGetOrCreate", (req: any) => {
    expect(req.version).toBe(VolumeFsVersion.VOLUME_FS_VERSION_V2);
    return {
      volumeId: "vo-v1",
      metadata: { version: VolumeFsVersion.VOLUME_FS_VERSION_V1 },
    };
  });

  const error = await mc.volumes.fromName("test-volume", { version: 2 }).then(
    () => undefined,
    (err: unknown) => err,
  );
  expect(error).toBeInstanceOf(InvalidError);
  expect(error).toHaveProperty(
    "message",
    "Volume 'test-volume' exists but has version v1, not v2 as requested. " +
      "To access this Volume, either omit the version parameter or use the correct version.",
  );
  mock.assertExhausted();
});

test("client.volumes.fromName version: v1 accepts unspecified response", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/VolumeGetOrCreate", (req: any) => {
    expect(req.version).toBe(VolumeFsVersion.VOLUME_FS_VERSION_V1);
    return {
      volumeId: "vo-v1",
      metadata: { version: VolumeFsVersion.VOLUME_FS_VERSION_UNSPECIFIED },
    };
  });

  const volume = await mc.volumes.fromName("test-volume", { version: 1 });
  expect(volume.volumeId).toBe("vo-v1");
  mock.assertExhausted();
});

test("client.volumes.fromName version: v2 accepts matching response", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/VolumeGetOrCreate", (req: any) => {
    expect(req.version).toBe(VolumeFsVersion.VOLUME_FS_VERSION_V2);
    return {
      volumeId: "vo-v2",
      metadata: { version: VolumeFsVersion.VOLUME_FS_VERSION_V2 },
    };
  });

  const volume = await mc.volumes.fromName("test-volume", { version: 2 });
  expect(volume.volumeId).toBe("vo-v2");
  mock.assertExhausted();
});

test("client.volumes.fromName version: invalid version makes no RPC", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  const error = await mc.volumes
    .fromName("test-volume", { version: 3 as any })
    .then(
      () => undefined,
      (err: unknown) => err,
    );
  expect(error).toBeInstanceOf(InvalidError);
  expect(error).toHaveProperty(
    "message",
    "VolumeFS version must be either 1 or 2",
  );
  mock.assertExhausted();
});

test("client.volumes.fromName version: omitted accepts any response", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/VolumeGetOrCreate", (req: any) => {
    expect(req.version).toBe(VolumeFsVersion.VOLUME_FS_VERSION_UNSPECIFIED);
    return {
      volumeId: "vo-v2",
      metadata: { version: VolumeFsVersion.VOLUME_FS_VERSION_V2 },
    };
  });

  const volume = await mc.volumes.fromName("test-volume");
  expect(volume.volumeId).toBe("vo-v2");
  mock.assertExhausted();
});

test("Volume.withMountOptions", async () => {
  const volume = await tc.volumes.fromName("libmodal-test-volume", {
    createIfMissing: true,
  });

  const mount = volume.withMountOptions({
    readOnly: true,
    subPath: "/items",
  });
  expect(mount.volumeId).toBe(volume.volumeId);

  const mountProto = volumeToMountProto("/mnt", mount);
  expect(mountProto.readOnly).toBe(true);
  expect(mountProto.subPath).toBe("/items");

  const unconfiguredProto = volumeToMountProto("/mnt", volume);
  expect(unconfiguredProto.readOnly).toBe(false);
  expect(unconfiguredProto.subPath).toBeUndefined();
});

test("Volume.withMountOptions stacks", async () => {
  const volume = await tc.volumes.fromName("libmodal-test-volume", {
    createIfMissing: true,
  });

  const configured = volume.withMountOptions({
    readOnly: true,
    subPath: "/nested",
  });

  // Setting only subPath preserves readOnly from the previous call.
  const withNewSubPath = configured.withMountOptions({ subPath: "/other" });
  const newSubPathProto = volumeToMountProto("/mnt", withNewSubPath);
  expect(newSubPathProto.readOnly).toBe(true);
  expect(newSubPathProto.subPath).toBe("/other");

  // Setting only readOnly preserves subPath from the previous call.
  const withReadOnlyDisabled = configured.withMountOptions({ readOnly: false });
  const readOnlyDisabledProto = volumeToMountProto(
    "/mnt",
    withReadOnlyDisabled,
  );
  expect(readOnlyDisabledProto.readOnly).toBe(false);
  expect(readOnlyDisabledProto.subPath).toBe("/nested");

  // subPath "/" is normalized to undefined (mount the whole volume).
  const withClearedSubPath = configured.withMountOptions({ subPath: "/" });
  const clearedSubPathProto = volumeToMountProto("/mnt", withClearedSubPath);
  expect(clearedSubPathProto.readOnly).toBe(true);
  expect(clearedSubPathProto.subPath).toBeUndefined();
});

test("VolumeEphemeral", async () => {
  const volume = await tc.volumes.ephemeral();
  onTestFinished(() => volume.closeEphemeral());

  expect(volume.name).toBeUndefined();
  expect(volume.volumeId).toMatch(/^vo-/);
  const readOnlyVolume = volume.withMountOptions({ readOnly: true });
  expect(volumeToMountProto("/mnt", readOnlyVolume).readOnly).toBe(true);
});

test("VolumeDelete success", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/VolumeGetOrCreate", () => ({
    volumeId: "vo-test-123",
    metadata: { name: "test-volume" },
  }));

  mock.handleUnary("/VolumeDelete", (req: any) => {
    expect(req.volumeId).toBe("vo-test-123");
    return {};
  });

  await mc.volumes.delete("test-volume");
  mock.assertExhausted();
});

test("VolumeDelete with allowMissing=true", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/VolumeGetOrCreate", () => {
    throw new NotFoundError("Volume 'missing' not found");
  });

  await mc.volumes.delete("missing", { allowMissing: true });
  mock.assertExhausted();
});

test("VolumeDelete with allowMissing=true when delete RPC returns NOT_FOUND", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  // fromName succeeds — volume exists at lookup time
  mock.handleUnary("/VolumeGetOrCreate", () => ({
    volumeId: "vo-test-123",
    metadata: { name: "test-volume" },
  }));

  // volumeDelete fails — volume was deleted between lookup and delete
  mock.handleUnary("/VolumeDelete", () => {
    throw new ClientError(
      "/modal.client.ModalClient/VolumeDelete",
      Status.NOT_FOUND,
      "No Volume with ID 'vo-test-123' found",
    );
  });

  // With allowMissing=true, this should succeed silently
  await mc.volumes.delete("test-volume", { allowMissing: true });
  mock.assertExhausted();
});

test("VolumeDelete with allowMissing=false throws", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/VolumeGetOrCreate", () => {
    throw new NotFoundError("Volume 'missing' not found");
  });

  await expect(
    mc.volumes.delete("missing", { allowMissing: false }),
  ).rejects.toThrow(NotFoundError);
});
