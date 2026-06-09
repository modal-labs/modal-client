import { tc } from "../test-support/test-client";
import { ModalClient } from "../src/client";
import { expect, onTestFinished, test, vi } from "vitest";
import { createMockModalClients } from "../test-support/grpc_mock";
import { Secret } from "../src/secret";
import { App } from "../src/app";
import { Image } from "modal";

test("ImageFromId", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });

  const image = await tc.images.fromRegistry("alpine:3.21").build(app);
  expect(image.imageId).toBeTruthy();

  const imageFromId = await tc.images.fromId(image.imageId);
  expect(imageFromId.imageId).toBe(image.imageId);

  await expect(tc.images.fromId("im-nonexistent")).rejects.toThrow();
});

test("ImageFromName", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/ImageGetByTag", (req: any) => {
    expect(req).toMatchObject({
      environmentName: "dev",
      tag: "analytics-runtime:v1",
    });
    return { imageId: "im-tagged" };
  });

  const image = await mc.images.fromName("analytics-runtime:v1", {
    environment: "dev",
  });
  expect(image.imageId).toBe("im-tagged");

  mock.assertExhausted();
});

test("ImageNamedRefsRejectImageIDNames", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  await expect(mc.images.fromName("im-looks-like-an-id")).rejects.toThrow(
    "cannot start with 'im-'",
  );

  const image = new Image(mc, "im-built", "");
  await expect(image.publish("im-looks-like-an-id")).rejects.toThrow(
    "cannot start with 'im-'",
  );

  mock.assertExhausted();
});

test("ImageFromRegistry", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });

  const image = await tc.images.fromRegistry("alpine:3.21").build(app);
  expect(image.imageId).toBeTruthy();
  expect(image.imageId).toMatch(/^im-/);
});

test("ImageFromRegistryWithSecret", async () => {
  // GCP Artifact Registry also supports auth using username and password, if the username is "_json_key"
  // and the password is the service account JSON blob. See:
  // https://cloud.google.com/artifact-registry/docs/docker/authentication#json-key
  // So we use GCP Artifact Registry to test this too.

  // Original image was built with 2024.10, so we set `MODAL_IMAGE_BUILDER_VERSION` to not trigger
  // the image builder.
  vi.stubEnv("MODAL_IMAGE_BUILDER_VERSION", "2024.10");
  onTestFinished(() => {
    vi.unstubAllEnvs();
  });
  // Use a new client to pull in MODAL_IMAGE_BUILDER_VERSION
  const tc = new ModalClient();

  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });

  const image = await tc.images
    .fromRegistry(
      "us-east1-docker.pkg.dev/modal-prod-367916/private-repo-test/my-image",
      await tc.secrets.fromName("libmodal-gcp-artifact-registry-test", {
        requiredKeys: ["REGISTRY_USERNAME", "REGISTRY_PASSWORD"],
      }),
    )
    .build(app);
  expect(image.imageId).toBeTruthy();
  expect(image.imageId).toMatch(/^im-/);
});

test("ImagePublish", async () => {
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();
  const image = new Image(mc, "im-built", "");

  mock.handleUnary("/ImagePublish", (req: any) => {
    expect(req).toMatchObject({
      imageId: "im-built",
      environmentName: "dev",
      isPublic: false,
      tag: "analytics-runtime:v1",
    });
    return {
      imageId: "im-built",
      revisionId: "ir-01H00000000000000000000000",
    };
  });

  await image.publish("analytics-runtime:v1", {
    environment: "dev",
  });

  mock.assertExhausted();
});

test("ImagePublishRequiresBuiltImage", async () => {
  const { mockClient: mc } = createMockModalClients();
  const image = new Image(mc, "", "alpine:3.21");

  await expect(image.publish("analytics-runtime")).rejects.toThrow(
    "Cannot publish an image that has not been built yet",
  );
});

test("ImageFromAwsEcr", async () => {
  // Original image was built with 2024.10, so we set `MODAL_IMAGE_BUILDER_VERSION` to not trigger
  // the image builder.
  vi.stubEnv("MODAL_IMAGE_BUILDER_VERSION", "2024.10");
  onTestFinished(() => {
    vi.unstubAllEnvs();
  });

  // Use a new client to pull in MODAL_IMAGE_BUILDER_VERSION
  const tc = new ModalClient();
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });

  const image = await tc.images
    .fromAwsEcr(
      "459781239556.dkr.ecr.us-east-1.amazonaws.com/ecr-private-registry-test-7522615:python",
      await tc.secrets.fromName("libmodal-aws-ecr-test", {
        requiredKeys: ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
      }),
    )
    .build(app);
  expect(image.imageId).toBeTruthy();
  expect(image.imageId).toMatch(/^im-/);
});

test("ImageFromGcpArtifactRegistry", { timeout: 30_000 }, async () => {
  // Original image was built with 2024.10, so we set `MODAL_IMAGE_BUILDER_VERSION` to not trigger
  // the image builder.
  vi.stubEnv("MODAL_IMAGE_BUILDER_VERSION", "2024.10");
  onTestFinished(() => {
    vi.unstubAllEnvs();
  });
  // Use a new client to pull in MODAL_IMAGE_BUILDER_VERSION
  const tc = new ModalClient();

  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });

  const image = await tc.images
    .fromGcpArtifactRegistry(
      "us-east1-docker.pkg.dev/modal-prod-367916/private-repo-test/my-image",
      await tc.secrets.fromName("libmodal-gcp-artifact-registry-test", {
        requiredKeys: ["SERVICE_ACCOUNT_JSON"],
      }),
    )
    .build(app);
  expect(image.imageId).toBeTruthy();
  expect(image.imageId).toMatch(/^im-/);
});

test("ImageDelete", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });

  const baseImage = await tc.images.fromRegistry("alpine:3.13").build(app);
  const image = await baseImage
    .dockerfileCommands(["RUN touch dummy.txt"])
    .build(app);
  expect(baseImage.imageId).toBeTruthy();
  expect(image.imageId).toBeTruthy();
  expect(image.imageId).toMatch(/^im-/);

  const imageFromId = await tc.images.fromId(image.imageId);
  expect(imageFromId.imageId).toBe(image.imageId);

  await tc.images.delete(image.imageId);

  await tc.images.fromId(baseImage.imageId); // should not throw
  await expect(tc.images.fromId(image.imageId)).rejects.toThrow(
    "Could not find image with ID",
  );

  const newImage = await tc.images.fromRegistry("alpine:3.13").build(app);
  expect(newImage.imageId).toBeTruthy();
  expect(newImage.imageId).not.toBe(image.imageId);

  await expect(tc.images.delete("im-nonexistent")).rejects.toThrow(
    "No Image with ID",
  );
});

test("DockerfileCommands", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });

  const image = tc.images
    .fromRegistry("alpine:3.21")
    .dockerfileCommands(["RUN echo hey > /root/hello.txt"]);

  const sb = await tc.sandboxes.create(app, image, {
    command: ["cat", "/root/hello.txt"],
  });
  onTestFinished(async () => await sb.terminate());

  const stdout = await sb.stdout.readText();
  expect(stdout).toBe("hey\n");
});

test("DockerfileCommandsEmptyArrayNoOp", () => {
  const image1 = tc.images.fromRegistry("alpine:3.21");
  const image2 = image1.dockerfileCommands([]);
  expect(image2).toBe(image1);
});

test("DockerfileCommandsFromNameUsesResolvedImageAsBase", async () => {
  vi.stubEnv("MODAL_IMAGE_BUILDER_VERSION", "2024.10");
  onTestFinished(() => {
    vi.unstubAllEnvs();
  });
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/ImageGetByTag", (req: any) => {
    expect(req).toMatchObject({
      tag: "analytics-runtime:latest",
    });
    return { imageId: "im-tagged" };
  });

  mock.handleUnary("/ImageGetOrCreate", (req: any) => {
    expect(req).toMatchObject({
      appId: "ap-test",
      image: {
        dockerfileCommands: ["FROM base", "RUN echo layer"],
        baseImages: [{ dockerTag: "base", imageId: "im-tagged" }],
      },
    });
    return { imageId: "im-layer", result: { status: 1 } };
  });

  const image = await mc.images.fromName("analytics-runtime");
  const builtImage = await image
    .dockerfileCommands(["RUN echo layer"])
    .build(new App("ap-test", "libmodal-test"));

  expect(builtImage.imageId).toBe("im-layer");
  mock.assertExhausted();
});

test("DockerfileCommandsFromBuiltRegistryImageDropsRegistryConfig", async () => {
  vi.stubEnv("MODAL_IMAGE_BUILDER_VERSION", "2024.10");
  onTestFinished(() => {
    vi.unstubAllEnvs();
  });
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/ImageGetOrCreate", (req: any) => {
    expect(req).toMatchObject({
      appId: "ap-test",
      image: {
        dockerfileCommands: ["FROM private.example.com/app:latest"],
        baseImages: [],
        imageRegistryConfig: {
          secretId: "sc-registry",
        },
      },
    });
    return { imageId: "im-private-base", result: { status: 1 } };
  });

  mock.handleUnary("/ImageGetOrCreate", (req: any) => {
    expect(req).toMatchObject({
      appId: "ap-test",
      image: {
        dockerfileCommands: ["FROM base", "RUN echo layer"],
        baseImages: [{ dockerTag: "base", imageId: "im-private-base" }],
      },
    });
    expect(req.image.imageRegistryConfig).toBeUndefined();
    return { imageId: "im-layer", result: { status: 1 } };
  });

  const baseImage = await mc.images
    .fromRegistry(
      "private.example.com/app:latest",
      new Secret("sc-registry", "registry"),
    )
    .build(new App("ap-test", "libmodal-test"));
  const builtImage = await baseImage
    .dockerfileCommands(["RUN echo layer"])
    .build(new App("ap-test", "libmodal-test"));

  expect(builtImage.imageId).toBe("im-layer");
  mock.assertExhausted();
});

test("DockerfileCommandsChaining", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });

  const image = tc.images
    .fromRegistry("alpine:3.21")
    .dockerfileCommands(["RUN echo ${SECRET:-unset} > /root/layer1.txt"])
    .dockerfileCommands(["RUN echo ${SECRET:-unset} > /root/layer2.txt"], {
      secrets: [await tc.secrets.fromObject({ SECRET: "hello" })],
    })
    .dockerfileCommands(["RUN echo ${SECRET:-unset} > /root/layer3.txt"]);

  const sb = await tc.sandboxes.create(app, image, {
    command: [
      "cat",
      "/root/layer1.txt",
      "/root/layer2.txt",
      "/root/layer3.txt",
    ],
  });
  onTestFinished(async () => await sb.terminate());

  const stdout = await sb.stdout.readText();
  expect(stdout).toBe("unset\nhello\nunset\n");
});

test("DockerfileCommandsCopyCommandValidation", () => {
  expect(() => {
    tc.images
      .fromRegistry("alpine:3.21")
      .dockerfileCommands([
        "COPY --from=alpine:latest /etc/os-release /tmp/os-release",
      ]);
  }).not.toThrow();

  expect(() => {
    tc.images
      .fromRegistry("alpine:3.21")
      .dockerfileCommands(["COPY ./file.txt /root/"]);
  }).toThrow(
    "COPY commands that copy from local context are not yet supported",
  );

  expect(() => {
    tc.images
      .fromRegistry("alpine:3.21")
      .dockerfileCommands(["RUN echo 'COPY ./file.txt /root/'"]);
  }).not.toThrow();

  expect(() => {
    tc.images
      .fromRegistry("alpine:3.21")
      .dockerfileCommands([
        "RUN echo hey",
        "copy ./file.txt /root/",
        "RUN echo hey",
      ]);
  }).toThrow(
    "COPY commands that copy from local context are not yet supported",
  );
});

test("DockerfileCommandsWithOptions", async () => {
  // Original image was built with 2024.10, so we set `MODAL_IMAGE_BUILDER_VERSION` to use local image builder version.
  vi.stubEnv("MODAL_IMAGE_BUILDER_VERSION", "2024.10");
  onTestFinished(() => {
    vi.unstubAllEnvs();
  });
  const { mockClient: mc, mockCpClient: mock } = createMockModalClients();

  mock.handleUnary("/ImageGetOrCreate", (req: any) => {
    expect(req).toMatchObject({
      appId: "ap-test",
      image: {
        dockerfileCommands: ["FROM alpine:3.21"],
        secretIds: [],
        baseImages: [],
        gpuConfig: undefined,
      },
      forceBuild: false,
    });
    return { imageId: "im-base", result: { status: 1 } };
  });

  mock.handleUnary("/ImageGetOrCreate", (req: any) => {
    expect(req).toMatchObject({
      appId: "ap-test",
      image: {
        dockerfileCommands: ["FROM base", "RUN echo layer1"],
        secretIds: [],
        baseImages: [{ dockerTag: "base", imageId: "im-base" }],
        gpuConfig: undefined,
      },
      forceBuild: false,
    });
    return { imageId: "im-layer1", result: { status: 1 } };
  });

  mock.handleUnary("/ImageGetOrCreate", (req: any) => {
    expect(req).toMatchObject({
      appId: "ap-test",
      image: {
        dockerfileCommands: ["FROM base", "RUN echo layer2"],
        secretIds: ["sc-test"],
        baseImages: [{ dockerTag: "base", imageId: "im-layer1" }],
        gpuConfig: {
          type: 0,
          count: 1,
          gpuType: "A100",
        },
      },
      forceBuild: true,
    });
    return { imageId: "im-layer2", result: { status: 1 } };
  });

  mock.handleUnary("/ImageGetOrCreate", (req: any) => {
    expect(req).toMatchObject({
      appId: "ap-test",
      image: {
        dockerfileCommands: ["FROM base", "RUN echo layer3"],
        secretIds: [],
        baseImages: [{ dockerTag: "base", imageId: "im-layer2" }],
        gpuConfig: undefined,
      },
      forceBuild: true,
    });
    return { imageId: "im-layer3", result: { status: 1 } };
  });

  const image = await mc.images
    .fromRegistry("alpine:3.21")
    .dockerfileCommands(["RUN echo layer1"])
    .dockerfileCommands(["RUN echo layer2"], {
      secrets: [new Secret("sc-test", "test_secret")],
      gpu: "A100",
      forceBuild: true,
    })
    .dockerfileCommands(["RUN echo layer3"], {
      forceBuild: true,
    })
    .build(new App("ap-test", "libmodal-test"));

  expect(image.imageId).toBe("im-layer3");

  mock.assertExhausted();
});
