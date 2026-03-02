import { tc } from "../test-support/test-client";
import { expect, onTestFinished, test } from "vitest";
import { createMockModalClients } from "../test-support/grpc_mock";
import { Secret } from "../src/secret";
import { App } from "../src/app";

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

test("ImageFromAwsEcr", async () => {
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

  const image = await tc.images.fromRegistry("alpine:3.13").build(app);
  expect(image.imageId).toBeTruthy();
  expect(image.imageId).toMatch(/^im-/);

  const imageFromId = await tc.images.fromId(image.imageId);
  expect(imageFromId.imageId).toBe(image.imageId);

  await tc.images.delete(image.imageId);

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
