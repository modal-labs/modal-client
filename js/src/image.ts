import {
  GenericResult,
  GenericResult_GenericStatus,
  RegistryAuthType,
  ImageRegistryConfig,
  Image as ImageProto,
  GPUConfig,
} from "../proto/modal_proto/api";
import { type ModalClient } from "./client";
import { App, parseGpuConfig } from "./app";
import { Secret, mergeEnvIntoSecrets } from "./secret";
import { ClientError } from "nice-grpc";
import { Status } from "nice-grpc";
import { NotFoundError, InvalidError } from "./errors";
import { checkObjectName } from "./name_utils";

const DEFAULT_IMAGE_TAG = "latest";

function validateImageName(name: string): void {
  checkObjectName(name, "Image");
  if (name.startsWith("im-")) {
    throw new InvalidError(
      "Image name cannot start with 'im-' (reserved for image IDs).",
    );
  }
}

function validateImageTag(tag: string): void {
  checkObjectName(tag, "Image tag");
}

/**
 * Parse an image reference, returning [namespacePrefix, nameTag].
 *
 * If the name contains a '/', the part before the last '/' is extracted as
 * a namespace prefix (intended for environment/name or workspace/env/name
 * syntax). The actual image name (after the last '/') is validated as a
 * standard image name.
 */
function parseNamedImageRef(value: string): [string, string] {
  const separatorIndex = value.indexOf(":");
  let imageName: string;
  let tag: string;
  if (separatorIndex === -1) {
    imageName = value;
    tag = DEFAULT_IMAGE_TAG;
  } else {
    imageName = value.slice(0, separatorIndex);
    tag = value.slice(separatorIndex + 1);
  }

  let prefix = "";
  const lastSlashIndex = imageName.lastIndexOf("/");
  if (lastSlashIndex !== -1) {
    prefix = imageName.slice(0, lastSlashIndex);
    const after = imageName.slice(lastSlashIndex + 1);
    if (prefix === "") {
      throw new InvalidError(
        "Invalid Image name: '/' prefix must be non-empty.",
      );
    }
    if (after === "") {
      throw new InvalidError(
        "Invalid Image name: name after '/' must be non-empty.",
      );
    }
    imageName = after;
  }

  validateImageName(imageName);
  validateImageTag(tag);

  const fullName = prefix ? `${prefix}/${imageName}` : imageName;
  return [prefix, `${fullName}:${tag}`];
}

/** Optional parameters for {@link ImageService#fromName client.images.fromName()}. */
export type ImageFromNameParams = {
  /** Modal Environment to resolve the named Image in. */
  environment?: string;
};

/**
 * Service for managing {@link Image}s.
 *
 * Normally only ever accessed via the client as:
 * ```typescript
 * const modal = new ModalClient();
 * const image = await modal.images.fromRegistry("alpine");
 * ```
 */
export class ImageService {
  readonly #client: ModalClient;
  constructor(client: ModalClient) {
    this.#client = client;
  }

  /**
   * Creates an {@link Image} from an Image ID
   *
   * @param imageId - Image ID.
   */
  async fromId(imageId: string): Promise<Image> {
    try {
      const resp = await this.#client.cpClient.imageFromId({ imageId });
      return new Image(this.#client, resp.imageId, "");
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(err.details);
      if (
        err instanceof ClientError &&
        err.code === Status.FAILED_PRECONDITION &&
        err.details.includes("Could not find image with ID")
      )
        throw new NotFoundError(err.details);
      throw err;
    }
  }

  /**
   * Reference a named {@link Image} that was previously published.
   *
   * @param name - Name of the published Image, optionally including a tag as `name:tag`.
   * If no tag is included, `:latest` is used.
   * @param params - Optional environment.
   */
  async fromName(
    name: string,
    params: ImageFromNameParams = {},
  ): Promise<Image> {
    const [namespacePrefix, tag] = parseNamedImageRef(name);

    let environmentName: string;
    if (namespacePrefix) {
      if (params.environment) {
        throw new InvalidError(
          "Cannot specify 'environment' when the image name contains a '/'.",
        );
      }
      environmentName = "";
    } else {
      environmentName = this.#client.environmentName(params.environment);
    }

    try {
      const resp = await this.#client.cpClient.imageGetByTag({
        environmentName,
        tag,
      });
      return new Image(this.#client, resp.imageId, "");
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(err.details);
      throw err;
    }
  }

  /**
   * Creates an {@link Image} from a raw registry tag, optionally using a {@link Secret} for authentication.
   *
   * @param tag - The registry tag for the Image.
   * @param secret - Optional. A Secret containing credentials for registry authentication.
   */
  fromRegistry(tag: string, secret?: Secret): Image {
    let imageRegistryConfig;
    if (secret) {
      if (!(secret instanceof Secret)) {
        throw new TypeError(
          "secret must be a reference to an existing Secret, e.g. `await client.secrets.fromName('my_secret')`",
        );
      }
      imageRegistryConfig = {
        registryAuthType: RegistryAuthType.REGISTRY_AUTH_TYPE_STATIC_CREDS,
        secretId: secret.secretId,
      };
    }
    return new Image(this.#client, "", tag, imageRegistryConfig);
  }

  /**
   * Creates an {@link Image} from a raw registry tag, optionally using a {@link Secret} for authentication.
   *
   * @param tag - The registry tag for the Image.
   * @param secret - A Secret containing credentials for registry authentication.
   */
  fromAwsEcr(tag: string, secret: Secret): Image {
    let imageRegistryConfig;
    if (secret) {
      if (!(secret instanceof Secret)) {
        throw new TypeError(
          "secret must be a reference to an existing Secret, e.g. `await client.secrets.fromName('my_secret')`",
        );
      }
      imageRegistryConfig = {
        registryAuthType: RegistryAuthType.REGISTRY_AUTH_TYPE_AWS,
        secretId: secret.secretId,
      };
    }
    return new Image(this.#client, "", tag, imageRegistryConfig);
  }

  /**
   * Creates an {@link Image} from a raw registry tag, optionally using a {@link Secret} for authentication.
   *
   * @param tag - The registry tag for the Image.
   * @param secret - A Secret containing credentials for registry authentication.
   */
  fromGcpArtifactRegistry(tag: string, secret: Secret): Image {
    let imageRegistryConfig;
    if (secret) {
      if (!(secret instanceof Secret)) {
        throw new TypeError(
          "secret must be a reference to an existing Secret, e.g. `await client.secrets.fromName('my_secret')`",
        );
      }
      imageRegistryConfig = {
        registryAuthType: RegistryAuthType.REGISTRY_AUTH_TYPE_GCP,
        secretId: secret.secretId,
      };
    }
    return new Image(this.#client, "", tag, imageRegistryConfig);
  }

  /**
   * Delete an {@link Image} by ID.
   *
   * Deletion is irreversible and will prevent Functions/Sandboxes from using the Image.
   *
   * Note: When building an Image, each chained method call will create an
   * intermediate Image layer, each with its own ID. Deleting an Image will not
   * delete any of its intermediate layers, only the image identified by the
   * provided ID.
   */
  async delete(imageId: string, _: ImageDeleteParams = {}): Promise<void> {
    try {
      await this.#client.cpClient.imageDelete({ imageId });
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(err.details);
      if (
        err instanceof ClientError &&
        err.code === Status.FAILED_PRECONDITION &&
        err.details.includes("Could not find image with ID")
      )
        throw new NotFoundError(err.details);
      throw err;
    }
  }
}

/** Optional parameters for {@link ImageService#delete client.images.delete()}. */
export type ImageDeleteParams = Record<never, never>;

/** Optional parameters for {@link Image#publish Image.publish()}. */
export type ImagePublishParams = {
  /** Modal Environment to publish the named Image in. */
  environment?: string;
};

/** Optional parameters for {@link Image#dockerfileCommands Image.dockerfileCommands()}. */
export type ImageDockerfileCommandsParams = {
  /** Environment variables to set in the build environment. */
  env?: Record<string, string>;

  /** {@link Secret}s that will be made available as environment variables to this layer's build environment. */
  secrets?: Secret[];

  /** GPU reservation for this layer's build environment (e.g. "A100", "T4:2", "A100-80GB:4"). */
  gpu?: string;

  /** Ignore cached builds for this layer, similar to 'docker build --no-cache'. */
  forceBuild?: boolean;
};

/** Represents a single image layer with its build configuration. */
type Layer = {
  commands: string[];
  env?: Record<string, string>;
  secrets?: Secret[];
  gpuConfig?: GPUConfig;
  forceBuild?: boolean;
};

/** A container image, used for starting {@link Sandbox}es. */
export class Image {
  #client: ModalClient;
  #imageId: string;
  #tag: string;
  // Image ID of the parent image layer to use as `FROM base`.
  #baseImageId: string;
  #imageRegistryConfig?: ImageRegistryConfig;
  #layers: Layer[];

  /** @ignore */
  constructor(
    client: ModalClient,
    imageId: string,
    tag: string,
    imageRegistryConfig?: ImageRegistryConfig,
    layers?: Layer[],
    baseImageId?: string,
  ) {
    this.#client = client;
    this.#imageId = imageId;
    this.#tag = tag;
    this.#baseImageId = baseImageId || "";
    this.#imageRegistryConfig = imageRegistryConfig;
    this.#layers = layers || [
      {
        commands: [],
        env: undefined,
        secrets: undefined,
        gpuConfig: undefined,
        forceBuild: false,
      },
    ];
  }
  get imageId(): string {
    return this.#imageId;
  }

  private static validateDockerfileCommands(commands: string[]): void {
    for (const command of commands) {
      const trimmed = command.trim().toUpperCase();
      if (trimmed.startsWith("COPY ") && !trimmed.startsWith("COPY --FROM=")) {
        throw new InvalidError(
          "COPY commands that copy from local context are not yet supported.",
        );
      }
    }
  }

  /**
   * Extend an image with arbitrary Dockerfile-like commands.
   *
   * Each call creates a new Image layer that will be built sequentially.
   * The provided options apply only to this layer.
   *
   * @param commands - Array of Dockerfile commands as strings
   * @param params - Optional configuration for this layer's build
   * @returns A new Image instance
   */
  dockerfileCommands(
    commands: string[],
    params?: ImageDockerfileCommandsParams,
  ): Image {
    if (commands.length === 0) {
      return this;
    }

    Image.validateDockerfileCommands(commands);

    const newLayer: Layer = {
      commands: [...commands],
      env: params?.env,
      secrets: params?.secrets,
      gpuConfig: params?.gpu ? parseGpuConfig(params.gpu) : undefined,
      forceBuild: params?.forceBuild,
    };

    const baseImageId = this.#imageId || this.#baseImageId;
    const layers = this.#imageId === "" ? this.#layers : [];

    return new Image(
      this.#client,
      "",
      this.#tag,
      this.#imageRegistryConfig,
      [...layers, newLayer],
      baseImageId,
    );
  }

  /**
   * Eagerly builds an Image on Modal.
   *
   * @param app - App to use to build the Image.
   */
  async build(app: App): Promise<Image> {
    if (this.imageId !== "") {
      // Image is already built with an Image ID
      return this;
    }

    this.#client.logger.debug("Building image", "app_id", app.appId);

    let baseImageId: string | undefined = this.#baseImageId || undefined;
    const imageBuilderVersion = await this.#client.getImageBuilderVersion(
      app.environmentName,
    );

    for (let i = 0; i < this.#layers.length; i++) {
      const layer = this.#layers[i];

      const mergedSecrets = await mergeEnvIntoSecrets(
        this.#client,
        layer.env,
        layer.secrets,
      );

      const secretIds = mergedSecrets.map((secret) => secret.secretId);
      const gpuConfig = layer.gpuConfig;

      let dockerfileCommands: string[];
      let baseImages: Array<{ dockerTag: string; imageId: string }>;

      if (i === 0 && baseImageId) {
        dockerfileCommands = ["FROM base", ...layer.commands];
        baseImages = [{ dockerTag: "base", imageId: baseImageId }];
      } else if (i === 0) {
        dockerfileCommands = [`FROM ${this.#tag}`, ...layer.commands];
        baseImages = [];
      } else {
        dockerfileCommands = ["FROM base", ...layer.commands];
        baseImages = [{ dockerTag: "base", imageId: baseImageId! }];
      }

      const imageRegistryConfig =
        i === 0 && !baseImageId ? this.#imageRegistryConfig : undefined;

      const resp = await this.#client.cpClient.imageGetOrCreate({
        appId: app.appId,
        image: ImageProto.create({
          dockerfileCommands,
          imageRegistryConfig,
          secretIds,
          gpuConfig,
          contextFiles: [],
          baseImages,
        }),
        builderVersion: imageBuilderVersion,
        forceBuild: layer.forceBuild || false,
      });

      let result: GenericResult;

      if (resp.result?.status) {
        // Image has already been built
        result = resp.result;
      } else {
        // Not built or in the process of building - wait for build
        let lastEntryId = "";
        let resultJoined: GenericResult | undefined = undefined;
        while (!resultJoined) {
          for await (const item of this.#client.cpClient.imageJoinStreaming({
            imageId: resp.imageId,
            timeout: 55,
            lastEntryId,
          })) {
            if (item.entryId) lastEntryId = item.entryId;
            if (item.result?.status) {
              resultJoined = item.result;
              break;
            }
            // Ignore all log lines and progress updates.
          }
        }
        result = resultJoined;
      }

      if (
        result.status === GenericResult_GenericStatus.GENERIC_STATUS_FAILURE
      ) {
        throw new Error(
          `Image build for ${resp.imageId} failed with the exception:\n${result.exception}`,
        );
      } else if (
        result.status === GenericResult_GenericStatus.GENERIC_STATUS_TERMINATED
      ) {
        throw new Error(
          `Image build for ${resp.imageId} terminated due to external shut-down. Please try again.`,
        );
      } else if (
        result.status === GenericResult_GenericStatus.GENERIC_STATUS_TIMEOUT
      ) {
        throw new Error(
          `Image build for ${resp.imageId} timed out. Please try again with a larger timeout parameter.`,
        );
      } else if (
        result.status !== GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS
      ) {
        throw new Error(
          `Image build for ${resp.imageId} failed with unknown status: ${result.status}`,
        );
      }

      // the new image is the base for the next layer
      baseImageId = resp.imageId;
    }
    this.#imageId = baseImageId!;
    this.#client.logger.debug("Image build completed", "image_id", baseImageId);
    return this;
  }

  /**
   * Publish this built Image under a stable name and tag.
   *
   * @param name - Name to publish the Image under, optionally including a tag as `name:tag`.
   * If no tag is included, `:latest` is used.
   * @param params - Optional environment.
   */
  async publish(name: string, params: ImagePublishParams = {}): Promise<void> {
    const [namespacePrefix, tag] = parseNamedImageRef(name);

    if (this.#imageId === "") {
      throw new InvalidError(
        "Cannot publish an image that has not been built yet. Call build() first.",
      );
    }

    let environmentName: string;
    if (namespacePrefix) {
      if (params.environment) {
        throw new InvalidError(
          "Cannot specify 'environment' when the image name contains a '/'.",
        );
      }
      environmentName = "";
    } else {
      environmentName = this.#client.environmentName(params.environment);
    }

    await this.#client.cpClient.imagePublish({
      imageId: this.#imageId,
      environmentName,
      allowPublic: false,
      tag,
    });
  }
}
