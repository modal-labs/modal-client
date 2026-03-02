import {
  GenericResult,
  GenericResult_GenericStatus,
  RegistryAuthType,
  ImageRegistryConfig,
  Image as ImageProto,
  GPUConfig,
} from "../proto/modal_proto/api";
import { getDefaultClient, type ModalClient } from "./client";
import { App, parseGpuConfig } from "./app";
import { Secret, mergeEnvIntoSecrets } from "./secret";
import { ClientError } from "nice-grpc";
import { Status } from "nice-grpc";
import { NotFoundError, InvalidError } from "./errors";

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
          "secret must be a reference to an existing Secret, e.g. `await Secret.fromName('my_secret')`",
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
          "secret must be a reference to an existing Secret, e.g. `await Secret.fromName('my_secret')`",
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
          "secret must be a reference to an existing Secret, e.g. `await Secret.fromName('my_secret')`",
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
  #imageRegistryConfig?: ImageRegistryConfig;
  #layers: Layer[];

  /** @ignore */
  constructor(
    client: ModalClient,
    imageId: string,
    tag: string,
    imageRegistryConfig?: ImageRegistryConfig,
    layers?: Layer[],
  ) {
    this.#client = client;
    this.#imageId = imageId;
    this.#tag = tag;
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

  /**
   * @deprecated Use {@link ImageService#fromId client.images.fromId()} instead.
   */
  static async fromId(imageId: string): Promise<Image> {
    return getDefaultClient().images.fromId(imageId);
  }

  /**
   * @deprecated Use {@link ImageService#fromRegistry client.images.fromRegistry()} instead.
   */
  static fromRegistry(tag: string, secret?: Secret): Image {
    return getDefaultClient().images.fromRegistry(tag, secret);
  }

  /**
   * @deprecated Use {@link ImageService#fromAwsEcr client.images.fromAwsEcr()} instead.
   */
  static fromAwsEcr(tag: string, secret: Secret): Image {
    return getDefaultClient().images.fromAwsEcr(tag, secret);
  }

  /**
   * @deprecated Use {@link ImageService#fromGcpArtifactRegistry client.images.fromGcpArtifactRegistry()} instead.
   */
  static fromGcpArtifactRegistry(tag: string, secret: Secret): Image {
    return getDefaultClient().images.fromGcpArtifactRegistry(tag, secret);
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

    return new Image(this.#client, "", this.#tag, this.#imageRegistryConfig, [
      ...this.#layers,
      newLayer,
    ]);
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

    let baseImageId: string | undefined;

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

      if (i === 0) {
        dockerfileCommands = [`FROM ${this.#tag}`, ...layer.commands];
        baseImages = [];
      } else {
        dockerfileCommands = ["FROM base", ...layer.commands];
        baseImages = [{ dockerTag: "base", imageId: baseImageId! }];
      }

      const resp = await this.#client.cpClient.imageGetOrCreate({
        appId: app.appId,
        image: ImageProto.create({
          dockerfileCommands,
          imageRegistryConfig: this.#imageRegistryConfig,
          secretIds,
          gpuConfig,
          contextFiles: [],
          baseImages,
        }),
        builderVersion: this.#client.imageBuilderVersion(),
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
   * @deprecated Use {@link ImageService#delete client.images.delete()} instead.
   */
  static async delete(
    imageId: string,
    _: ImageDeleteParams = {},
  ): Promise<void> {
    return getDefaultClient().images.delete(imageId);
  }
}
