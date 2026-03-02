import { ClientError, Status } from "nice-grpc";
import { ObjectCreationType } from "../proto/modal_proto/api";
import { getDefaultClient, type ModalClient } from "./client";
import { Image } from "./image";
import { Sandbox, SandboxCreateParams } from "./sandbox";
import { NotFoundError } from "./errors";
import { Secret } from "./secret";
import { GPUConfig } from "../proto/modal_proto/api";

/**
 * Service for managing {@link App}s.
 *
 * Normally only ever accessed via the client as:
 * ```typescript
 * const modal = new ModalClient();
 * const app = await modal.apps.fromName("my-app");
 * ```
 */
export class AppService {
  readonly #client: ModalClient;
  constructor(client: ModalClient) {
    this.#client = client;
  }

  /**
   * Reference a deployed {@link App} by name, or create if it does not exist.
   */
  async fromName(name: string, params: AppFromNameParams = {}): Promise<App> {
    try {
      const resp = await this.#client.cpClient.appGetOrCreate({
        appName: name,
        environmentName: this.#client.environmentName(params.environment),
        objectCreationType: params.createIfMissing
          ? ObjectCreationType.OBJECT_CREATION_TYPE_CREATE_IF_MISSING
          : ObjectCreationType.OBJECT_CREATION_TYPE_UNSPECIFIED,
      });
      this.#client.logger.debug(
        "Retrieved App",
        "app_id",
        resp.appId,
        "app_name",
        name,
      );
      return new App(resp.appId, name);
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(`App '${name}' not found`);
      throw err;
    }
  }
}

/** Optional parameters for {@link AppService#fromName client.apps.fromName()}. */
export type AppFromNameParams = {
  environment?: string;
  createIfMissing?: boolean;
};

/** @deprecated Use specific Params types instead. */
export type LookupOptions = {
  environment?: string;
  createIfMissing?: boolean;
};

/** @deprecated Use specific Params types instead. */
export type DeleteOptions = {
  environment?: string;
};

/** @deprecated Use specific Params types instead. */
export type EphemeralOptions = {
  environment?: string;
};

/**
 * Parse a GPU configuration string into a GPUConfig object.
 * @param gpu - GPU string in format "type" or "type:count" (e.g. "T4", "A100:2")
 * @returns GPUConfig object (empty config if no GPU specified)
 */
export function parseGpuConfig(gpu: string | undefined): GPUConfig {
  if (!gpu) {
    return GPUConfig.create({});
  }

  let gpuType = gpu;
  let count = 1;

  if (gpu.includes(":")) {
    const [type, countStr] = gpu.split(":", 2);
    gpuType = type;
    count = parseInt(countStr, 10);
    if (isNaN(count) || count < 1) {
      throw new Error(
        `Invalid GPU count: ${countStr}. Value must be a positive integer.`,
      );
    }
  }

  return GPUConfig.create({
    count,
    gpuType: gpuType.toUpperCase(),
  });
}

/** Represents a deployed Modal App. */
export class App {
  readonly appId: string;
  readonly name?: string;

  /** @ignore */
  constructor(appId: string, name?: string) {
    this.appId = appId;
    this.name = name;
  }

  /**
   * @deprecated Use {@link AppService#fromName client.apps.fromName()} instead.
   */
  // eslint-disable-next-line @typescript-eslint/no-deprecated
  static async lookup(name: string, options: LookupOptions = {}): Promise<App> {
    return getDefaultClient().apps.fromName(name, options);
  }

  /**
   * @deprecated Use {@link SandboxService#create client.sandboxes.create()} instead.
   */
  async createSandbox(
    image: Image,
    options: SandboxCreateParams = {},
  ): Promise<Sandbox> {
    return getDefaultClient().sandboxes.create(this, image, options);
  }

  /**
   * @deprecated Use {@link ImageService#fromRegistry client.images.fromRegistry()} instead.
   */
  async imageFromRegistry(tag: string, secret?: Secret): Promise<Image> {
    return getDefaultClient().images.fromRegistry(tag, secret).build(this);
  }

  /**
   * @deprecated Use {@link ImageService#fromAwsEcr client.images.fromAwsEcr()} instead.
   */
  async imageFromAwsEcr(tag: string, secret: Secret): Promise<Image> {
    return getDefaultClient().images.fromAwsEcr(tag, secret).build(this);
  }

  /**
   * @deprecated Use {@link ImageService#fromGcpArtifactRegistry client.images.fromGcpArtifactRegistry()} instead.
   */
  async imageFromGcpArtifactRegistry(
    tag: string,
    secret: Secret,
  ): Promise<Image> {
    return getDefaultClient()
      .images.fromGcpArtifactRegistry(tag, secret)
      .build(this);
  }
}
