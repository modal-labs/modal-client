import { ClientError, Status } from "nice-grpc";
import {
  AppGetInfoRequest,
  type AppHandleMetadata,
  AppState as AppStateProto,
  ObjectCreationType,
} from "../proto/modal_proto/api";
import { type ModalClient } from "./client";
import { InvalidError, NotFoundError } from "./errors";
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
    const environmentName = this.#client.environmentName(params.environment);
    try {
      const resp = await this.#client.cpClient.appGetOrCreate({
        appName: name,
        environmentName,
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
      return new App(
        resp.appId,
        name,
        environmentName,
        this.#client,
        resp.handleMetadata,
      );
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

/** Lifecycle state of a Modal App. */
export type AppState =
  | "ephemeral"
  | "detached"
  | "deployed"
  | "stopping"
  | "stopped"
  | "initializing"
  | "disabled"
  | "unknown";

/** Timestamps and attributions for events in an App's lifecycle. */
export interface AppLifecycle {
  state: AppState;
  version?: number;
  createdAt: Date;
  createdBy: string;
  deployedAt?: Date;
  deployedBy?: string;
  stoppedAt?: Date;
  stoppedBy?: string;
}

/** Information about an App, its lifecycle, and its member Functions and Servers. */
export interface AppInfo {
  appId: string;
  description: string;
  lifecycle: AppLifecycle;
  /** Function names mapped to Function IDs. */
  functions: Record<string, string>;
  /** Server names mapped to Function IDs. */
  servers: Record<string, string>;
}

/** Optional parameters for {@link App#info}. */
export type AppInfoParams = {
  /** Fetch current information rather than the cached lookup or listing snapshot. */
  refresh?: boolean;
};

function appInfoFromProto(metadata: AppHandleMetadata, appId: string): AppInfo {
  const states: Partial<Record<AppStateProto, AppState>> = {
    [AppStateProto.APP_STATE_UNSPECIFIED]: "ephemeral",
    [AppStateProto.APP_STATE_EPHEMERAL]: "ephemeral",
    [AppStateProto.APP_STATE_DETACHED]: "detached",
    [AppStateProto.APP_STATE_DETACHED_DISCONNECTED]: "detached",
    [AppStateProto.APP_STATE_DEPLOYED]: "deployed",
    [AppStateProto.APP_STATE_STOPPING]: "stopping",
    [AppStateProto.APP_STATE_STOPPED]: "stopped",
    [AppStateProto.APP_STATE_INITIALIZING]: "initializing",
    [AppStateProto.APP_STATE_DISABLED]: "disabled",
  };
  const lifecycle = metadata.lifecycle;
  return {
    appId,
    description: metadata.description,
    functions: { ...metadata.functions },
    servers: { ...metadata.servers },
    lifecycle: {
      state:
        states[lifecycle?.appState ?? AppStateProto.APP_STATE_UNSPECIFIED] ??
        "unknown",
      version: lifecycle?.version || undefined,
      createdAt: new Date((lifecycle?.createdAt ?? 0) * 1000),
      createdBy: lifecycle?.createdBy ?? "",
      deployedAt:
        lifecycle && lifecycle.deployedAt > 0
          ? new Date(lifecycle.deployedAt * 1000)
          : undefined,
      deployedBy: lifecycle?.deployedBy || undefined,
      stoppedAt:
        lifecycle && lifecycle.stoppedAt > 0
          ? new Date(lifecycle.stoppedAt * 1000)
          : undefined,
      stoppedBy: lifecycle?.stoppedBy || undefined,
    },
  };
}

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

/** Represents a Modal App. */
export class App {
  readonly appId: string;
  readonly name?: string;
  readonly environmentName?: string;
  readonly #client?: ModalClient;
  #info?: AppInfo;

  /** @ignore */
  constructor(
    appId: string,
    name?: string,
    environmentName?: string,
    client?: ModalClient,
    metadata?: AppHandleMetadata,
  ) {
    this.appId = appId;
    this.name = name;
    this.environmentName = environmentName;
    this.#client = client;
    this.#info = metadata ? appInfoFromProto(metadata, appId) : undefined;
  }

  /**
   * Return information about this App, including its lifecycle and member Functions and Servers.
   * By default, returns the snapshot from the previous lookup, listing, or refresh.
   */
  async info(params: AppInfoParams = {}): Promise<AppInfo> {
    if (!this.#info || params.refresh) {
      if (!this.#client) {
        throw new InvalidError(
          "App.info() requires a handle retrieved through a ModalClient.",
        );
      }
      try {
        const resp = await this.#client.cpClient.appGetInfo(
          AppGetInfoRequest.create({ appId: this.appId }),
        );
        if (!resp.info) {
          throw new InvalidError(
            `No information returned for App '${this.appId}'`,
          );
        }
        this.#info = appInfoFromProto(resp.info, this.appId);
      } catch (err) {
        if (err instanceof ClientError && err.code === Status.NOT_FOUND) {
          throw new NotFoundError(`App '${this.appId}' not found`);
        }
        throw err;
      }
    }
    return structuredClone(this.#info);
  }
}
