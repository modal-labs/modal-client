import { type Logger } from "./logger";
import { type ModalClient, type ModalGrpcClient } from "./client";
import { ClientError, Status } from "nice-grpc";
import {
  AppListRequest,
  AppState,
  EnvironmentCreateRequest,
  EnvironmentDeleteRequest,
  EnvironmentGetOrCreateRequest,
  ObjectCreationType,
} from "../proto/modal_proto/api";
import { App } from "./app";
import { AlreadyExistsError, InvalidError, NotFoundError } from "./errors";
import { Empty } from "../proto/google/protobuf/empty";

function checkEnvironmentName(name: string, allowPeriods: boolean): void {
  const invalidCharacters = allowPeriods ? /[^a-zA-Z0-9_.-]/ : /[^a-zA-Z0-9_-]/;
  if (
    name.length > 64 ||
    !/^[a-zA-Z0-9]/.test(name) ||
    invalidCharacters.test(name) ||
    name.toLowerCase().startsWith("en-")
  ) {
    const allowedCharacters = allowPeriods
      ? "alphanumeric characters, dashes, periods, and underscores"
      : "alphanumeric characters, dashes, and underscores";
    throw new InvalidError(
      `Invalid Environment name: '${name}'. Environment names must be 1-64 characters, start with an alphanumeric character, contain only ${allowedCharacters}, and cannot start with 'en-'.`,
    );
  }
}

/** Service for looking up {@link Environment}s through `client.environments`. */
export class EnvironmentService {
  readonly #client: ModalClient;
  /** Namespace for creating, listing, and deleting Environments. */
  readonly objects: EnvironmentObjectsService;

  /** @ignore */
  constructor(client: ModalClient) {
    this.#client = client;
    this.objects = new EnvironmentObjectsService(client);
  }

  /**
   * Look up an Environment by name, optionally creating it if it does not exist.
   * @throws {InvalidError} The Environment name is invalid.
   */
  async fromName(
    name: string,
    params: EnvironmentFromNameParams = {},
  ): Promise<Environment> {
    checkEnvironmentName(name, true);
    try {
      const resp = await this.#client.cpClient.environmentGetOrCreate(
        EnvironmentGetOrCreateRequest.create({
          deploymentName: name,
          objectCreationType: params.createIfMissing
            ? ObjectCreationType.OBJECT_CREATION_TYPE_CREATE_IF_MISSING
            : ObjectCreationType.OBJECT_CREATION_TYPE_UNSPECIFIED,
        }),
      );
      return new Environment(
        this.#client,
        resp.environmentId,
        resp.metadata?.name ?? name,
      );
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND) {
        throw new NotFoundError(`Environment '${name}' not found`);
      }
      throw err;
    }
  }
}

/** Optional parameters for {@link EnvironmentService#fromName client.environments.fromName()}. */
export type EnvironmentFromNameParams = {
  createIfMissing?: boolean;
};

/** Namespace for creating, listing, and deleting Environments. */
export class EnvironmentObjectsService {
  readonly #client: ModalClient;

  /** @ignore */
  constructor(client: ModalClient) {
    this.#client = client;
  }

  /** Create an Environment with the given name. */
  async create(
    name: string,
    params: EnvironmentCreateParams = {},
  ): Promise<void> {
    checkEnvironmentName(name, false);
    try {
      await this.#client.cpClient.environmentCreate(
        EnvironmentCreateRequest.create({
          name,
          isManaged: params.restricted ?? false,
          defaultMemberRoleStr: params.defaultRole ?? "",
        }),
      );
    } catch (err) {
      if (err instanceof ClientError) {
        if (err.code === Status.ALREADY_EXISTS) {
          throw new AlreadyExistsError(`Environment '${name}' already exists`);
        }
        if (err.code === Status.INVALID_ARGUMENT) {
          throw new InvalidError(err.details);
        }
      }
      throw err;
    }
  }

  /**
   * Yield Environment handles without additional lookup requests.
   *
   * ```typescript
   * for await (const environment of client.environments.objects.list()) {
   *   console.log(environment.environmentId, environment.name);
   * }
   * ```
   */
  async *list(): AsyncGenerator<Environment, void, unknown> {
    const resp = await this.#client.cpClient.environmentList(Empty.create());
    for (const item of resp.items) {
      yield new Environment(this.#client, item.environmentId, item.name);
    }
  }

  /**
   * Delete an Environment by name.
   * This permanently deletes the Environment and all objects in it.
   */
  async delete(name: string): Promise<void> {
    try {
      await this.#client.cpClient.environmentDelete(
        EnvironmentDeleteRequest.create({ name }),
      );
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND) {
        throw new NotFoundError(`Environment '${name}' not found`);
      }
      throw err;
    }
  }
}

/** Optional parameters for {@link EnvironmentObjectsService#create}. */
export type EnvironmentCreateParams = {
  /** Restrict access to users with explicitly assigned Environment roles. */
  restricted?: boolean;
  /** Default Environment role for workspace members. */
  defaultRole?: string;
};

/** A Modal Environment, with namespaces for accessing its resources. */
export class Environment {
  readonly environmentId: string;
  readonly name: string;
  readonly apps: EnvironmentAppsService;

  /** @ignore */
  constructor(client: ModalClient, environmentId: string, name: string) {
    this.environmentId = environmentId;
    this.name = name;
    this.apps = new EnvironmentAppsService(client, name);
  }
}

/** Namespace for accessing Apps in an {@link Environment}. */
export class EnvironmentAppsService {
  readonly #client: ModalClient;
  readonly #environmentName: string;

  /** @ignore */
  constructor(client: ModalClient, environmentName: string) {
    this.#client = client;
    this.#environmentName = environmentName;
  }

  /**
   * Yield handles for live Apps in this Environment, excluding stopped and disabled Apps.
   * Listing Apps does not create Apps.
   *
   * ```typescript
   * const environment = await client.environments.fromName("prod");
   * for await (const app of environment.apps.list()) {
   *   console.log(app.appId, app.name);
   * }
   * ```
   */
  async *list(): AsyncGenerator<App, void, unknown> {
    const resp = await this.#client.cpClient.appList(
      AppListRequest.create({ environmentName: this.#environmentName }),
    );
    for (const info of resp.apps) {
      if (
        info.state === AppState.APP_STATE_STOPPED ||
        info.state === AppState.APP_STATE_DISABLED
      ) {
        continue;
      }
      yield new App(
        info.appId,
        info.name || undefined,
        this.#environmentName,
        this.#client,
        info.metadata,
      );
    }
  }
}

/** Environment-scoped configuration from the server. */
interface EnvironmentSettings {
  imageBuilderVersion: string;
  webhookSuffix: string;
}

/** Modal Environment with its server-provided settings. */
interface EnvironmentData {
  id: string;
  name: string;
  settings: EnvironmentSettings;
}

export class EnvironmentManager {
  #cache: Map<string, Promise<EnvironmentData>>;
  #logger: Logger;
  #client: ModalGrpcClient;

  constructor(client: ModalGrpcClient, logger: Logger) {
    this.#cache = new Map();
    this.#client = client;
    this.#logger = logger;
  }

  private async fetchEnvironment(name?: string): Promise<EnvironmentData> {
    const key = name ?? "";
    const cached = this.#cache.get(key);
    if (cached) {
      return cached;
    }

    // Cache the promise so concurrent callers share a single fetch. On failure,
    // evict it so the next call retries instead of replaying the rejection.
    const promise = this.doFetchEnvironment(key);
    this.#cache.set(key, promise);
    promise.catch(() => this.#cache.delete(key));
    return promise;
  }

  private async doFetchEnvironment(name: string): Promise<EnvironmentData> {
    this.#logger.debug(
      "Fetching environment from server",
      "environment_name",
      name,
    );
    const resp = await this.#client.environmentGetOrCreate({
      deploymentName: name,
    });

    const env: EnvironmentData = {
      id: resp.environmentId,
      name: resp.metadata?.name ?? "",
      settings: {
        // The server should **always** return a non-empty image builder version.
        imageBuilderVersion: resp.metadata?.settings?.imageBuilderVersion || "",
        webhookSuffix: resp.metadata?.settings?.webhookSuffix || "",
      },
    };
    this.#logger.debug(
      "Cached environment",
      "environment_name",
      name,
      "environment_id",
      env.id,
      "image_builder_version",
      env.settings.imageBuilderVersion,
    );
    return env;
  }

  /**
   * Returns the image builder version by querying the server
   */
  async getImageBuilderVersion(environmentName?: string): Promise<string> {
    const env = await this.fetchEnvironment(environmentName);
    return env.settings.imageBuilderVersion;
  }
}
