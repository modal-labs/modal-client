import { getDefaultClient, type ModalClient } from "./client";
import { ClientError, Status } from "nice-grpc";
import { InvalidError, NotFoundError } from "./errors";
import { ObjectCreationType } from "../proto/modal_proto/api";

/** Optional parameters for {@link SecretService#fromName client.secrets.fromName()}. */
export type SecretFromNameParams = {
  environment?: string;
  requiredKeys?: string[];
};

/** Optional parameters for {@link SecretService#fromObject client.secrets.fromObject()}. */
export type SecretFromObjectParams = {
  environment?: string;
};

/** Optional parameters for {@link SecretService#delete client.secrets.delete()}. */
export type SecretDeleteParams = {
  environment?: string;
  allowMissing?: boolean;
};

/**
 * Service for managing {@link Secret Secrets}.
 *
 * Normally only ever accessed via the client as:
 * ```typescript
 * const modal = new ModalClient();
 * const secret = await modal.secrets.fromName("my-secret");
 * ```
 */
export class SecretService {
  readonly #client: ModalClient;
  constructor(client: ModalClient) {
    this.#client = client;
  }

  /** Reference a {@link Secret} by its name. */
  async fromName(name: string, params?: SecretFromNameParams): Promise<Secret> {
    try {
      const resp = await this.#client.cpClient.secretGetOrCreate({
        deploymentName: name,
        environmentName: this.#client.environmentName(params?.environment),
        requiredKeys: params?.requiredKeys ?? [],
      });
      this.#client.logger.debug(
        "Retrieved Secret",
        "secret_id",
        resp.secretId,
        "secret_name",
        name,
      );
      return new Secret(resp.secretId, name);
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(err.details);
      if (
        err instanceof ClientError &&
        err.code === Status.FAILED_PRECONDITION &&
        err.details.includes("Secret is missing key")
      )
        throw new NotFoundError(err.details);
      throw err;
    }
  }

  /** Create a {@link Secret} from a plain object of key-value pairs. */
  async fromObject(
    entries: Record<string, string>,
    params?: SecretFromObjectParams,
  ): Promise<Secret> {
    for (const [, value] of Object.entries(entries)) {
      if (value == null || typeof value !== "string") {
        throw new InvalidError(
          "entries must be an object mapping string keys to string values, but got:\n" +
            JSON.stringify(entries),
        );
      }
    }

    try {
      const resp = await this.#client.cpClient.secretGetOrCreate({
        objectCreationType: ObjectCreationType.OBJECT_CREATION_TYPE_EPHEMERAL,
        envDict: entries as Record<string, string>,
        environmentName: this.#client.environmentName(params?.environment),
      });
      this.#client.logger.debug(
        "Created ephemeral Secret",
        "secret_id",
        resp.secretId,
      );
      return new Secret(resp.secretId);
    } catch (err) {
      if (
        err instanceof ClientError &&
        (err.code === Status.INVALID_ARGUMENT ||
          err.code === Status.FAILED_PRECONDITION)
      )
        throw new InvalidError(err.details);
      throw err;
    }
  }

  /**
   * Delete a named {@link Secret}.
   *
   * Warning: Deletion is irreversible and will affect any Apps currently using the Secret.
   */
  async delete(name: string, params?: SecretDeleteParams): Promise<void> {
    try {
      const secret = await this.fromName(name, {
        environment: params?.environment,
      });
      await this.#client.cpClient.secretDelete({
        secretId: secret.secretId,
      });
      this.#client.logger.debug(
        "Deleted Secret",
        "secret_name",
        name,
        "secret_id",
        secret.secretId,
      );
    } catch (err) {
      const isNotFound =
        err instanceof NotFoundError ||
        (err instanceof ClientError && err.code === Status.NOT_FOUND);
      if (isNotFound && params?.allowMissing) {
        return;
      }
      throw err;
    }
  }
}

/** Secrets provide a dictionary of environment variables for {@link Image}s. */
export class Secret {
  readonly secretId: string;
  readonly name?: string;

  /** @ignore */
  constructor(secretId: string, name?: string) {
    this.secretId = secretId;
    this.name = name;
  }

  /**
   * @deprecated Use {@link SecretService#fromName client.secrets.fromName()} instead.
   */
  static async fromName(
    name: string,
    params?: SecretFromNameParams,
  ): Promise<Secret> {
    return getDefaultClient().secrets.fromName(name, params);
  }

  /**
   * @deprecated Use {@link SecretService#fromObject client.secrets.fromObject()} instead.
   */
  static async fromObject(
    entries: Record<string, string>,
    params?: SecretFromObjectParams,
  ): Promise<Secret> {
    return getDefaultClient().secrets.fromObject(entries, params);
  }
}

export async function mergeEnvIntoSecrets(
  client: ModalClient,
  env?: Record<string, string>,
  secrets?: Secret[],
): Promise<Secret[]> {
  const result = [...(secrets || [])];
  if (env && Object.keys(env).length > 0) {
    result.push(await client.secrets.fromObject(env));
  }
  return result;
}
