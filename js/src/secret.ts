import { type ModalClient } from "./client";
import { ClientError, Status } from "nice-grpc";
import { InvalidError, NotFoundError } from "./errors";
import { ObjectCreationType } from "../proto/modal_proto/api";

// Environment variable names must consist of letters, numbers, and
// underscores, and may not start with a number. Mirrors the server-side
// validation in modal_server (SECRET_KEYNAME_REGEX).
const ENV_VAR_KEY_REGEX = /^[a-zA-Z_][a-zA-Z0-9_]*$/;

/**
 * Validate that every key in an env dict is a valid environment variable name
 * (letters, numbers, and underscores, not starting with a number). Throws an
 * {@link InvalidError} on the first invalid key. Mirrors the server-side
 * validation, and is applied client-side wherever env vars are sent directly to
 * a worker without a `SecretGetOrCreate` round-trip to validate them.
 *
 * @internal
 * @hidden
 */
export function validateEnvVarKeys(env: Record<string, string>): void {
  for (const key of Object.keys(env)) {
    if (!ENV_VAR_KEY_REGEX.test(key)) {
      throw new InvalidError(
        `Secret key name ${JSON.stringify(key)} is invalid for environment ` +
          "variables. Only letters, numbers, and underscores are allowed, and " +
          "the name may not start with a number.",
      );
    }
  }
}

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

  /**
   * Create a {@link Secret} from a plain object of key-value pairs.
   *
   * The returned Secret is lazy: no server-side Secret is created (and
   * {@link Secret#secretId secretId} stays empty) until it is first used. When
   * used with {@link Sandbox#exec Sandbox.exec()} or
   * {@link SandboxService#experimentalCreate client.sandboxes.experimentalCreate()},
   * the values are sent directly to the worker as environment variables,
   * avoiding a `SecretGetOrCreate` round-trip entirely.
   */
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
    validateEnvVarKeys(entries);

    // Copy the entries so later mutations of the caller's object don't leak in.
    const envDict = { ...entries };
    const environment = this.#client.environmentName(params?.environment);
    return new Secret(
      "",
      undefined,
      new SecretFromObjectHydrator(envDict, environment),
    );
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

/**
 * Resolves a lazy {@link Secret} to a server-side secretId. Each construction
 * path (fromName, fromObject, ...) supplies its own implementation carrying
 * whatever inputs that path needs.
 *
 * @internal
 * @hidden
 */
export interface SecretHydrator {
  hydrate(client: ModalClient): Promise<string>;
}

/**
 * Creates an ephemeral server-side Secret from a locally provided env map. Used
 * by Secrets created via {@link SecretService#fromObject}.
 *
 * @internal
 * @hidden
 */
export class SecretFromObjectHydrator implements SecretHydrator {
  readonly envDict: Record<string, string>;
  readonly #environment?: string;

  constructor(envDict: Record<string, string>, environment?: string) {
    this.envDict = envDict;
    this.#environment = environment;
  }

  async hydrate(client: ModalClient): Promise<string> {
    try {
      const resp = await client.cpClient.secretGetOrCreate({
        objectCreationType: ObjectCreationType.OBJECT_CREATION_TYPE_EPHEMERAL,
        envDict: this.envDict,
        environmentName: this.#environment,
      });
      client.logger.debug(
        "Created ephemeral Secret",
        "secret_id",
        resp.secretId,
      );
      return resp.secretId;
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
}

/** Secrets provide a dictionary of environment variables for {@link Image}s. */
export class Secret {
  #secretId: string;
  readonly name?: string;

  // Resolves secretId lazily the first time it is needed. Undefined for Secrets
  // constructed already-hydrated (e.g. from fromName), and not consulted again
  // once secretId is set.
  readonly #hydrator?: SecretHydrator;

  // Caches the single in-flight (or successfully completed) hydration so the
  // ephemeral Secret is created at most once, even if multiple callers hydrate
  // the same Secret concurrently. Cleared on failure so a transient error
  // doesn't permanently break the Secret; subsequent calls can retry.
  #hydratePromise?: Promise<void>;

  /** @ignore */
  constructor(secretId: string, name?: string, hydrator?: SecretHydrator) {
    this.#secretId = secretId;
    this.name = name;
    this.#hydrator = hydrator;
  }

  /** The ID of the server-side Secret, or an empty string if not yet hydrated. */
  get secretId(): string {
    return this.#secretId;
  }

  /**
   * The hydrator resolving a lazy Secret to a secretId, or `undefined` for
   * Secrets that already reference a server-side Secret (e.g. from fromName).
   *
   * @internal
   * @hidden
   */
  get _hydrator(): SecretHydrator | undefined {
    return this.#hydrator;
  }

  /**
   * Lazily create the ephemeral server-side Secret backing a fromObject Secret
   * and populate {@link Secret#secretId secretId}. A no-op once secretId is set
   * (e.g. for Secrets returned by fromName).
   *
   * @internal
   * @hidden
   */
  async _hydrate(client: ModalClient): Promise<void> {
    if (this.#secretId !== "" || this.#hydrator === undefined) {
      return;
    }
    if (this.#hydratePromise === undefined) {
      this.#hydratePromise = this.#doHydrate(client);
    }
    const promise = this.#hydratePromise;
    try {
      await promise;
    } catch (err) {
      // Clear the cached Promise so a transient failure doesn't permanently
      // break this Secret; subsequent calls can retry hydration.
      if (this.#hydratePromise === promise) {
        this.#hydratePromise = undefined;
      }
      throw err;
    }
  }

  async #doHydrate(client: ModalClient): Promise<void> {
    // #hydrator is guaranteed defined by the guard in _hydrate.
    this.#secretId = await this.#hydrator!.hydrate(client);
  }
}

/**
 * Reports whether `secret` is a non-null env-dict Secret and, if so, returns
 * its hydrator. This is the worker fast path: such Secrets can be passed to the
 * worker as environment variables without a SecretGetOrCreate round-trip.
 *
 * @internal
 * @hidden
 */
export function secretEnvDictHydrator(
  secret: Secret,
): SecretFromObjectHydrator | undefined {
  const hydrator = secret?._hydrator;
  return hydrator instanceof SecretFromObjectHydrator ? hydrator : undefined;
}

/**
 * Merge environment variables into a list of Secrets. If `env` is non-empty, a
 * lazy Secret is created from it (via {@link SecretService#fromObject}) and
 * appended. The appended Secret is hydrated together with the others when its
 * secretId is needed.
 *
 * @internal
 * @hidden
 */
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

/**
 * Hydrate each Secret in the list so its {@link Secret#secretId secretId} is
 * available, running the hydrations concurrently. Rejects if any Secret is
 * null/undefined.
 *
 * @internal
 * @hidden
 */
export async function hydrateSecrets(
  client: ModalClient,
  secrets: Secret[],
): Promise<void> {
  secrets.forEach((secret, i) => {
    if (secret == null) {
      throw new InvalidError(`secret at index ${i} must not be null`);
    }
  });
  await Promise.all(secrets.map((secret) => secret._hydrate(client)));
}

/**
 * Collect the secretIds from a list of Secrets, rejecting any that are
 * null/undefined or have not yet been hydrated.
 *
 * @internal
 * @hidden
 */
export function collectSecretIds(secrets: Secret[]): string[] {
  return secrets.map((secret, i) => {
    if (secret == null) {
      throw new InvalidError(`secret at index ${i} must not be null`);
    }
    if (secret.secretId === "") {
      throw new InvalidError(`secret at index ${i} has not been hydrated`);
    }
    return secret.secretId;
  });
}

/**
 * Partition secrets into a merged env dict (from Secrets created locally via
 * {@link SecretService#fromObject}) and the remaining "resolvable" Secrets that
 * must be hydrated to a secretId before use (e.g. from
 * {@link SecretService#fromName}).
 *
 * Locally-created Secrets can be passed directly to the worker as environment
 * variables, avoiding a SecretGetOrCreate round-trip. Local Secrets are merged
 * in list order (so later ones win on key collisions). This function does not
 * validate its input: null/undefined Secrets are placed in the resolvable list
 * rather than dropped, leaving it to {@link hydrateSecrets} to reject them.
 *
 * @internal
 * @hidden
 */
export function splitEnvDictAndResolvableSecrets(
  secrets: Secret[],
): [Record<string, string>, Secret[]] {
  const envDict: Record<string, string> = {};
  const resolvable: Secret[] = [];
  for (const secret of secrets) {
    const hydrator = secret != null ? secretEnvDictHydrator(secret) : undefined;
    if (hydrator !== undefined) {
      Object.assign(envDict, hydrator.envDict);
    } else {
      resolvable.push(secret);
    }
  }
  return [envDict, resolvable];
}
