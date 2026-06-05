import { type Logger } from "./logger";
import { type ModalGrpcClient } from "./client";

/** Environment-scoped configuration from the server. */
interface EnvironmentSettings {
  imageBuilderVersion: string;
  webhookSuffix: string;
}

/** Modal Environment with its server-provided settings. */
interface Environment {
  id: string;
  name: string;
  settings: EnvironmentSettings;
}

export class EnvironmentManager {
  #cache: Map<string, Promise<Environment>>;
  #logger: Logger;
  #client: ModalGrpcClient;

  constructor(client: ModalGrpcClient, logger: Logger) {
    this.#cache = new Map();
    this.#client = client;
    this.#logger = logger;
  }

  private async fetchEnvironment(name?: string): Promise<Environment> {
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

  private async doFetchEnvironment(name: string): Promise<Environment> {
    this.#logger.debug(
      "Fetching environment from server",
      "environment_name",
      name,
    );
    const resp = await this.#client.environmentGetOrCreate({
      deploymentName: name,
    });

    const env: Environment = {
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
