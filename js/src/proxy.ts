import { getDefaultClient, type ModalClient } from "./client";
import { ClientError, Status } from "nice-grpc";
import { NotFoundError } from "./errors";

/**
 * Service for managing {@link Proxy Proxies}.
 */
export class ProxyService {
  readonly #client: ModalClient;
  constructor(client: ModalClient) {
    this.#client = client;
  }

  /**
   * Reference a {@link Proxy} by its name.
   *
   * Normally only ever accessed via the client as:
   * ```typescript
   * const modal = new ModalClient();
   * const proxy = await modal.proxies.fromName("my-proxy");
   * ```
   */
  async fromName(name: string, params?: ProxyFromNameParams): Promise<Proxy> {
    try {
      const resp = await this.#client.cpClient.proxyGet({
        name,
        environmentName: this.#client.environmentName(params?.environment),
      });
      if (!resp.proxy?.proxyId) {
        throw new NotFoundError(`Proxy '${name}' not found`);
      }
      return new Proxy(resp.proxy.proxyId);
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(`Proxy '${name}' not found`);
      throw err;
    }
  }
}

/** Optional parameters for {@link ProxyService#fromName client.proxies.fromName()}. */
export type ProxyFromNameParams = {
  environment?: string;
};

/** Proxy objects give your Modal containers a static outbound IP address. */
export class Proxy {
  readonly proxyId: string;

  /** @ignore */
  constructor(proxyId: string) {
    this.proxyId = proxyId;
  }

  /**
   * @deprecated Use {@link ProxyService#fromName client.proxies.fromName()} instead.
   */
  static async fromName(
    name: string,
    params?: ProxyFromNameParams,
  ): Promise<Proxy> {
    return getDefaultClient().proxies.fromName(name, params);
  }
}
