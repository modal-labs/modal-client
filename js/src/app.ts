import { ClientError, Status } from "nice-grpc";
import {
  AppGetInfoRequest,
  AppGetTimeRangeStatsRequest,
  type AppHandleMetadata,
  type AppLifecycle as AppLifecycleProto,
  type AppContainerCounts as AppContainerCountsProto,
  type AppFunctionCounts as AppFunctionCountsProto,
  type AppServerCounts as AppServerCountsProto,
  AppState as AppStateProto,
  AppStopRequest,
  AppStopSource,
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

/** Optional parameters for {@link App#stats}. */
export interface AppStatsParams {
  /** The beginning of the time range, inclusive. Defaults to an hour before `until`. */
  since?: Date;
  /** The end of the time range, exclusive. Defaults to the current time. */
  until?: Date;
}

/** Completed Function inputs; each map item counts separately. */
export interface AppStatsCallCounts {
  total: number;
  succeeded: number;
  failed: number;
}

/** Terminal container errors and distinct containers reporting resource activity in the window. */
export interface AppStatsContainerCounts {
  errors: number;
  total: number;
}

/** Function call and container counts. */
export interface AppStatsFunctionCounts {
  calls: AppStatsCallCounts;
  containers: AppStatsContainerCounts;
}

/** Object ID and counts for a Function. */
export interface AppStatsFunctionStats {
  objectId: string;
  counts: AppStatsFunctionCounts;
}

/** Totals across current Functions, with individual Function stats keyed by name. */
export interface AppStatsFunctionStatsGroup {
  total: AppStatsFunctionCounts;
  byName: Record<string, AppStatsFunctionStats>;
}

/** Server requests split into HTTP 1xx/2xx/3xx and all other outcomes, including status zero. */
export interface AppStatsRequestCounts {
  total: number;
  ok: number;
  failed: number;
}

/** Server request and container counts. */
export interface AppStatsServerCounts {
  requests: AppStatsRequestCounts;
  containers: AppStatsContainerCounts;
}

/** Object ID and counts for a Server. */
export interface AppStatsServerStats {
  objectId: string;
  counts: AppStatsServerCounts;
}

/** Totals across current Servers, with individual Server stats keyed by name. */
export interface AppStatsServerStatsGroup {
  total: AppStatsServerCounts;
  byName: Record<string, AppStatsServerStats>;
}

/**
 * Sandboxes that finished in the window and those that are live as of the until cutoff.
 *
 * Total is finished + live.
 */
export interface AppStatsSandboxCounts {
  total: number;
  finished: number;
  live: number;
}

/**
 * App-wide statistics over [since, until), with current Functions and Servers indexed by name.
 *
 * Lifecycle metadata describes the App when queried, including for historical windows.
 */
export interface AppStats {
  since: Date;
  until: Date;
  functions: AppStatsFunctionStatsGroup;
  servers: AppStatsServerStatsGroup;
  sandboxes: AppStatsSandboxCounts;
  lifecycle: AppLifecycle;
  deploymentCount?: number;
}

function appContainerCountsFromProto(
  counts?: AppContainerCountsProto,
): AppStatsContainerCounts {
  return {
    errors: counts?.containerErrors ?? 0,
    total: counts?.containerActiveCount ?? 0,
  };
}

function appFunctionCountsFromProto(
  counts?: AppFunctionCountsProto,
): AppStatsFunctionCounts {
  return {
    calls: {
      total: counts?.calls?.total ?? 0,
      succeeded: counts?.calls?.succeeded ?? 0,
      failed: counts?.calls?.failed ?? 0,
    },
    containers: appContainerCountsFromProto(counts?.containers),
  };
}

function appServerCountsFromProto(
  counts?: AppServerCountsProto,
): AppStatsServerCounts {
  return {
    requests: {
      total: counts?.requests?.total ?? 0,
      ok: counts?.requests?.ok ?? 0,
      failed: counts?.requests?.failed ?? 0,
    },
    containers: appContainerCountsFromProto(counts?.containers),
  };
}

function appLifecycleFromProto(lifecycle?: AppLifecycleProto): AppLifecycle {
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
  return {
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
  };
}

function appInfoFromProto(metadata: AppHandleMetadata, appId: string): AppInfo {
  return {
    appId,
    description: metadata.description,
    functions: { ...metadata.functions },
    servers: { ...metadata.servers },
    lifecycle: appLifecycleFromProto(metadata.lifecycle),
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

/** Optional parameters for {@link App#stop App.stop()}. */
export type AppStopParams = Record<never, never>;

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

  /**
   * Return aggregate Function, Server, and Sandbox statistics for this App.
   *
   * The default window is the most recent hour. Windows can span at most seven days,
   * with a start in the past 30 days.
   *
   * App stats show total call counts, request counts, and the count of active and errored containers in
   * the time window. The count of sandboxes that finished in the interval as well as the count
   * that were still running as of the end of the window is also returned.
   *
   * The available stats and their definitions are subject to change and are provided
   * on a best-effort basis. They may be delayed or incomplete.
   *
   * @param params - The beginning (inclusive) and end (exclusive) of the time range.
   * `until` defaults to the current time; `since` defaults to an hour before `until`.
   *
   * @example
   * ```ts
   * const stats = await app.stats();
   * console.log(stats.functions.total.calls.failed);
   * console.log(stats.functions.byName["embed"].counts.containers.errors);
   * ```
   */
  async stats(params: AppStatsParams = {}): Promise<AppStats> {
    if (!this.#client) {
      throw new InvalidError(
        "App.stats() requires a handle retrieved through a ModalClient.",
      );
    }
    const until = params.until ?? new Date();
    const since = params.since ?? new Date(until.getTime() - 60 * 60 * 1000);
    if (
      !Number.isFinite(since.getTime()) ||
      !Number.isFinite(until.getTime())
    ) {
      throw new InvalidError("`since` and `until` must be valid dates.");
    }
    if (since >= until) {
      throw new InvalidError("`since` must be before `until`.");
    }
    const resp = await this.#client.cpClient.appGetTimeRangeStats(
      AppGetTimeRangeStatsRequest.create({ appId: this.appId, since, until }),
    );
    return {
      since: resp.since ?? new Date(0),
      until: resp.until ?? new Date(0),
      functions: {
        total: appFunctionCountsFromProto(resp.functions?.total),
        byName: Object.fromEntries(
          Object.entries(resp.functions?.byName ?? {}).map(([name, row]) => [
            name,
            {
              objectId: row.objectId,
              counts: appFunctionCountsFromProto(row.counts),
            },
          ]),
        ),
      },
      servers: {
        total: appServerCountsFromProto(resp.servers?.total),
        byName: Object.fromEntries(
          Object.entries(resp.servers?.byName ?? {}).map(([name, row]) => [
            name,
            {
              objectId: row.objectId,
              counts: appServerCountsFromProto(row.counts),
            },
          ]),
        ),
      },
      sandboxes: {
        total: resp.sandboxes?.total ?? 0,
        finished: resp.sandboxes?.finished ?? 0,
        live: resp.sandboxes?.live ?? 0,
      },
      lifecycle: appLifecycleFromProto(resp.lifecycle),
      deploymentCount: resp.deploymentCount,
    };
  }

  /**
   * Stop the App.
   *
   * Stopping an App terminates all of its running containers and Sandboxes and
   * prevents new inputs from being processed.
   */
  async stop(_: AppStopParams = {}): Promise<void> {
    if (!this.appId) {
      throw new InvalidError(
        "App.stop() cannot be called on an App without an ID.",
      );
    }
    if (!this.#client) {
      throw new InvalidError(
        "App.stop() requires a handle retrieved through a ModalClient.",
      );
    }
    try {
      await this.#client.cpClient.appStop(
        AppStopRequest.create({
          appId: this.appId,
          source: AppStopSource.APP_STOP_SOURCE_PYTHON_CLIENT,
        }),
      );
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(`App '${this.appId}' not found`);
      throw err;
    }
  }
}
