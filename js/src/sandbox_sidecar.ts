import { ClientError, Status } from "nice-grpc";

import {
  GenericResult,
  GenericResult_GenericStatus,
} from "../proto/modal_proto/api";
import {
  TaskContainerCreateRequest,
  TaskContainerGetRequest,
  TaskContainerInfo,
  TaskContainerListRequest,
  TaskContainerTerminateRequest,
  TaskContainerWaitRequest,
} from "../proto/modal_proto/task_command_router";
import {
  ContainerProcess,
  getReturnCode,
  validateExecArgs,
  validateWorkdir,
  type SandboxExecParams,
} from "./sandbox";
import { SandboxFilesystem } from "./sandbox_fs";
import type { TaskCommandRouterClientImpl } from "./task_command_router_client";
import {
  AlreadyExistsError,
  InternalFailure,
  InvalidError,
  NotFoundError,
} from "./errors";
import type { Image } from "./image";
import type { ModalClient } from "./client";
import {
  collectSecretIds,
  hydrateSecrets,
  splitEnvDictAndResolvableSecrets,
  validateEnvVarKeys,
  type Secret,
} from "./secret";

/** Reserved name of a Sandbox's main container. */
const MAIN_CONTAINER_NAME = "main";

/**
 * Per-RPC server-side wait timeout (in seconds) used while polling for a
 * sidecar container's terminal status.
 */
const CONTAINER_WAIT_POLL_TIMEOUT_SECONDS = 10;

type SandboxSidecarCommandRouter = Pick<
  TaskCommandRouterClientImpl,
  | "containerCreate"
  | "containerGet"
  | "containerList"
  | "containerTerminate"
  | "containerWait"
>;

type SandboxSidecarAccess = {
  client: ModalClient;
  exec(
    command: string[],
    params: SandboxExecParams | undefined,
    containerId: string,
  ): Promise<ContainerProcess>;
  commandRouter(): Promise<[string, SandboxSidecarCommandRouter]>;
};

/** Options for {@link SidecarService#create SidecarService.create()}. */
export type SidecarCreateParams = {
  /** Command to run in the sidecar container on startup. */
  command?: string[];
  /** Environment variables to set in the sidecar container. */
  env?: Record<string, string>;
  /** {@link Secret}s to inject into the sidecar container as environment variables. */
  secrets?: Secret[];
  /** Working directory of the sidecar container. */
  workdir?: string;
};

/** Options for {@link SidecarService#get SidecarService.get()}. */
export type SidecarGetParams = {
  /** If true, return the latest container with the name even if it has terminated. */
  includeTerminated?: boolean;
};

/** Options for {@link SidecarService#list SidecarService.list()}. */
export type SidecarListParams = {
  /** If true, include terminated containers. */
  includeTerminated?: boolean;
};

/** Options for {@link SidecarContainer#exec SidecarContainer.exec()}. */
export type SidecarExecParams = SandboxExecParams;

/** Options for {@link SidecarContainer#terminate SidecarContainer.terminate()}. */
export type SidecarTerminateParams = {
  /** If true, wait for the sidecar container to terminate. */
  wait?: boolean;
};

function validateSidecarName(name: string): void {
  if (name === "") {
    throw new InvalidError("sidecar name must not be empty");
  }
  if (name === MAIN_CONTAINER_NAME) {
    throw new InvalidError(
      `the name "${MAIN_CONTAINER_NAME}" is reserved for the Sandbox's main ` +
        `container. Use the Sandbox methods directly to interact with it`,
    );
  }
}

function sidecarContainerFromProto(
  access: SandboxSidecarAccess,
  info: TaskContainerInfo,
): SidecarContainer {
  return new SidecarContainer(
    access,
    info.containerId,
    info.containerName,
    info.result,
  );
}

/**
 * Creates and manages sidecar containers inside a Sandbox.
 *
 * EXPERIMENTAL: the API is subject to change.
 */
export class SidecarService {
  readonly #access: SandboxSidecarAccess;

  /** @internal @hidden */
  constructor(access: SandboxSidecarAccess) {
    this.#access = access;
  }

  /**
   * Start a new sidecar container in the Sandbox. The {@link Image} must
   * already be built by calling {@link Image#build Image.build()} before it is
   * passed to `create`.
   */
  async create(
    name: string,
    image: Image,
    params?: SidecarCreateParams,
  ): Promise<SidecarContainer> {
    validateSidecarName(name);
    if (!image || image.imageId === "") {
      throw new InvalidError(
        "sidecar image must already be built. Call image.build(app) first " +
          "or use client.images.fromId(...)",
      );
    }
    const command = params?.command ?? [];
    validateExecArgs(command);
    validateWorkdir(params?.workdir);

    // Locally-created Secrets (fromObject) are passed directly to the worker as
    // environment variables, so only the remaining Secrets need hydrating. This
    // avoids a SecretGetOrCreate round-trip for env-dict Secrets.
    validateEnvVarKeys(params?.env ?? {});
    const [envDict, resolvableSecrets] = splitEnvDictAndResolvableSecrets(
      params?.secrets ?? [],
    );
    Object.assign(envDict, params?.env ?? {});
    await hydrateSecrets(this.#access.client, resolvableSecrets);
    const secretIds = collectSecretIds(resolvableSecrets);

    const [taskId, client] = await this.#access.commandRouter();

    let resp;
    try {
      resp = await client.containerCreate(
        TaskContainerCreateRequest.create({
          taskId,
          containerName: name,
          imageId: image.imageId,
          args: command,
          env: envDict,
          workdir: params?.workdir ?? "",
          secretIds,
        }),
      );
    } catch (err) {
      if (err instanceof ClientError) {
        if (err.code === Status.ALREADY_EXISTS) {
          throw new AlreadyExistsError(err.details || err.message);
        }
        if (err.code === Status.INVALID_ARGUMENT) {
          throw new InvalidError(err.details || err.message);
        }
      }
      throw err;
    }

    return new SidecarContainer(
      this.#access,
      resp.containerId,
      resp.containerName || name,
    );
  }

  /** Return a sidecar container by name. */
  async get(
    name: string,
    params?: SidecarGetParams,
  ): Promise<SidecarContainer> {
    validateSidecarName(name);

    const [taskId, client] = await this.#access.commandRouter();

    let resp;
    try {
      resp = await client.containerGet(
        TaskContainerGetRequest.create({
          taskId,
          containerName: name,
          includeTerminated: params?.includeTerminated ?? false,
        }),
      );
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND) {
        throw new NotFoundError(`Sidecar container "${name}" not found`);
      }
      throw err;
    }
    if (!resp.container) {
      throw new InternalFailure(
        `server returned no container for sidecar "${name}"`,
      );
    }
    return sidecarContainerFromProto(this.#access, resp.container);
  }

  /** Return all sidecar containers (not including the main container). */
  async list(params?: SidecarListParams): Promise<SidecarContainer[]> {
    const [taskId, client] = await this.#access.commandRouter();

    const resp = await client.containerList(
      TaskContainerListRequest.create({
        taskId,
        includeTerminated: params?.includeTerminated ?? false,
      }),
    );

    return resp.containers
      .filter((info) => info.containerName !== MAIN_CONTAINER_NAME)
      .map((info) => sidecarContainerFromProto(this.#access, info));
  }
}

/**
 * A handle to a sidecar container running in a Sandbox.
 *
 * EXPERIMENTAL: the API is subject to change.
 */
export class SidecarContainer {
  /** The fully qualified container ID. */
  readonly containerId: string;
  /** The logical name of the container within the Sandbox. */
  readonly containerName: string;

  readonly #access: SandboxSidecarAccess;
  #result?: GenericResult;
  #filesystem?: SandboxFilesystem;

  /** @internal @hidden */
  constructor(
    access: SandboxSidecarAccess,
    containerId: string,
    containerName: string,
    result?: GenericResult,
  ) {
    this.#access = access;
    this.containerId = containerId;
    this.containerName = containerName;
    this.#result = result;
  }

  async exec(
    command: string[],
    params?: SidecarExecParams & { mode?: "text" },
  ): Promise<ContainerProcess<string>>;

  async exec(
    command: string[],
    params: SidecarExecParams & { mode: "binary" },
  ): Promise<ContainerProcess<Uint8Array>>;

  /** Run a command in the sidecar container and return the process handle. */
  async exec(
    command: string[],
    params?: SidecarExecParams,
  ): Promise<ContainerProcess> {
    return this.#access.exec(command, params, this.containerId);
  }

  /** Namespace for filesystem APIs scoped to this sidecar container. */
  get filesystem(): SandboxFilesystem {
    if (!this.#filesystem) {
      this.#filesystem = new SandboxFilesystem((command, params) =>
        this.#access.exec(command, params, this.containerId),
      );
    }
    return this.#filesystem;
  }

  /** Block until the sidecar container exits, and return its exit code. */
  async wait(): Promise<number> {
    if (
      this.#result &&
      this.#result.status !==
        GenericResult_GenericStatus.GENERIC_STATUS_UNSPECIFIED
    ) {
      return getReturnCode(this.#result) ?? 0;
    }

    const [taskId, client] = await this.#access.commandRouter();
    while (true) {
      const resp = await client.containerWait(
        TaskContainerWaitRequest.create({
          taskId,
          containerId: this.containerId,
          timeout: CONTAINER_WAIT_POLL_TIMEOUT_SECONDS,
        }),
      );
      const result = resp.result;
      if (
        !result ||
        result.status === GenericResult_GenericStatus.GENERIC_STATUS_UNSPECIFIED
      ) {
        continue;
      }
      this.#result = result;
      return getReturnCode(result) ?? 0;
    }
  }

  /**
   * Check if the sidecar container has finished running.
   *
   * Returns `null` if the container is still running, else the exit code.
   */
  async poll(): Promise<number | null> {
    if (
      this.#result &&
      this.#result.status !==
        GenericResult_GenericStatus.GENERIC_STATUS_UNSPECIFIED
    ) {
      return getReturnCode(this.#result);
    }

    const [taskId, client] = await this.#access.commandRouter();
    const resp = await client.containerWait(
      TaskContainerWaitRequest.create({
        taskId,
        containerId: this.containerId,
        timeout: 0,
      }),
    );
    const result = resp.result;
    if (
      result &&
      result.status !== GenericResult_GenericStatus.GENERIC_STATUS_UNSPECIFIED
    ) {
      this.#result = result;
    }
    return getReturnCode(result);
  }

  async terminate(): Promise<void>;
  async terminate(params: { wait: true }): Promise<number>;
  /**
   * Stop the sidecar container.
   *
   * The returned exit code is only meaningful when `wait` is true.
   */
  async terminate(params?: SidecarTerminateParams): Promise<number | void> {
    const [taskId, client] = await this.#access.commandRouter();
    await client.containerTerminate(
      TaskContainerTerminateRequest.create({
        taskId,
        containerId: this.containerId,
      }),
    );
    if (params?.wait) {
      return this.wait();
    }
  }
}
