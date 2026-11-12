import { ClientError, Status, type CallOptions } from "nice-grpc";
import { v4 as uuidv4 } from "uuid";
import {
  FileDescriptor,
  Resources,
  Sandbox as SandboxDefinition,
  SandboxContainerCreateV2Request,
  SandboxGetTaskIdRequest,
  SandboxRestoreRequest_SandboxNameOverrideType,
  type SandboxTagsGetResponse,
} from "../proto/modal_proto/api";
import { TaskSnapshotMemoryRequest } from "../proto/modal_proto/task_command_router";
import type { App } from "./app";
import {
  type ModalClient,
  type RetryOptions,
  type TimeoutOptions,
} from "./client";
import { parseBooleanFlag } from "./config";
import {
  AlreadyExistsError,
  ConflictError,
  InvalidError,
  NotFoundError,
} from "./errors";
import type { Image } from "./image";
import type { Logger } from "./logger";
import { checkObjectName } from "./name_utils";
import {
  type SandboxCreateConnectCredentials,
  type SandboxCreateParams,
  type SandboxExperimentalFromNameParams,
  type SandboxExperimentalListParams,
  Tunnel,
  buildSandboxCreateV2RequestProto,
  encodeIfString,
  hydrateSandboxSecrets,
  newSandboxImpl,
} from "./sandbox";
import {
  type SandboxImpl,
  type SidecarCreateOptions,
  type SidecarCreateRequest,
  type SidecarCreateResult,
  SandboxImplBase,
} from "./sandbox_impl";
import { SandboxSnapshot } from "./sandbox_snapshot";
import { resolvableSecrets, secretSources, validateEnvVarKeys } from "./secret";
import {
  decodeTextIter,
  type ModalReadStream,
  type ModalWriteStream,
  streamConsumingIter,
  toModalReadStream,
  toModalWriteStream,
} from "./streams";
import type { TaskCommandRouterClientImpl } from "./task_command_router_client";

/**
 * Opts a client out of sending sidecar create requests to the Modal server:
 * set it to "0" or "false" to create sidecars over the Sandbox connection
 * instead.
 */
const CONTROL_PLANE_SIDECAR_CREATE_ENV_VAR =
  "MODAL_USE_CONTROL_PLANE_SIDECAR_CREATE";

/**
 * Whether sidecar create sends its request to the Modal server, which V2
 * Sandboxes do unless opted out.
 */
function useControlPlaneSidecarCreate(): boolean {
  const value = process.env[CONTROL_PLANE_SIDECAR_CREATE_ENV_VAR];
  return value === undefined || parseBooleanFlag(value);
}

/**
 * Sandbox backend whose control plane RPCs are the V2 Sandbox RPCs and whose
 * stdio goes through the task command router.
 *
 * @internal
 * @hidden
 */
export class SandboxImplV2 extends SandboxImplBase {
  protected readonly isV2 = true;

  /** Create a V2 Sandbox. */
  static async create(
    client: ModalClient,
    app: App,
    image: Image,
    params: SandboxCreateParams,
  ): Promise<SandboxImpl> {
    await image.build(app);

    // Secrets and params.env are sent as ordered secret sources, so unlike
    // the V1 backend we don't fold env vars into a server-side Secret.
    // Locally-created Secrets (fromObject) are inlined, avoiding a
    // SecretGetOrCreate round-trip; only the remaining resolvable Secrets
    // (e.g. from fromName) need hydrating to secret IDs, which are also
    // recorded on the definition.
    validateEnvVarKeys(params.env ?? {});
    const resolvable = resolvableSecrets(params.secrets ?? []);

    await hydrateSandboxSecrets(
      client,
      [...resolvable, ...(params.experimentalOutboundPolicy?._secrets() ?? [])],
      params.cloudBucketMounts,
    );

    const createReq = await buildSandboxCreateV2RequestProto(
      app.appId,
      image.imageId,
      { ...params, secrets: resolvable },
      secretSources(params.secrets ?? [], params.env),
    );
    let createResp;
    try {
      createResp = await client.cpClient.sandboxCreateV2(createReq);
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.ALREADY_EXISTS) {
        throw new AlreadyExistsError(err.details || err.message);
      }
      throw err;
    }

    client.logger.debug(
      "Created experimental V2 Sandbox",
      "sandbox_id",
      createResp.sandboxId,
    );

    // Unencrypted tunnel endpoints are not known at create time. Only cache a
    // complete set so tunnels() fetches the rest otherwise.
    const createRespHasAllTunnels =
      createResp.tunnels.length > 0 &&
      createResp.tunnels.length ===
        (createReq.definition?.openPorts?.ports.length ?? 0);
    const tunnels = createRespHasAllTunnels
      ? Object.fromEntries(
          createResp.tunnels.map((t) => [
            t.containerPort,
            new Tunnel(t.host, t.port, t.unencryptedHost, t.unencryptedPort),
          ]),
        )
      : undefined;

    return newSandboxImpl(client, createResp.sandboxId, {
      taskId: createResp.taskId,
      tunnels,
      initCommandRouterAccess: createResp.commandRouterAccess,
    });
  }

  /** Look up a V2 Sandbox by name. */
  static async fromName(
    client: ModalClient,
    appName: string,
    name: string,
    params: SandboxExperimentalFromNameParams | undefined,
  ): Promise<SandboxImpl> {
    try {
      // SandboxGetFromNameV2 only returns V2 Sandboxes and authenticates via
      // the auth-token metadata (attached automatically by the client), like
      // the other V2 Sandbox RPCs.
      const resp = await client.cpClient.sandboxGetFromNameV2({
        sandboxName: name,
        appName,
        environmentName: client.environmentName(params?.environment),
        includeTerminated: params?.includeTerminated ?? false,
      });

      return newSandboxImpl(client, resp.sandboxId);
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(
          `Sandbox with name '${name}' not found in App '${appName}'`,
        );
      throw err;
    }
  }

  /** List running Sandboxes through the V2 list RPC. */
  static async *list(
    client: ModalClient,
    params: SandboxExperimentalListParams,
  ): AsyncGenerator<SandboxImpl, void, unknown> {
    let environmentName: string | undefined = undefined;
    if (!params.appId) {
      client.logger.warn(
        "experimentalList without an `appId` lists across the whole environment and is deprecated; pass `appId` to scope the query.",
      );
      environmentName = client.environmentName(params.environment);
    }

    const tagsList = params.tags
      ? Object.entries(params.tags).map(([tagName, tagValue]) => ({
          tagName,
          tagValue,
        }))
      : [];

    let beforeTimestamp: number | undefined = undefined;
    while (true) {
      try {
        const resp = await client.cpClient.sandboxListV2({
          appId: params.appId,
          beforeTimestamp,
          environmentName,
          includeFinished: false,
          tags: tagsList,
        });
        if (!resp.sandboxes || resp.sandboxes.length === 0) {
          return;
        }
        for (const info of resp.sandboxes) {
          yield newSandboxImpl(client, info.id);
        }
        // Fetch the next batch starting from the end of the current one.
        beforeTimestamp = resp.sandboxes[resp.sandboxes.length - 1].createdAt;
      } catch (err) {
        if (
          err instanceof ClientError &&
          err.code === Status.INVALID_ARGUMENT
        ) {
          throw new InvalidError(err.details || err.message);
        }
        throw err;
      }
    }
  }

  /** Restore a V2 memory snapshot into a new V2 Sandbox. */
  static async fromSnapshot(
    client: ModalClient,
    snapshot: SandboxSnapshot,
    overrideType: SandboxRestoreRequest_SandboxNameOverrideType,
    nameOverride: string | undefined,
  ): Promise<SandboxImpl> {
    const restoreResp = await client.cpClient.sandboxRestoreV2({
      snapshotId: snapshot.snapshotId,
      sandboxNameOverrideType: overrideType,
      sandboxNameOverride: nameOverride ?? "",
    });
    return newSandboxImpl(client, restoreResp.sandboxId, {
      taskId: restoreResp.taskId,
      initCommandRouterAccess: restoreResp.commandRouterAccess,
    });
  }

  stdin(): ModalWriteStream<string> {
    return toModalWriteStream(inputStreamSbV2(() => this.getCommandRouter()));
  }

  outputStream(
    fileDescriptor: FileDescriptor,
    ensureAttached: () => void,
  ): ModalReadStream<string> {
    ensureAttached();
    const abort = new AbortController();
    return toModalReadStream(
      streamConsumingIter(
        decodeTextIter(
          outputStreamSbV2(
            () => this.getCommandRouter(abort.signal),
            this.sandboxId,
            fileDescriptor,
            this.client.logger,
            abort.signal,
          ),
        ),
        () => abort.abort(),
      ),
    );
  }

  protected rpcWait(timeoutSecs: number) {
    return this.client.cpClient.sandboxWaitV2({
      sandboxId: this.sandboxId,
      timeout: timeoutSecs,
    });
  }

  protected rpcGetTaskId(timeoutSecs?: number, options?: CallOptions) {
    const req = SandboxGetTaskIdRequest.create({
      sandboxId: this.sandboxId,
      timeout: timeoutSecs,
    });
    return this.client.cpClient.sandboxGetTaskIdV2(req, options);
  }

  protected rpcGetTunnels(timeoutMs: number) {
    return this.client.cpClient.sandboxGetTunnelsV2({
      sandboxId: this.sandboxId,
      timeout: timeoutMs / 1000,
    });
  }

  protected rpcGetExitSnapshot(
    timeoutSecs: number,
    options?: CallOptions & TimeoutOptions & RetryOptions,
  ) {
    return this.client.cpClient.sandboxGetExitSnapshotV2(
      { sandboxId: this.sandboxId, timeout: timeoutSecs },
      options,
    );
  }

  protected async rpcTerminate(): Promise<void> {
    await this.client.cpClient.sandboxTerminateV2({
      sandboxId: this.sandboxId,
    });
  }

  async setTags(tags: Record<string, string>): Promise<void> {
    const tagsList = Object.entries(tags).map(([tagName, tagValue]) => ({
      tagName,
      tagValue,
    }));
    try {
      await this.client.cpClient.sandboxTagsSetV2({
        sandboxId: this.sandboxId,
        tags: tagsList,
      });
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.INVALID_ARGUMENT) {
        throw new InvalidError(err.details || err.message);
      }
      throw err;
    }
  }

  async getTags(): Promise<Record<string, string>> {
    let resp: SandboxTagsGetResponse;
    try {
      resp = await this.client.cpClient.sandboxTagsGetV2({
        sandboxId: this.sandboxId,
      });
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.INVALID_ARGUMENT) {
        throw new InvalidError(err.details || err.message);
      }
      throw err;
    }

    const tags: Record<string, string> = {};
    for (const tag of resp.tags) {
      tags[tag.tagName] = tag.tagValue;
    }
    return tags;
  }

  async setName(name: string): Promise<void> {
    checkObjectName(name, "Sandbox");
    try {
      await this.client.cpClient.sandboxSetName({
        sandboxId: this.sandboxId,
        name,
      });
    } catch (err) {
      if (err instanceof ClientError) {
        const message = err.details || err.message;
        switch (err.code) {
          case Status.ALREADY_EXISTS:
            throw new AlreadyExistsError(message);
          case Status.INVALID_ARGUMENT:
            throw new InvalidError(message);
          case Status.FAILED_PRECONDITION:
            throw new ConflictError(message);
        }
      }
      throw err;
    }
  }

  async createConnectToken(
    port: number,
    userMetadata: string | undefined,
  ): Promise<SandboxCreateConnectCredentials> {
    const resp = await this.client.cpClient.sandboxCreateConnectTokenV2({
      sandboxId: this.sandboxId,
      userMetadata,
      port,
    });
    return { url: resp.url, token: resp.token };
  }

  async snapshotMemory(): Promise<SandboxSnapshot> {
    const [taskId, commandRouterClient] = await this.getCommandRouter();
    const resp = await commandRouterClient.snapshotMemory(
      TaskSnapshotMemoryRequest.create({ taskId, idempotencyKey: uuidv4() }),
      { timeoutMs: 300000 },
    );
    return new SandboxSnapshot(this.client, resp.snapshotId, { isV2: true });
  }

  validateSidecarCreate(options: SidecarCreateOptions): void {
    if (useControlPlaneSidecarCreate()) {
      return;
    }
    if (options.hasCloudBucketMounts) {
      throw new InvalidError(
        `cloudBucketMounts are not supported in sidecars when ${CONTROL_PLANE_SIDECAR_CREATE_ENV_VAR}=0 is set; unset it to use them`,
      );
    }
    if (options.includeOidcIdentityToken) {
      throw new InvalidError(
        `includeOidcIdentityToken is not supported when ${CONTROL_PLANE_SIDECAR_CREATE_ENV_VAR}=0 is set; unset it to use it`,
      );
    }
    if (options.hasProxy) {
      throw new InvalidError(
        `proxy is not supported when ${CONTROL_PLANE_SIDECAR_CREATE_ENV_VAR}=0 is set; unset it to use a proxy`,
      );
    }
  }

  async createSidecar(
    request: SidecarCreateRequest,
  ): Promise<SidecarCreateResult> {
    if (!useControlPlaneSidecarCreate()) {
      return this.createSidecarViaCommandRouter(request);
    }
    const resp = await this.client.cpClient.sandboxContainerCreateV2(
      SandboxContainerCreateV2Request.create({
        sandboxId: this.sandboxId,
        containerName: request.containerName,
        definition: SandboxDefinition.create({
          imageId: request.imageId,
          entrypointArgs: request.command,
          workdir: request.workdir,
          secretIds: request.secretIds,
          volumeMounts: request.volumeMounts,
          cloudBucketMounts: request.cloudBucketMounts,
          networkAccess: request.networkAccess,
          includeOidcIdentityToken: request.includeOidcIdentityToken,
          proxyId: request.proxyId,
          ptyInfo: request.ptyInfo,
          resources:
            request.memoryReserveConsumeMiB !== undefined
              ? Resources.create({ memoryMb: request.memoryReserveConsumeMiB })
              : undefined,
        }),
        secretSources: request.secretSources,
      }),
    );
    return { containerId: resp.containerId, containerName: resp.containerName };
  }
}

async function* outputStreamSbV2(
  getCommandRouter: () => Promise<[string, TaskCommandRouterClientImpl]>,
  sandboxId: string,
  fileDescriptor: FileDescriptor,
  logger: Logger,
  signal: AbortSignal,
): AsyncIterable<Uint8Array> {
  const [taskId, commandRouterClient] = await getCommandRouter();
  let firstChunk = true;
  for await (const item of commandRouterClient.sandboxStdioReadV2(
    taskId,
    fileDescriptor,
    signal,
  )) {
    if (item.data.length === 0) {
      throw new Error(
        `Received empty message streaming stdio from Sandbox ${sandboxId}`,
      );
    }
    if (firstChunk) {
      firstChunk = false;
      if (item.startingOffset > 0) {
        logger.warn(
          "V2 Sandbox stdio: dropped bytes. Only the most recent portion of output is retained",
          "sandbox_id",
          sandboxId,
          "dropped_bytes",
          item.startingOffset,
        );
      }
    }
    yield item.data;
  }
}

function inputStreamSbV2(
  getCommandRouter: () => Promise<[string, TaskCommandRouterClientImpl]>,
): WritableStream<string> {
  let offset = 0;
  return new WritableStream<string>({
    async write(chunk) {
      const [taskId, commandRouterClient] = await getCommandRouter();
      const data = encodeIfString(chunk);
      await commandRouterClient.sandboxStdinWriteV2(
        taskId,
        offset,
        data,
        false, // eof
      );
      offset += data.length;
    },
    async close() {
      const [taskId, commandRouterClient] = await getCommandRouter();
      await commandRouterClient.sandboxStdinWriteV2(
        taskId,
        offset,
        new Uint8Array(0),
        true, // eof
      );
    },
  });
}
