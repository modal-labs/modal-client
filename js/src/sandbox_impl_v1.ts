import { ClientError, Status, type CallOptions } from "nice-grpc";
import { setTimeout } from "timers/promises";
import {
  FileDescriptor,
  GenericResult_GenericStatus,
  SandboxGetTaskIdRequest,
  SandboxRestoreRequest_SandboxNameOverrideType,
  type SandboxTagsGetResponse,
} from "../proto/modal_proto/api";
import type { App } from "./app";
import {
  type ModalClient,
  type ModalGrpcClient,
  type RetryOptions,
  type TimeoutOptions,
  isRetryableGrpc,
} from "./client";
import {
  AlreadyExistsError,
  ExecutionError,
  InvalidError,
  NotFoundError,
} from "./errors";
import { IdleCountdown } from "./idle_countdown";
import type { Image } from "./image";
import {
  type SandboxCreateConnectCredentials,
  type SandboxCreateParams,
  type SandboxFromNameParams,
  type SandboxListParams,
  SANDBOX_SCHEDULING_TIMEOUT_MS,
  buildSandboxCreateRequestProto,
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
import { mergeEnvIntoSecrets } from "./secret";
import {
  type ModalReadStream,
  type ModalWriteStream,
  streamConsumingIter,
  toModalReadStream,
  toModalWriteStream,
} from "./streams";

// Backoff configuration for SandboxGetLogs retry behavior.
const SB_LOGS_INITIAL_DELAY_MS = 10;
const SB_LOGS_DELAY_FACTOR = 2;
const SB_LOGS_MAX_RETRIES = 10;

/**
 * Sandbox backend whose control plane RPCs are the original (V1) Sandbox RPCs.
 *
 * @internal
 * @hidden
 */
export class SandboxImplV1 extends SandboxImplBase {
  protected readonly isV2 = false;

  /** Create a V1 Sandbox and wait for it to be scheduled. */
  static async create(
    client: ModalClient,
    app: App,
    image: Image,
    params: SandboxCreateParams,
  ): Promise<SandboxImpl> {
    await image.build(app);

    const mergedSecrets = await mergeEnvIntoSecrets(
      client,
      params.env,
      params.secrets,
    );

    // The SandboxCreate request only carries secret IDs, so any locally-created
    // Secrets (and env vars) must be hydrated into server-side Secrets first.
    await hydrateSandboxSecrets(
      client,
      [
        ...mergedSecrets,
        ...(params.experimentalOutboundPolicy?._secrets() ?? []),
      ],
      params.cloudBucketMounts,
    );

    const mergedParams = {
      ...params,
      secrets: mergedSecrets,
      env: undefined, // setting env to undefined just to clarify it's not needed anymore
    };

    const createReq = await buildSandboxCreateRequestProto(
      app.appId,
      image.imageId,
      mergedParams,
    );
    let createResp;
    try {
      createResp = await client.cpClient.sandboxCreate(createReq);
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.ALREADY_EXISTS) {
        throw new AlreadyExistsError(err.details || err.message);
      }
      throw err;
    }

    client.logger.debug("Created Sandbox", "sandbox_id", createResp.sandboxId);
    const impl = newSandboxImpl(client, createResp.sandboxId);
    await impl.waitForScheduling(SANDBOX_SCHEDULING_TIMEOUT_MS);
    return impl;
  }

  /** Look up a running V1 Sandbox by name. */
  static async fromName(
    client: ModalClient,
    appName: string,
    name: string,
    params: SandboxFromNameParams | undefined,
  ): Promise<SandboxImpl> {
    try {
      const resp = await client.cpClient.sandboxGetFromName({
        sandboxName: name,
        appName,
        environmentName: client.environmentName(params?.environment),
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

  /** List running Sandboxes through the V1 list RPC. */
  static async *list(
    client: ModalClient,
    params: SandboxListParams,
  ): AsyncGenerator<SandboxImpl, void, unknown> {
    const env = client.environmentName(params.environment);
    const tagsList = params.tags
      ? Object.entries(params.tags).map(([tagName, tagValue]) => ({
          tagName,
          tagValue,
        }))
      : [];

    let beforeTimestamp: number | undefined = undefined;
    while (true) {
      try {
        const resp = await client.cpClient.sandboxList({
          appId: params.appId,
          beforeTimestamp,
          environmentName: env,
          includeFinished: false,
          tags: tagsList,
        });
        if (!resp.sandboxes || resp.sandboxes.length === 0) {
          return;
        }
        for (const info of resp.sandboxes) {
          yield newSandboxImpl(client, info.id);
        }
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

  /** Restore a V1 memory snapshot into a new V1 Sandbox and wait for it to be ready. */
  static async fromSnapshot(
    client: ModalClient,
    snapshot: SandboxSnapshot,
    overrideType: SandboxRestoreRequest_SandboxNameOverrideType,
    nameOverride: string | undefined,
  ): Promise<SandboxImpl> {
    const restoreResp = await client.cpClient.sandboxRestore({
      snapshotId: snapshot.snapshotId,
      sandboxNameOverrideType: overrideType,
      sandboxNameOverride: nameOverride ?? "",
    });

    const impl = newSandboxImpl(client, restoreResp.sandboxId);

    const resp = await client.cpClient.sandboxGetTaskId({
      sandboxId: restoreResp.sandboxId,
      waitUntilReady: true,
      timeout: 55,
    });
    if (
      resp.taskResult &&
      resp.taskResult.status !==
        GenericResult_GenericStatus.GENERIC_STATUS_UNSPECIFIED &&
      resp.taskResult.status !==
        GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS
    ) {
      throw new ExecutionError(resp.taskResult.exception);
    }
    return impl;
  }

  stdin(): ModalWriteStream<string> {
    return toModalWriteStream(
      inputStreamSb(this.client.cpClient, this.sandboxId),
    );
  }

  // V1 output is served by the control plane, so it stays readable after the
  // Sandbox handle is detached.
  outputStream(fileDescriptor: FileDescriptor): ModalReadStream<string> {
    const abort = new AbortController();
    return toModalReadStream(
      streamConsumingIter(
        outputStreamSb(
          this.client.cpClient,
          this.sandboxId,
          fileDescriptor,
          this.client.profile.sandboxChannelIdleTimeoutMs,
          abort.signal,
        ),
        () => abort.abort(),
      ),
    );
  }

  protected rpcWait(timeoutSecs: number) {
    return this.client.cpClient.sandboxWait({
      sandboxId: this.sandboxId,
      timeout: timeoutSecs,
    });
  }

  protected rpcGetTaskId(timeoutSecs?: number, options?: CallOptions) {
    const req = SandboxGetTaskIdRequest.create({
      sandboxId: this.sandboxId,
      timeout: timeoutSecs,
    });
    return this.client.cpClient.sandboxGetTaskId(req, options);
  }

  protected rpcGetTunnels(timeoutMs: number) {
    return this.client.cpClient.sandboxGetTunnels({
      sandboxId: this.sandboxId,
      timeout: timeoutMs / 1000,
    });
  }

  protected rpcGetExitSnapshot(
    timeoutSecs: number,
    options?: CallOptions & TimeoutOptions & RetryOptions,
  ) {
    return this.client.cpClient.sandboxGetExitSnapshot(
      { sandboxId: this.sandboxId, timeout: timeoutSecs },
      options,
    );
  }

  protected async rpcTerminate(): Promise<void> {
    await this.client.cpClient.sandboxTerminate({ sandboxId: this.sandboxId });
  }

  async setTags(tags: Record<string, string>): Promise<void> {
    const tagsList = Object.entries(tags).map(([tagName, tagValue]) => ({
      tagName,
      tagValue,
    }));
    try {
      await this.client.cpClient.sandboxTagsSet({
        environmentName: this.client.environmentName(),
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
      resp = await this.client.cpClient.sandboxTagsGet({
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

  async setName(_name: string): Promise<void> {
    throw new InvalidError(
      "Sandbox.experimentalSetName is only supported for V2 sandboxes",
    );
  }

  async createConnectToken(
    port: number,
    userMetadata: string | undefined,
  ): Promise<SandboxCreateConnectCredentials> {
    const resp = await this.client.cpClient.sandboxCreateConnectToken({
      sandboxId: this.sandboxId,
      userMetadata,
      port,
    });
    return { url: resp.url, token: resp.token };
  }

  async snapshotMemory(): Promise<SandboxSnapshot> {
    await this.getTaskId();
    const snapResp = await this.client.cpClient.sandboxSnapshot({
      sandboxId: this.sandboxId,
    });
    const snapshotId = snapResp.snapshotId;

    // wait for the snapshot to succeed. this is implemented as a second idempotent rpc
    // because the snapshot itself may take a while to complete.
    const waitResp = await this.client.cpClient.sandboxSnapshotWait({
      snapshotId,
      timeout: 55,
    });
    if (
      waitResp.result?.status !==
      GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS
    ) {
      throw new ExecutionError(waitResp.result?.exception ?? "");
    }

    return new SandboxSnapshot(this.client, snapshotId, { isV2: false });
  }

  validateSidecarCreate(options: SidecarCreateOptions): void {
    if (options.hasCloudBucketMounts) {
      throw new InvalidError(
        "cloudBucketMounts are not supported in sidecars of V1 Sandboxes. A Sandbox is V1 when it has a GPU or MODAL_SANDBOX_V2=0 is set; contact Modal support for more information",
      );
    }
    if (options.includeOidcIdentityToken) {
      throw new InvalidError(
        "includeOidcIdentityToken is not supported for GPU Sandboxes",
      );
    }
    if (options.hasProxy) {
      throw new InvalidError("proxy is not supported for GPU Sandboxes");
    }
  }

  async createSidecar(
    request: SidecarCreateRequest,
  ): Promise<SidecarCreateResult> {
    return this.createSidecarViaCommandRouter(request);
  }
}

async function* outputStreamSb(
  cpClient: ModalGrpcClient,
  sandboxId: string,
  fileDescriptor: FileDescriptor,
  idleTimeoutMs: number,
  signal?: AbortSignal,
): AsyncIterable<string> {
  let lastIndex = "0-0";
  let completed = false;
  let retriesRemaining = SB_LOGS_MAX_RETRIES;
  let delayMs = SB_LOGS_INITIAL_DELAY_MS;
  while (!completed) {
    if (signal?.aborted) return;
    // Given back when nobody has read for the idle timeout. The caller is
    // suspended at a yield for exactly as long as it is not reading, so that
    // is where the countdown runs.
    const idle = new IdleCountdown();
    const idleAbort = new AbortController();
    const releaseIfIdle = () => {
      idle.stop();
      idleAbort.abort();
    };
    const callSignal =
      signal === undefined
        ? idleAbort.signal
        : AbortSignal.any([signal, idleAbort.signal]);
    try {
      const outputIterator = cpClient.sandboxGetLogs(
        {
          sandboxId,
          fileDescriptor,
          timeout: 55,
          lastEntryId: lastIndex,
        },
        { signal: callSignal },
      );
      for await (const batch of outputIterator) {
        // Successful read - reset backoff counters.
        delayMs = SB_LOGS_INITIAL_DELAY_MS;
        retriesRemaining = SB_LOGS_MAX_RETRIES;
        lastIndex = batch.entryId;
        // Finish the buffered batch before reopening from its entry ID.
        for (const item of batch.items) {
          if (!idleAbort.signal.aborted) {
            idle.arm(idleTimeoutMs, releaseIfIdle);
          }
          try {
            yield item.data;
          } finally {
            idle.stop();
          }
        }
        if (batch.eof) {
          completed = true;
          break;
        }
        if (signal?.aborted) {
          return;
        }
        if (idleAbort.signal.aborted) {
          break;
        }
      }
      if (idleAbort.signal.aborted) {
        // Given back while nobody was reading: carry on from lastEntryId,
        // without spending the retry budget on it.
        continue;
      }
    } catch (err) {
      // If cancelled, exit cleanly regardless of error type.
      if (signal?.aborted) {
        return;
      }
      if (idleAbort.signal.aborted) {
        // Our own abort, surfacing as the call unwinds: leaving an aborted
        // call raises it rather than ending quietly. Nothing is classified
        // here - the abort belongs to this reader, so it can only mean the
        // stream was given back while nobody was reading. Carry on from
        // lastEntryId, without spending the retry budget on it.
        continue;
      }
      if (isRetryableGrpc(err) && retriesRemaining > 0) {
        // Short exponential backoff to avoid tight retry loops.
        try {
          await setTimeout(delayMs, undefined, { signal });
        } catch {
          // Abort during sleep - exit cleanly.
          return;
        }
        delayMs *= SB_LOGS_DELAY_FACTOR;
        retriesRemaining--;
        continue;
      } else {
        throw err;
      }
    } finally {
      idle.stop();
    }
  }
}

// outputStreamSbV2 reads a V2 Sandbox's stdout or stderr. The first read is
// what resolves the task and opens the stream.

function inputStreamSb(
  cpClient: ModalGrpcClient,
  sandboxId: string,
): WritableStream<string> {
  let index = 1;
  return new WritableStream<string>({
    async write(chunk) {
      await cpClient.sandboxStdinWrite({
        sandboxId,
        input: encodeIfString(chunk),
        index,
      });
      index++;
    },
    async close() {
      await cpClient.sandboxStdinWrite({
        sandboxId,
        index,
        eof: true,
      });
    },
  });
}
