import { ClientError, Status } from "nice-grpc";
import { setTimeout } from "timers/promises";
import {
  FileDescriptor,
  GenericResult,
  GenericResult_GenericStatus,
  PersistentSandboxCreateRequest,
  PersistentSandboxGetRequest,
  PersistentSandboxResumeRequest,
  PersistentSandboxTerminateRequest,
  PersistentSandboxTunnelsListRequest,
} from "../proto/modal_proto/api";
import type { App } from "./app";
import type { ModalClient } from "./client";
import { ClientClosedError, InvalidError } from "./errors";
import type { Image } from "./image";
import type { ExperimentalOutboundPolicy } from "./outbound_policy";
import {
  type ContainerProcess,
  type SandboxCreateConnectCredentials,
  type SandboxCreateParams,
  type SandboxExecParams,
  type SandboxExperimentalGetExitSnapshotParams,
  type SandboxMountImageParams,
  type SandboxReloadVolumesParams,
  type SandboxSnapshotDirectoryParams,
  type SandboxSnapshotFilesystemParams,
  type SandboxUpdateNetworkPolicyParams,
  type Tunnel,
  buildSandboxCreateV2RequestProto,
  getReturnCode,
  hydrateSandboxSecrets,
} from "./sandbox";
import {
  type SandboxImpl,
  type SidecarCreateOptions,
  type SidecarCreateRequest,
  type SidecarCreateResult,
  tunnelsFromProtos,
} from "./sandbox_common";
import {
  SandboxImplV2,
  inputStreamSbV2,
  outputStreamSbV2,
} from "./sandbox_impl_v2";
import type { SandboxSnapshot } from "./sandbox_snapshot";
import { resolvableSecrets, secretSources } from "./secret";
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
 * Lifetime limits of each session of a persistent Sandbox. A session that hits
 * one is paused, and resumed on next use.
 *
 * @internal
 * @hidden
 */
export const PERSISTENT_SANDBOX_SESSION_TIMEOUT_MS = 60 * 60 * 1000;
/** @internal @hidden */
export const PERSISTENT_SANDBOX_SESSION_IDLE_TIMEOUT_MS = 5 * 60 * 1000;

// How long to retry resuming while the previous session is still saving its state.
const RESUME_RETRY_TIMEOUT_MS = 120_000;
const RESUME_RETRY_INTERVAL_MS = 1000;

// How often wait() checks whether the persistent Sandbox has been terminated.
const WAIT_POLL_INTERVAL_MS = 1000;

const TERMINATED_RESULT = GenericResult.create({
  status: GenericResult_GenericStatus.GENERIC_STATUS_TERMINATED,
});

/** Errors caused by the caller rather than by the session, which a new session can't fix. */
function isNonRecoverableError(err: unknown): boolean {
  return err instanceof InvalidError || err instanceof ClientClosedError;
}

/**
 * Backs a persistent Sandbox, whose object ID is the ID of its first session.
 *
 * A persistent Sandbox runs as a sequence of regular V2 Sandboxes, its
 * sessions, each started from the saved state of the previous one. Operations
 * are forwarded to the current session. When one fails and the Sandbox hasn't
 * been terminated through this handle, the running session is looked up,
 * resuming the Sandbox from its latest saved state if there is none, and the
 * operation is retried once on the new session.
 *
 * @internal
 * @hidden
 */
export class SandboxImplPersistent implements SandboxImpl {
  readonly client: ModalClient;
  readonly sandboxId: string;
  readonly persistentSandboxId: string;

  #session: SandboxImplV2;
  #terminated = false;
  // Session replacements run one at a time, so concurrent failures on the same
  // session trigger a single lookup/resume.
  #replaceChain: Promise<unknown> = Promise.resolve();

  constructor(
    client: ModalClient,
    sandboxId: string,
    params: { persistentSandboxId: string; session?: SandboxImplV2 },
  ) {
    this.client = client;
    this.sandboxId = sandboxId;
    this.persistentSandboxId = params.persistentSandboxId;
    this.#session = params.session ?? new SandboxImplV2(client, sandboxId);
  }

  /** Create a persistent Sandbox and return the implementation for its first session. */
  static async create(
    client: ModalClient,
    app: App,
    image: Image,
    params: SandboxCreateParams,
  ): Promise<SandboxImpl> {
    if (params.timeoutMs !== undefined || params.idleTimeoutMs !== undefined) {
      throw new InvalidError(
        "Persistent Sandboxes manage their own session timeouts and don't support `timeoutMs` or `idleTimeoutMs`.",
      );
    }
    if (params.name !== undefined) {
      throw new InvalidError(
        "Named persistent Sandboxes are not supported yet.",
      );
    }
    if (params.gpu || params.pty) {
      throw new InvalidError(
        "GPUs and PTYs are not supported by persistent Sandboxes yet.",
      );
    }
    if (params.experimentalEnableSnapshot) {
      throw new InvalidError(
        "Memory snapshots are not supported by persistent Sandboxes.",
      );
    }
    if (
      Object.keys(params.env ?? {}).length > 0 ||
      Object.keys(params.tags ?? {}).length > 0
    ) {
      throw new InvalidError(
        "`env` and `tags` are not supported by persistent Sandboxes yet.",
      );
    }

    await image.build(app);

    const secrets = params.secrets ?? [];
    const resolvable = resolvableSecrets(secrets);
    await hydrateSandboxSecrets(
      client,
      [...resolvable, ...(params.experimentalOutboundPolicy?._secrets() ?? [])],
      params.cloudBucketMounts,
    );
    // Every session is created from the same definition, so credentials must
    // live in server-side Secrets rather than inline in the request.
    const sources = secretSources(secrets);
    if (sources.some((source) => source.env !== undefined)) {
      throw new InvalidError(
        "Ephemeral credentials are not supported by persistent Sandboxes yet.",
      );
    }

    const v2Req = await buildSandboxCreateV2RequestProto(
      app.appId,
      image.imageId,
      {
        ...params,
        secrets: resolvable,
        timeoutMs: PERSISTENT_SANDBOX_SESSION_TIMEOUT_MS,
        idleTimeoutMs: PERSISTENT_SANDBOX_SESSION_IDLE_TIMEOUT_MS,
      },
      sources,
    );
    const resp = await client.cpClient.persistentSandboxCreate(
      PersistentSandboxCreateRequest.create({
        appId: v2Req.appId,
        definition: v2Req.definition,
      }),
    );
    client.logger.debug(
      "Created persistent Sandbox",
      "persistent_sandbox_id",
      resp.persistentSandboxId,
      "sandbox_id",
      resp.sandboxId,
    );
    return new SandboxImplPersistent(client, resp.sandboxId, {
      persistentSandboxId: resp.persistentSandboxId,
    });
  }

  /** The session operations are currently forwarded to. */
  get session(): SandboxImplV2 {
    return this.#session;
  }

  // Session replacement

  /** Run `op` on the session, switching to the running session and retrying once if it fails. */
  async #run<T>(op: (session: SandboxImplV2) => Promise<T>): Promise<T> {
    const session = this.#session;
    try {
      return await op(session);
    } catch (err) {
      // A terminated persistent Sandbox is stopped for good, so it isn't resumed on use.
      if (isNonRecoverableError(err) || this.#terminated) {
        throw err;
      }
      const replacement = await this.#replaceSession(session);
      if (replacement === undefined) {
        throw err;
      }
      this.client.logger.debug(
        "Retrying operation on persistent Sandbox",
        "persistent_sandbox_id",
        this.persistentSandboxId,
        "error",
        err,
      );
      return await op(replacement);
    }
  }

  async #getRunningSessionId(): Promise<string> {
    const resp = await this.client.cpClient.persistentSandboxGet(
      PersistentSandboxGetRequest.create({
        persistentSandboxId: this.persistentSandboxId,
      }),
    );
    return resp.sandboxId;
  }

  /** Resume the persistent Sandbox, returning the ID of its running session. */
  async #resumeSession(): Promise<string> {
    const req = PersistentSandboxResumeRequest.create({
      persistentSandboxId: this.persistentSandboxId,
    });
    const deadline = Date.now() + RESUME_RETRY_TIMEOUT_MS;
    while (true) {
      try {
        const resp = await this.client.cpClient.persistentSandboxResume(req);
        return resp.sandboxId;
      } catch (err) {
        // UNAVAILABLE means the previous session is still pausing or another
        // resume is in progress.
        if (
          !(err instanceof ClientError) ||
          err.code !== Status.UNAVAILABLE ||
          Date.now() > deadline
        ) {
          throw err;
        }
      }
      await setTimeout(RESUME_RETRY_INTERVAL_MS);
    }
  }

  /** Switch from the `failed` session, or return undefined if it is still the running session. */
  #replaceSession(failed: SandboxImplV2): Promise<SandboxImplV2 | undefined> {
    const run = this.#replaceChain.then(() => this.#doReplaceSession(failed));
    this.#replaceChain = run.catch(() => undefined);
    return run;
  }

  async #doReplaceSession(
    failed: SandboxImplV2,
  ): Promise<SandboxImplV2 | undefined> {
    if (this.#session !== failed) {
      // A concurrent operation has already replaced the session.
      return this.#session;
    }
    let sessionId = await this.#getRunningSessionId();
    if (sessionId === failed.sandboxId) {
      return undefined;
    }
    if (!sessionId) {
      sessionId = await this.#resumeSession();
    }
    this.client.logger.debug(
      "Switching persistent Sandbox to session",
      "persistent_sandbox_id",
      this.persistentSandboxId,
      "sandbox_id",
      sessionId,
    );
    const replacement = new SandboxImplV2(this.client, sessionId);
    this.#session = replacement;
    failed.close();
    return replacement;
  }

  #checkNotTerminated(): void {
    if (this.#terminated) {
      throw new InvalidError(
        `Persistent Sandbox ${this.persistentSandboxId} has been terminated.`,
      );
    }
  }

  // Operations

  stdin(): ModalWriteStream<string> {
    // Streams connect to whichever session is current when they first connect.
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

  async getTags(): Promise<Record<string, string>> {
    throw new InvalidError(
      "Tags aren't supported for persistent Sandboxes yet.",
    );
  }

  async setTags(_tags: Record<string, string>): Promise<void> {
    throw new InvalidError(
      "Tags aren't supported for persistent Sandboxes yet.",
    );
  }

  async setName(_name: string): Promise<void> {
    throw new InvalidError(
      "The name of a persistent Sandbox can't be changed.",
    );
  }

  async createConnectToken(
    port: number,
    userMetadata: string | undefined,
  ): Promise<SandboxCreateConnectCredentials> {
    return await this.#run((s) => s.createConnectToken(port, userMetadata));
  }

  async terminate(): Promise<void> {
    await this.client.cpClient.persistentSandboxTerminate(
      PersistentSandboxTerminateRequest.create({
        persistentSandboxId: this.persistentSandboxId,
      }),
    );
    this.#terminated = true;
  }

  /** Stop the current session, which saves its state for the next `resume`. */
  async pause(): Promise<void> {
    this.#checkNotTerminated();
    const session = this.#session;
    await session.terminate();
    await session.wait();
  }

  /** Start a session from the latest saved state, unless one is already running. */
  async resume(): Promise<void> {
    this.#checkNotTerminated();
    const session = this.#session;
    const sessionId = await this.#resumeSession();
    if (sessionId !== session.sandboxId) {
      this.#session = new SandboxImplV2(this.client, sessionId);
      session.close();
    }
  }

  close(): void {
    this.#session.close();
  }

  async wait(): Promise<number> {
    // A persistent Sandbox keeps running across sessions, until it is terminated.
    while (!this.#terminated) {
      await setTimeout(WAIT_POLL_INTERVAL_MS);
    }
    return getReturnCode(TERMINATED_RESULT)!;
  }

  async poll(): Promise<number | null> {
    return this.#terminated ? getReturnCode(TERMINATED_RESULT) : null;
  }

  async waitForScheduling(timeoutMs: number): Promise<void> {
    await this.#session.waitForScheduling(timeoutMs);
  }

  async tunnels(_timeoutMs: number): Promise<Record<number, Tunnel>> {
    // Tunnels are addressed by the persistent Sandbox, not a session, so they
    // are served without a running session and never go stale across sessions.
    const resp = await this.client.cpClient.persistentSandboxTunnelsList(
      PersistentSandboxTunnelsListRequest.create({
        persistentSandboxId: this.persistentSandboxId,
      }),
    );
    return tunnelsFromProtos(resp.tunnels);
  }

  async getAppId(): Promise<string> {
    return await this.#run((s) => s.getAppId());
  }

  async resolveTaskIdForLogs(): Promise<string> {
    return await this.#session.resolveTaskIdForLogs();
  }

  async getExitSnapshot(
    params: SandboxExperimentalGetExitSnapshotParams | undefined,
  ): Promise<Image> {
    return await this.#session.getExitSnapshot(params);
  }

  async snapshotMemory(): Promise<SandboxSnapshot> {
    throw new InvalidError(
      "Memory snapshots aren't supported for persistent Sandboxes.",
    );
  }

  async getCommandRouter(
    signal?: AbortSignal,
  ): Promise<[string, TaskCommandRouterClientImpl]> {
    return await this.#run((s) => s.getCommandRouter(signal));
  }

  async waitUntilReady(timeoutMs: number): Promise<void> {
    await this.#run((s) => s.waitUntilReady(timeoutMs));
  }

  async updateNetworkPolicy(
    _params: SandboxUpdateNetworkPolicyParams,
  ): Promise<void> {
    throw new InvalidError(
      "Updating network access isn't supported for persistent Sandboxes yet.",
    );
  }

  async updateOutboundPolicy(
    _outboundPolicy: ExperimentalOutboundPolicy,
  ): Promise<void> {
    throw new InvalidError(
      "Updating the outbound policy isn't supported for persistent Sandboxes yet.",
    );
  }

  async snapshotFilesystem(
    containerId: string,
    params: SandboxSnapshotFilesystemParams | undefined,
  ): Promise<Image> {
    return await this.#run((s) => s.snapshotFilesystem(containerId, params));
  }

  async snapshotDirectory(
    path: string,
    params: SandboxSnapshotDirectoryParams | undefined,
  ): Promise<Image> {
    return await this.#run((s) => s.snapshotDirectory(path, params));
  }

  async mountImage(
    _path: string,
    _image: Image | undefined,
    _params: SandboxMountImageParams | undefined,
  ): Promise<void> {
    throw new InvalidError(
      "Mounting images isn't supported for persistent Sandboxes yet.",
    );
  }

  async unmountImage(_path: string): Promise<void> {
    throw new InvalidError(
      "Mounting images isn't supported for persistent Sandboxes yet.",
    );
  }

  async reloadVolumes(
    containerId: string,
    params: SandboxReloadVolumesParams | undefined,
  ): Promise<void> {
    await this.#run((s) => s.reloadVolumes(containerId, params));
  }

  async exec(
    command: string[],
    params: SandboxExecParams | undefined,
    containerId?: string,
  ): Promise<ContainerProcess> {
    return await this.#run((s) => s.exec(command, params, containerId));
  }

  validateSidecarCreate(options: SidecarCreateOptions): void {
    this.#session.validateSidecarCreate(options);
  }

  async createSidecar(
    request: SidecarCreateRequest,
  ): Promise<SidecarCreateResult> {
    return await this.#run((s) => s.createSidecar(request));
  }
}
