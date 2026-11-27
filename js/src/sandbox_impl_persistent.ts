import { ClientError, Status } from "nice-grpc";
import { isRetryableGrpc } from "./client";
import { setTimeout } from "timers/promises";
import {
  FileDescriptor,
  GenericResult,
  GenericResult_GenericStatus,
  PersistentSandboxCreateConnectTokenRequest,
  PersistentSandboxCreateRequest,
  PersistentSandboxGetRequest,
  PersistentSandboxResumeRequest,
  PersistentSandboxTerminateRequest,
  PersistentSandboxTunnelsListRequest,
} from "../proto/modal_proto/api";
import type { App } from "./app";
import type { ModalClient } from "./client";
import { ClientClosedError, InvalidError, TimeoutError } from "./errors";
import type { Image } from "./image";
import type { ExperimentalOutboundPolicy } from "./experimental_outbound_policy";
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
  DRAIN_TIMEOUT_SECS_OPTION,
  getReturnCode,
  hydrateSandboxSecrets,
  warnDeprecatedNetworkParams,
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
import {
  TaskShuttingDownError,
  type TaskCommandRouterClientImpl,
} from "./task_command_router_client";

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
/**
 * How long a session lets running commands finish before it stops, by default
 * and at most.
 *
 * @internal
 * @hidden
 */
export const PERSISTENT_SANDBOX_DRAIN_TIMEOUT_MS = 60 * 1000;
/** @internal @hidden */
export const PERSISTENT_SANDBOX_MAX_DRAIN_TIMEOUT_MS = 10 * 60 * 1000;

/**
 * The drain timeout the `drain_timeout_secs` experimental option sets, in
 * seconds, or the default when it is unset.
 */
function persistentDrainTimeoutSecs(
  options: Record<string, any> | undefined,
): number {
  const value: unknown = options?.[DRAIN_TIMEOUT_SECS_OPTION];
  if (value === undefined) {
    return PERSISTENT_SANDBOX_DRAIN_TIMEOUT_MS / 1000;
  }
  const maxSecs = PERSISTENT_SANDBOX_MAX_DRAIN_TIMEOUT_MS / 1000;
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value <= 0 ||
    value > maxSecs
  ) {
    throw new InvalidError(
      `Experimental option "drain_timeout_secs" must be a whole number between 1 and ${maxSecs}, got ${String(value)}.`,
    );
  }
  return value;
}

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

function maySucceedOnNewSession(err: unknown): boolean {
  return err instanceof TaskShuttingDownError || !isNonRecoverableError(err);
}

/**
 * A failure to reach a session's command router, which comes before the
 * command is sent to it.
 */
class ExecNotSentError extends Error {
  constructor(readonly inner: unknown) {
    super(String(inner));
  }
}

/**
 * Whether a failed exec certainly didn't run its command, so it can be started
 * on another session.
 */
function execNeverRan(err: unknown): boolean {
  // Reaching the router takes no input from the caller, so only a closed
  // client is their doing.
  if (err instanceof ExecNotSentError) {
    return !(err.inner instanceof ClientClosedError);
  }
  if (err instanceof TaskShuttingDownError) return true;
  // A NOT_FOUND status means the session was gone before the command could
  // reach it. Any other failure, including a dropped connection, may come
  // after the command started.
  return err instanceof ClientError && err.code === Status.NOT_FOUND;
}

/**
 * Backs a persistent Sandbox, whose object ID is its persistent Sandbox ID.
 *
 * A missing session is resumed from saved state on the first operation that
 * needs one. Failed session operations are retried once on a replacement
 * session unless the failure is non-recoverable.
 *
 * @internal
 * @hidden
 */
export class SandboxImplPersistent implements SandboxImpl {
  readonly client: ModalClient;
  /** The persistent Sandbox ID. */
  readonly sandboxId: string;

  #session: SandboxImplV2 | undefined;
  #terminated = false;
  #closed = false;
  // Session replacements run one at a time, so concurrent failures on the same
  // session trigger a single lookup/resume.
  #replaceChain: Promise<unknown> = Promise.resolve();

  constructor(
    client: ModalClient,
    sandboxId: string,
    params: { session?: SandboxImplV2; terminated?: boolean },
  ) {
    this.client = client;
    this.sandboxId = sandboxId;
    this.#session = params.session;
    this.#terminated = params.terminated ?? false;
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
    const drainTimeoutSecs = persistentDrainTimeoutSecs(
      params.experimentalOptions,
    );
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
      [
        ...resolvable,
        // eslint-disable-next-line @typescript-eslint/no-deprecated -- implements the deprecated parameter
        ...(params.experimentalOutboundPolicy?._secrets() ?? []),
        ...(params.outboundPolicy?._secrets() ?? []),
      ],
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
        experimentalPersistent: true,
        drainTimeoutSecs,
      },
      sources,
    );
    warnDeprecatedNetworkParams(client.logger, params);
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
    return new SandboxImplPersistent(client, resp.persistentSandboxId, {
      session: new SandboxImplV2(client, resp.sandboxId),
    });
  }

  /** The session operations are currently forwarded to. */
  get session(): SandboxImplV2 | undefined {
    return this.#session;
  }

  // Session replacement

  async #currentSession(): Promise<SandboxImplV2> {
    // A terminated Sandbox keeps its last session only for reading the exit
    // snapshot and logs.
    this.#checkNotTerminated();
    if (this.#session) return this.#session;
    this.#checkOpen();
    return (await this.#replaceSession(undefined))!;
  }

  /**
   * Run `op` on the session, switching to the running session and retrying
   * once if it fails. Only failures accepted by `canRetry` are retried.
   */
  async #run<T>(
    op: (session: SandboxImplV2) => Promise<T>,
    canRetry: (err: unknown) => boolean = maySucceedOnNewSession,
  ): Promise<T> {
    const session = await this.#currentSession();
    try {
      return await op(session);
    } catch (err) {
      // A terminated persistent Sandbox is stopped for good, so it isn't resumed on use.
      if (!canRetry(err) || this.#terminated) {
        throw err;
      }
      const replacement = await this.#replaceSession(session);
      if (replacement === undefined) {
        throw err;
      }
      this.client.logger.debug(
        "Retrying operation on persistent Sandbox",
        "persistent_sandbox_id",
        this.sandboxId,
        "error",
        err,
      );
      return await op(replacement);
    }
  }

  /**
   * Look up the most recent session, which may have finished already. Returns
   * "" if there has never been one. Only termination is final: whether the
   * session is still running is learned from the session itself.
   */
  async #getCurrentSessionId(): Promise<string> {
    const resp = await this.client.cpClient.persistentSandboxGet(
      PersistentSandboxGetRequest.create({
        persistentSandboxId: this.sandboxId,
      }),
    );
    if (resp.terminatedAt !== undefined) this.#terminated = true;
    if (this.#terminated && resp.currentSandboxId !== undefined) {
      // The last session, which holds the exit snapshot and logs.
      this.#setSession(resp.currentSandboxId);
    }
    return resp.currentSandboxId ?? "";
  }

  /** Resume the persistent Sandbox, returning the ID of its running session. */
  async #resumeSession(): Promise<string> {
    const req = PersistentSandboxResumeRequest.create({
      persistentSandboxId: this.sandboxId,
    });
    const deadline = Date.now() + RESUME_RETRY_TIMEOUT_MS;
    while (true) {
      try {
        const resp = await this.client.cpClient.persistentSandboxResume(req);
        return resp.sandboxId;
      } catch (err) {
        // Resuming is idempotent, so any retryable error is retried. It is
        // UNAVAILABLE while the previous session is still pausing or another
        // resume is in progress.
        if (!isRetryableGrpc(err) || Date.now() > deadline) {
          throw err;
        }
      }
      await setTimeout(RESUME_RETRY_INTERVAL_MS);
    }
  }

  /**
   * Make `sessionId` the current session and close the session it replaces.
   *
   * Every session switch goes through here, so a replaced session never keeps
   * a connection open, and a detached Sandbox only ever holds closed sessions.
   */
  #setSession(sessionId: string): SandboxImplV2 {
    const previous = this.#session;
    if (previous?.sandboxId === sessionId) return previous;
    this.client.logger.debug(
      "Switching persistent Sandbox to session",
      "persistent_sandbox_id",
      this.sandboxId,
      "sandbox_id",
      sessionId,
    );
    const session = new SandboxImplV2(this.client, sessionId);
    this.#session = session;
    previous?.close();
    if (this.#closed) session.close();
    return session;
  }

  /** Switch to the most recent session, if there has been one, without resuming. */
  async #adoptCurrentSession(): Promise<SandboxImplV2 | undefined> {
    const sessionId = await this.#getCurrentSessionId();
    return sessionId ? this.#setSession(sessionId) : undefined;
  }

  /** Return the last known session for reading its logs or metadata, without resuming. */
  async #sessionForReads(): Promise<SandboxImplV2 | undefined> {
    if (!this.#session && !this.#terminated) await this.#adoptCurrentSession();
    return this.#session;
  }

  async #sessionForLogs(): Promise<SandboxImplV2> {
    const session = await this.#sessionForReads();
    if (!session) {
      throw new InvalidError(
        `Persistent Sandbox ${this.sandboxId} has no session to read logs from.`,
      );
    }
    return session;
  }

  /**
   * Switch from the `failed` session, or return undefined if it is still the
   * running session. With no `failed` session the handle has no session yet
   * and takes the most recent one, which may have finished: the first
   * operation on it then fails and switches again, resuming the Sandbox.
   */
  #replaceSession(
    failed: SandboxImplV2 | undefined,
  ): Promise<SandboxImplV2 | undefined> {
    const run = this.#replaceChain.then(() => this.#doReplaceSession(failed));
    this.#replaceChain = run.catch(() => undefined);
    return run;
  }

  async #doReplaceSession(
    failed: SandboxImplV2 | undefined,
  ): Promise<SandboxImplV2 | undefined> {
    if (this.#session !== failed) {
      // A concurrent operation has already replaced the session.
      return this.#session;
    }
    let sessionId = await this.#getCurrentSessionId();
    this.#checkOpen();
    this.#checkNotTerminated();
    // The lookup can't tell whether `failed` is still running. Resuming
    // returns the running session, which is `failed` itself if it never
    // stopped.
    if (!sessionId || sessionId === failed?.sandboxId) {
      sessionId = await this.#resumeSession();
      if (sessionId === failed?.sandboxId) return undefined;
    }
    return this.#setSession(sessionId);
  }

  #checkOpen(): void {
    if (this.#closed) {
      throw new ClientClosedError(
        "Unable to perform operation on a detached sandbox",
      );
    }
  }

  #checkNotTerminated(): void {
    if (this.#terminated) {
      throw new InvalidError(
        `Persistent Sandbox ${this.sandboxId} has been terminated.`,
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
    // Connect tokens are addressed by the persistent Sandbox, not a session, so
    // they keep working across sessions and are created without resuming it.
    const resp = await this.client.cpClient.persistentSandboxCreateConnectToken(
      PersistentSandboxCreateConnectTokenRequest.create({
        persistentSandboxId: this.sandboxId,
        userMetadata,
        port,
      }),
    );
    return { url: resp.url, token: resp.token };
  }

  async terminate(): Promise<void> {
    const resp = await this.client.cpClient.persistentSandboxTerminate(
      PersistentSandboxTerminateRequest.create({
        persistentSandboxId: this.sandboxId,
      }),
    );
    this.#terminated = true;
    // The last session, which holds the exit snapshot and logs.
    if (resp.lastSessionSandboxId !== undefined) {
      this.#setSession(resp.lastSessionSandboxId);
    }
  }

  /**
   * Stop the current session, which saves its state for the next `resume`.
   *
   * Does nothing if there is no session or it has already stopped. A paused
   * Sandbox is never resumed in order to pause it.
   */
  async pause(): Promise<void> {
    this.#checkNotTerminated();
    const session = await this.#adoptCurrentSession();
    this.#checkNotTerminated();
    if (!session) return;
    await session.terminate();
    await session.wait();
  }

  /** Start a session from the latest saved state, unless one is already running. */
  async resume(): Promise<void> {
    this.#checkNotTerminated();
    this.#setSession(await this.#resumeSession());
  }

  close(): void {
    this.#closed = true;
    this.#session?.close();
  }

  async wait(): Promise<number> {
    while ((await this.poll()) === null) {
      await setTimeout(WAIT_POLL_INTERVAL_MS);
    }
    return getReturnCode(TERMINATED_RESULT)!;
  }

  async poll(): Promise<number | null> {
    if (!this.#terminated) {
      await this.#adoptCurrentSession();
    }
    return this.#terminated ? getReturnCode(TERMINATED_RESULT) : null;
  }

  async waitForScheduling(timeoutMs: number): Promise<void> {
    await (await this.#currentSession()).waitForScheduling(timeoutMs);
  }

  async tunnels(_timeoutMs: number): Promise<Record<number, Tunnel>> {
    // Tunnels are addressed by the persistent Sandbox, not a session, so they
    // are served without a running session and never go stale across sessions.
    const resp = await this.client.cpClient.persistentSandboxTunnelsList(
      PersistentSandboxTunnelsListRequest.create({
        persistentSandboxId: this.sandboxId,
      }),
    );
    return tunnelsFromProtos(resp.tunnels);
  }

  // The app ID and task ID for logs are read from the last known session, so
  // reading logs never resumes a paused Sandbox.
  async getAppId(): Promise<string> {
    return await (await this.#sessionForLogs()).getAppId();
  }

  async resolveTaskIdForLogs(): Promise<string> {
    return await (await this.#sessionForLogs()).resolveTaskIdForLogs();
  }

  async getExitSnapshot(
    params: SandboxExperimentalGetExitSnapshotParams | undefined,
  ): Promise<Image> {
    // Pausing a persistent Sandbox is not an exit: only termination produces
    // its exit snapshot.
    const timeoutMs = params?.timeoutMs;
    const deadline =
      timeoutMs === undefined ? undefined : Date.now() + timeoutMs;
    while ((await this.poll()) === null) {
      if (
        deadline !== undefined &&
        Date.now() + WAIT_POLL_INTERVAL_MS > deadline
      ) {
        throw new TimeoutError(
          `Timed out waiting for exit snapshot for Sandbox ${this.sandboxId}`,
        );
      }
      await setTimeout(WAIT_POLL_INTERVAL_MS);
    }
    if (!this.#session) {
      throw new InvalidError(
        `Persistent Sandbox ${this.sandboxId} has no exit snapshot: it was terminated without a session.`,
      );
    }
    return await this.#session.getExitSnapshot(
      deadline === undefined
        ? undefined
        : { timeoutMs: Math.max(0, deadline - Date.now()) },
    );
  }

  async snapshotMemory(): Promise<SandboxSnapshot> {
    throw new InvalidError(
      "Memory snapshots aren't supported for persistent Sandboxes.",
    );
  }

  async getCommandRouter(
    signal?: AbortSignal,
  ): Promise<[string, TaskCommandRouterClientImpl]> {
    this.#checkOpen();
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

  async setOutboundPolicy(): Promise<void> {
    throw new InvalidError(
      "Setting the outbound policy isn't supported for persistent Sandboxes yet.",
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
    try {
      return await this.#run(async (s) => {
        try {
          await s.getCommandRouter();
        } catch (err) {
          throw new ExecNotSentError(err);
        }
        return await s.exec(command, params, containerId);
      }, execNeverRan);
    } catch (err) {
      throw err instanceof ExecNotSentError ? err.inner : err;
    }
  }

  validateSidecarCreate(options: SidecarCreateOptions): void {
    new SandboxImplV2(this.client, "").validateSidecarCreate(options);
  }

  async createSidecar(
    request: SidecarCreateRequest,
  ): Promise<SidecarCreateResult> {
    return await this.#run((s) => s.createSidecar(request));
  }
}
