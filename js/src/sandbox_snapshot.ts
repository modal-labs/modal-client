import { type ModalClient } from "./client";

/**
 * > Sandbox memory snapshots are in **early preview**.
 *
 * A `SandboxSnapshot` object lets you interact with a stored Sandbox snapshot that was created by calling
 * {@link Sandbox#experimentalSnapshot} on a Sandbox instance. This includes both the filesystem and memory state of
 * the original Sandbox at the time the snapshot was taken.
 */
export class SandboxSnapshot {
  readonly snapshotId: string;
  readonly #client: ModalClient;
  #isV2: boolean | undefined;

  /** @ignore */
  constructor(
    client: ModalClient,
    snapshotId: string,
    params: { isV2?: boolean } = {},
  ) {
    this.#client = client;
    this.snapshotId = snapshotId;
    this.#isV2 = params.isV2;
  }

  /**
   * @internal
   * Whether the snapshot came from a V2 sandbox. `undefined` until hydrated.
   */
  get _isV2(): boolean | undefined {
    return this.#isV2;
  }

  /** @internal */
  async _hydrate(): Promise<void> {
    if (this.#isV2 !== undefined) {
      return;
    }
    // hydration doesn't actually do much apart from validating the existance of the id
    // which is implicitly done by trying to start a sandbox from the snapshot as well
    const resp = await this.#client.cpClient.sandboxSnapshotGet({
      snapshotId: this.snapshotId,
    });
    this.#isV2 = resp.handleMetadata?.isV2 ?? false;
  }
}

/**
 * Service for managing {@link SandboxSnapshot}s.
 *
 * Normally only ever accessed via the client as:
 * ```typescript
 * const modal = new ModalClient();
 * const snapshot = await modal.sandboxSnapshots.fromId(snapshotId);
 * ```
 */
export class SandboxSnapshotService {
  readonly #client: ModalClient;
  constructor(client: ModalClient) {
    this.#client = client;
  }

  /**
   * Construct a {@link SandboxSnapshot} for an existing snapshot ID.
   *
   * @param snapshotId - Snapshot ID returned when the snapshot was created.
   */
  async fromId(snapshotId: string): Promise<SandboxSnapshot> {
    return new SandboxSnapshot(this.#client, snapshotId);
  }
}
