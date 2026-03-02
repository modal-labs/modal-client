// Queue object, to be used with Modal Queues.

import {
  ObjectCreationType,
  QueueNextItemsRequest,
} from "../proto/modal_proto/api";
import { getDefaultClient, type ModalClient } from "./client";
import {
  InvalidError,
  NotFoundError,
  QueueEmptyError,
  QueueFullError,
} from "./errors";
import { dumps as pickleEncode, loads as pickleDecode } from "./pickle";
import { ClientError, Status } from "nice-grpc";
import { EphemeralHeartbeatManager } from "./ephemeral";
import { checkForRenamedParams } from "./validation";

const queueInitialPutBackoffMs = 100;
const queueDefaultPartitionTtlMs = 24 * 3600 * 1000; // 24 hours

/** Optional parameters for {@link QueueService#fromName client.queues.fromName()}. */
export type QueueFromNameParams = {
  environment?: string;
  createIfMissing?: boolean;
};

/** Optional parameters for {@link QueueService#delete client.queues.delete()}. */
export type QueueDeleteParams = {
  environment?: string;
  allowMissing?: boolean;
};

/** Optional parameters for {@link QueueService#ephemeral client.queues.ephemeral()}. */
export type QueueEphemeralParams = {
  environment?: string;
};

/**
 * Service for managing {@link Queue}s.
 *
 * Normally only ever accessed via the client as:
 * ```typescript
 * const modal = new ModalClient();
 * const queue = await modal.queues.fromName("my-queue");
 * ```
 */
export class QueueService {
  readonly #client: ModalClient;
  constructor(client: ModalClient) {
    this.#client = client;
  }

  /**
   * Create a nameless, temporary {@link Queue}.
   * You will need to call {@link Queue#closeEphemeral Queue.closeEphemeral()} to delete the Queue.
   */
  async ephemeral(params: QueueEphemeralParams = {}): Promise<Queue> {
    const resp = await this.#client.cpClient.queueGetOrCreate({
      objectCreationType: ObjectCreationType.OBJECT_CREATION_TYPE_EPHEMERAL,
      environmentName: this.#client.environmentName(params.environment),
    });

    this.#client.logger.debug(
      "Created ephemeral Queue",
      "queue_id",
      resp.queueId,
    );

    const ephemeralHbManager = new EphemeralHeartbeatManager(() =>
      this.#client.cpClient.queueHeartbeat({ queueId: resp.queueId }),
    );

    return new Queue(this.#client, resp.queueId, undefined, ephemeralHbManager);
  }

  /**
   * Reference a {@link Queue} by name.
   */
  async fromName(
    name: string,
    params: QueueFromNameParams = {},
  ): Promise<Queue> {
    try {
      const resp = await this.#client.cpClient.queueGetOrCreate({
        deploymentName: name,
        objectCreationType: params.createIfMissing
          ? ObjectCreationType.OBJECT_CREATION_TYPE_CREATE_IF_MISSING
          : undefined,
        environmentName: this.#client.environmentName(params.environment),
      });
      this.#client.logger.debug(
        "Retrieved Queue",
        "queue_id",
        resp.queueId,
        "queue_name",
        name,
      );
      return new Queue(this.#client, resp.queueId, name);
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(err.details);
      throw err;
    }
  }

  /**
   * Delete a {@link Queue} by name.
   *
   * Warning: Deletion is irreversible and will affect any Apps currently using the Queue.
   */
  async delete(name: string, params: QueueDeleteParams = {}): Promise<void> {
    try {
      const queue = await this.fromName(name, {
        environment: params.environment,
        createIfMissing: false,
      });
      await this.#client.cpClient.queueDelete({ queueId: queue.queueId });
      this.#client.logger.debug(
        "Deleted Queue",
        "queue_name",
        name,
        "queue_id",
        queue.queueId,
      );
    } catch (err) {
      const isNotFound =
        err instanceof NotFoundError ||
        (err instanceof ClientError && err.code === Status.NOT_FOUND);
      if (isNotFound && params.allowMissing) {
        return;
      }
      throw err;
    }
  }
}

/** Optional parameters for {@link Queue#clear Queue.clear()}. */
export type QueueClearParams = {
  /** Partition to clear, uses default partition if not set. */
  partition?: string;

  /** Set to clear all Queue partitions. */
  all?: boolean;
};

/** Optional parameters for {@link Queue#get Queue.get()}. */
export type QueueGetParams = {
  /** How long to wait if the Queue is empty in milliseconds (default: indefinite). */
  timeoutMs?: number;

  /** Partition to fetch values from, uses default partition if not set. */
  partition?: string;
};

/** Optional parameters for {@link Queue#getMany Queue.getMany()}. */
export type QueueGetManyParams = QueueGetParams;

/** Optional parameters for {@link Queue#put Queue.put()}. */
export type QueuePutParams = {
  /** How long to wait if the Queue is full in milliseconds (default: indefinite). */
  timeoutMs?: number;

  /** Partition to add items to, uses default partition if not set. */
  partition?: string;

  /** TTL for the partition in milliseconds (default: 1 day). */
  partitionTtlMs?: number;
};

/** Optional parameters for {@link Queue#putMany Queue.putMany()}. */
export type QueuePutManyParams = QueuePutParams;

/** Optional parameters for {@link Queue#len Queue.len()}. */
export type QueueLenParams = {
  /** Partition to compute length, uses default partition if not set. */
  partition?: string;

  /** Return the total length across all partitions. */
  total?: boolean;
};

/** Optional parameters for {@link Queue#iterate Queue.iterate()}. */
export type QueueIterateParams = {
  /** How long to wait between successive items before exiting iteration in milliseconds (default: 0). */
  itemPollTimeoutMs?: number;

  /** Partition to iterate, uses default partition if not set. */
  partition?: string;
};

/**
 * Distributed, FIFO queue for data flow in Modal {@link App Apps}.
 */
export class Queue {
  readonly #client: ModalClient;
  readonly queueId: string;
  readonly name?: string;
  readonly #ephemeralHbManager?: EphemeralHeartbeatManager;

  /** @ignore */
  constructor(
    client: ModalClient,
    queueId: string,
    name?: string,
    ephemeralHbManager?: EphemeralHeartbeatManager,
  ) {
    this.#client = client;
    this.queueId = queueId;
    this.name = name;
    this.#ephemeralHbManager = ephemeralHbManager;
  }

  static #validatePartitionKey(partition: string | undefined): Uint8Array {
    if (partition) {
      const partitionKey = new TextEncoder().encode(partition);
      if (partitionKey.length === 0 || partitionKey.length > 64) {
        throw new InvalidError(
          "Queue partition key must be between 1 and 64 bytes.",
        );
      }
      return partitionKey;
    }
    return new Uint8Array();
  }

  /**
   * @deprecated Use {@link QueueService#ephemeral client.queues.ephemeral()} instead.
   */
  static async ephemeral(params: QueueEphemeralParams = {}): Promise<Queue> {
    return getDefaultClient().queues.ephemeral(params);
  }

  /** Delete the ephemeral Queue. Only usable with ephemeral Queues. */
  closeEphemeral(): void {
    if (this.#ephemeralHbManager) {
      this.#ephemeralHbManager.stop();
    } else {
      throw new InvalidError("Queue is not ephemeral.");
    }
  }

  /**
   * @deprecated Use {@link QueueService#fromName client.queues.fromName()} instead.
   */
  static async lookup(
    name: string,
    options: QueueFromNameParams = {},
  ): Promise<Queue> {
    return getDefaultClient().queues.fromName(name, options);
  }

  /**
   * @deprecated Use {@link QueueService#delete client.queues.delete()} instead.
   */
  static async delete(
    name: string,
    options: QueueDeleteParams = {},
  ): Promise<void> {
    return getDefaultClient().queues.delete(name, options);
  }

  /**
   * Remove all objects from a Queue partition.
   */
  async clear(params: QueueClearParams = {}): Promise<void> {
    if (params.partition && params.all) {
      throw new InvalidError(
        "Partition must be null when requesting to clear all.",
      );
    }
    await this.#client.cpClient.queueClear({
      queueId: this.queueId,
      partitionKey: Queue.#validatePartitionKey(params.partition),
      allPartitions: params.all,
    });
  }

  async #get(
    n: number,
    partition?: string,
    timeoutMs?: number,
  ): Promise<any[]> {
    const partitionKey = Queue.#validatePartitionKey(partition);

    const startTime = Date.now();
    let pollTimeoutMs = 50_000;
    if (timeoutMs !== undefined) {
      pollTimeoutMs = Math.min(pollTimeoutMs, timeoutMs);
    }

    while (true) {
      const response = await this.#client.cpClient.queueGet({
        queueId: this.queueId,
        partitionKey,
        timeout: pollTimeoutMs / 1000,
        nValues: n,
      });
      if (response.values && response.values.length > 0) {
        return response.values.map((value) => pickleDecode(value));
      }
      if (timeoutMs !== undefined) {
        const remainingMs = timeoutMs - (Date.now() - startTime);
        if (remainingMs <= 0) {
          const message = `Queue ${this.queueId} did not return values within ${timeoutMs}ms.`;
          throw new QueueEmptyError(message);
        }
        pollTimeoutMs = Math.min(pollTimeoutMs, remainingMs);
      }
    }
  }

  /**
   * Remove and return the next object from the Queue.
   *
   * By default, this will wait until at least one item is present in the Queue.
   * If `timeoutMs` is set, raises `QueueEmptyError` if no items are available
   * within that timeout in milliseconds.
   */
  async get(params: QueueGetParams = {}): Promise<any | null> {
    checkForRenamedParams(params, { timeout: "timeoutMs" });

    const values = await this.#get(1, params.partition, params.timeoutMs);
    return values[0]; // Must have length >= 1 if returned.
  }

  /**
   * Remove and return up to `n` objects from the Queue.
   *
   * By default, this will wait until at least one item is present in the Queue.
   * If `timeoutMs` is set, raises `QueueEmptyError` if no items are available
   * within that timeout in milliseconds.
   */
  async getMany(n: number, params: QueueGetManyParams = {}): Promise<any[]> {
    checkForRenamedParams(params, { timeout: "timeoutMs" });

    return await this.#get(n, params.partition, params.timeoutMs);
  }

  async #put(
    values: any[],
    timeoutMs?: number,
    partition?: string,
    partitionTtlMs?: number,
  ): Promise<void> {
    const valuesEncoded = values.map((v) => pickleEncode(v));
    const partitionKey = Queue.#validatePartitionKey(partition);

    let delay = queueInitialPutBackoffMs;
    const deadline = timeoutMs ? Date.now() + timeoutMs : undefined;
    while (true) {
      try {
        await this.#client.cpClient.queuePut({
          queueId: this.queueId,
          values: valuesEncoded,
          partitionKey,
          partitionTtlSeconds:
            (partitionTtlMs || queueDefaultPartitionTtlMs) / 1000,
        });
        break;
      } catch (e) {
        if (e instanceof ClientError && e.code === Status.RESOURCE_EXHAUSTED) {
          // Queue is full, retry with exponential backoff up to the deadline.
          delay = Math.min(delay * 2, 30_000);
          if (deadline !== undefined) {
            const remaining = deadline - Date.now();
            if (remaining <= 0)
              throw new QueueFullError(`Put failed on ${this.queueId}.`);
            delay = Math.min(delay, remaining);
          }
          await new Promise((resolve) => setTimeout(resolve, delay));
        } else {
          throw e;
        }
      }
    }
  }

  /**
   * Add an item to the end of the Queue.
   *
   * If the Queue is full, this will retry with exponential backoff until the
   * provided `timeoutMs` is reached, or indefinitely if `timeoutMs` is not set.
   * Raises {@link QueueFullError} if the Queue is still full after the timeout.
   */
  async put(v: any, params: QueuePutParams = {}): Promise<void> {
    checkForRenamedParams(params, {
      timeout: "timeoutMs",
      partitionTtl: "partitionTtlMs",
    });

    await this.#put(
      [v],
      params.timeoutMs,
      params.partition,
      params.partitionTtlMs,
    );
  }

  /**
   * Add several items to the end of the Queue.
   *
   * If the Queue is full, this will retry with exponential backoff until the
   * provided `timeoutMs` is reached, or indefinitely if `timeoutMs` is not set.
   * Raises {@link QueueFullError} if the Queue is still full after the timeout.
   */
  async putMany(values: any[], params: QueuePutManyParams = {}): Promise<void> {
    checkForRenamedParams(params, {
      timeout: "timeoutMs",
      partitionTtl: "partitionTtlMs",
    });

    await this.#put(
      values,
      params.timeoutMs,
      params.partition,
      params.partitionTtlMs,
    );
  }

  /** Return the number of objects in the Queue. */
  async len(params: QueueLenParams = {}): Promise<number> {
    if (params.partition && params.total) {
      throw new InvalidError(
        "Partition must be null when requesting total length.",
      );
    }
    const resp = await this.#client.cpClient.queueLen({
      queueId: this.queueId,
      partitionKey: Queue.#validatePartitionKey(params.partition),
      total: params.total,
    });
    return resp.len;
  }

  /** Iterate through items in a Queue without mutation. */
  async *iterate(
    params: QueueIterateParams = {},
  ): AsyncGenerator<any, void, unknown> {
    checkForRenamedParams(params, { itemPollTimeout: "itemPollTimeoutMs" });

    const { partition, itemPollTimeoutMs = 0 } = params;

    let lastEntryId = undefined;
    const validatedPartitionKey = Queue.#validatePartitionKey(partition);
    let fetchDeadline = Date.now() + itemPollTimeoutMs;

    const maxPollDurationMs = 30_000;
    while (true) {
      const pollDurationMs = Math.max(
        0.0,
        Math.min(maxPollDurationMs, fetchDeadline - Date.now()),
      );
      const request: QueueNextItemsRequest = {
        queueId: this.queueId,
        partitionKey: validatedPartitionKey,
        itemPollTimeout: pollDurationMs / 1000,
        lastEntryId: lastEntryId || "",
      };

      const response = await this.#client.cpClient.queueNextItems(request);
      if (response.items && response.items.length > 0) {
        for (const item of response.items) {
          yield pickleDecode(item.value);
          lastEntryId = item.entryId;
        }
        fetchDeadline = Date.now() + itemPollTimeoutMs;
      } else if (Date.now() > fetchDeadline) {
        break;
      }
    }
  }
}
