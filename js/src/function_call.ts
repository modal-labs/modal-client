// Manage existing Function Calls (look-ups, polling for output, cancellation).

import { type ModalClient } from "./client";
import { ControlPlaneInvocation } from "./invocation";
import { checkForRenamedParams } from "./validation";

/**
 * Service for managing {@link FunctionCall}s.
 *
 * Normally only ever accessed via the client as:
 *
 * ```typescript
 * const modal = new ModalClient();
 * const functionCall = await modal.functionCalls.fromId("123");
 * ```
 */
export class FunctionCallService {
  readonly #client: ModalClient;
  constructor(client: ModalClient) {
    this.#client = client;
  }

  /**
   * Create a new {@link FunctionCall} from ID.
   */
  async fromId(functionCallId: string): Promise<FunctionCall> {
    return new FunctionCall(this.#client, functionCallId);
  }
}

/** Optional parameters for {@link FunctionCall#get FunctionCall.get()}. */
export type FunctionCallGetParams = {
  timeoutMs?: number;
};

/** Optional parameters for {@link FunctionCall#cancel FunctionCall.cancel()}. */
export type FunctionCallCancelParams = {
  terminateContainers?: boolean;
};

/**
 * Represents a Modal FunctionCall. FunctionCalls are {@link Function_ Function} invocations with
 * a given input. They can be consumed asynchronously (see {@link FunctionCall#get FunctionCall.get()}) or cancelled
 * (see {@link FunctionCall#cancel FunctionCall.cancel()}).
 */
export class FunctionCall {
  readonly functionCallId: string;
  #client: ModalClient;

  /** @ignore */
  constructor(client: ModalClient, functionCallId: string) {
    this.#client = client;
    this.functionCallId = functionCallId;
  }

  /** Get the result of a FunctionCall, optionally waiting with a timeout. */
  async get(params: FunctionCallGetParams = {}): Promise<any> {
    checkForRenamedParams(params, { timeout: "timeoutMs" });

    const invocation = ControlPlaneInvocation.fromFunctionCallId(
      this.#client,
      this.functionCallId,
    );
    return invocation.awaitOutput(params.timeoutMs);
  }

  /** Cancel a running FunctionCall. */
  async cancel(params: FunctionCallCancelParams = {}) {
    const cpClient = this.#client.cpClient;

    await cpClient.functionCallCancel({
      functionCallId: this.functionCallId,
      terminateContainers: params.terminateContainers,
    });
  }
}
