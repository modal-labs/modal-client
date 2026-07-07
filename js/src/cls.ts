import { ClientError, Status } from "nice-grpc";
import {
  ClassParameterInfo_ParameterSerializationFormat,
  ClassParameterSpec,
  FunctionBindParamsResponse,
  FunctionHandleMetadata,
} from "../proto/modal_proto/api";
import { NotFoundError } from "./errors";
import { type ModalClient } from "./client";
import {
  bindParameters,
  Function_,
  FunctionOptions,
  FunctionWithBatchingParams,
  FunctionWithConcurrencyParams,
  FunctionWithOptionsParams,
  mergeServiceOptions,
} from "./function";

/** Optional parameters for {@link ClsService#fromName client.cls.fromName()}. */
export type ClsFromNameParams = {
  environment?: string;
  createIfMissing?: boolean;
  /** Look up a version-pinned Cls deployed at this App version. */
  version?: number;
};

/**
 * Service for managing {@link Cls}.
 *
 * Normally only ever accessed via the client as:
 * ```typescript
 * const modal = new ModalClient();
 * const cls = await modal.cls.fromName("my-app", "MyCls");
 * ```
 */
export class ClsService {
  readonly #client: ModalClient;
  constructor(client: ModalClient) {
    this.#client = client;
  }

  /**
   * Reference a {@link Cls} from a deployed {@link App} by its name.
   */
  async fromName(
    appName: string,
    name: string,
    params: ClsFromNameParams = {},
  ): Promise<Cls> {
    try {
      const serviceFunctionName = `${name}.*`;
      const serviceFunction = await this.#client.cpClient.functionGet({
        appName,
        objectTag: serviceFunctionName,
        environmentName: this.#client.environmentName(params.environment),
        appVersion: params.version,
      });

      const parameterInfo = serviceFunction.handleMetadata?.classParameterInfo;
      const schema = parameterInfo?.schema ?? [];
      if (
        schema.length > 0 &&
        parameterInfo?.format !==
          ClassParameterInfo_ParameterSerializationFormat.PARAM_SERIALIZATION_FORMAT_PROTO
      ) {
        throw new Error(
          `Unsupported parameter format: ${parameterInfo?.format}`,
        );
      }

      this.#client.logger.debug(
        "Retrieved Cls",
        "function_id",
        serviceFunction.functionId,
        "app_name",
        appName,
        "cls_name",
        name,
      );
      return new Cls(
        this.#client,
        serviceFunction.functionId,
        serviceFunction.handleMetadata!,
        undefined,
      );
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(`Class '${appName}/${name}' not found`);
      throw err;
    }
  }
}

export type ClsWithOptionsParams = FunctionWithOptionsParams;
export type ClsWithConcurrencyParams = FunctionWithConcurrencyParams;
export type ClsWithBatchingParams = FunctionWithBatchingParams;

/** Represents a deployed Modal Cls. */
export class Cls {
  #client: ModalClient;
  #serviceFunctionId: string;
  #serviceFunctionMetadata: FunctionHandleMetadata;
  #serviceFunctionOptions?: FunctionOptions;

  /** @ignore */
  constructor(
    client: ModalClient,
    serviceFunctionId: string,
    serviceFunctionMetadata: FunctionHandleMetadata,
    options?: FunctionOptions,
  ) {
    this.#client = client;
    this.#serviceFunctionId = serviceFunctionId;
    this.#serviceFunctionMetadata = serviceFunctionMetadata;
    this.#serviceFunctionOptions = options;
  }

  get #schema(): ClassParameterSpec[] {
    return this.#serviceFunctionMetadata.classParameterInfo?.schema ?? [];
  }

  /** Create a new instance of the Cls with parameters and/or runtime options. */
  async instance(parameters: Record<string, any> = {}): Promise<ClsInstance> {
    let functionId: string;
    let methodHandleMetadata: { [key: string]: FunctionHandleMetadata };
    if (
      this.#schema.length === 0 &&
      this.#serviceFunctionOptions === undefined
    ) {
      functionId = this.#serviceFunctionId;
      methodHandleMetadata = this.#serviceFunctionMetadata.methodHandleMetadata;
    } else {
      const bindResp = await this.#bindParameters(parameters);
      functionId = bindResp.boundFunctionId;
      // Use the bound variant's per-method metadata so dynamic options such as
      // routingRegion (surfaced as inputPlaneUrl/inputPlaneRegion) take effect
      // at invocation time.
      methodHandleMetadata =
        bindResp.handleMetadata?.methodHandleMetadata ?? {};
    }

    const methods = new Map<string, Function_>();
    for (const [name, methodMetadata] of Object.entries(methodHandleMetadata)) {
      methods.set(
        name,
        new Function_(this.#client, functionId, name, methodMetadata),
      );
    }
    return new ClsInstance(methods);
  }

  /** Override the static Function configuration at runtime. */
  withOptions(options: ClsWithOptionsParams): Cls {
    const merged = mergeServiceOptions(this.#serviceFunctionOptions, options);
    return new Cls(
      this.#client,
      this.#serviceFunctionId,
      this.#serviceFunctionMetadata,
      merged,
    );
  }

  /** Create an instance of the Cls with input concurrency enabled or overridden with new values. */
  withConcurrency(params: ClsWithConcurrencyParams): Cls {
    const merged = mergeServiceOptions(this.#serviceFunctionOptions, {
      maxConcurrentInputs: params.maxInputs,
      targetConcurrentInputs: params.targetInputs,
    });
    return new Cls(
      this.#client,
      this.#serviceFunctionId,
      this.#serviceFunctionMetadata,
      merged,
    );
  }

  /** Create an instance of the Cls with dynamic batching enabled or overridden with new values. */
  withBatching(params: ClsWithBatchingParams): Cls {
    const merged = mergeServiceOptions(this.#serviceFunctionOptions, {
      batchMaxSize: params.maxBatchSize,
      batchWaitMs: params.waitMs,
    });
    return new Cls(
      this.#client,
      this.#serviceFunctionId,
      this.#serviceFunctionMetadata,
      merged,
    );
  }

  /** Bind parameters to the Cls function. */
  async #bindParameters(
    parameters: Record<string, any>,
  ): Promise<FunctionBindParamsResponse> {
    return await bindParameters(
      this.#client,
      this.#serviceFunctionId,
      this.#serviceFunctionOptions,
      this.#schema,
      parameters,
    );
  }
}

/** Represents an instance of a deployed Modal {@link Cls}, optionally with parameters. */
export class ClsInstance {
  #methods: Map<string, Function_>;

  constructor(methods: Map<string, Function_>) {
    this.#methods = methods;
  }

  method(name: string): Function_ {
    const method = this.#methods.get(name);
    if (!method) {
      throw new NotFoundError(`Method '${name}' not found on class`);
    }
    return method;
  }
}
