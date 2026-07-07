// Function calls and invocations, to be used with Modal Functions.

import {
  ClassParameterSet,
  ClassParameterSpec,
  ClassParameterValue,
  DataFormat,
  FunctionBindParamsResponse,
  FunctionCallInvocationType,
  FunctionHandleMetadata,
  FunctionInput,
  FunctionOptions as FunctionOptionsProto,
  FunctionRetryPolicy,
  ParameterType,
  VolumeMount,
} from "../proto/modal_proto/api";
import { type ModalClient } from "./client";
import { FunctionCall } from "./function_call";
import { InternalFailure, InvalidError, NotFoundError } from "./errors";
import { cborEncode } from "./serialization";
import { ClientError, Status } from "nice-grpc";
import {
  ControlPlaneInvocation,
  InputPlaneInvocation,
  Invocation,
} from "./invocation";
import { checkForRenamedParams } from "./validation";
import { blobUpload } from "./blob";
import { mergeEnvIntoSecrets, hydrateSecrets, Secret } from "./secret";
import { Volume, volumeToMountProto } from "./volume";
import { parseRetries, Retries } from "./retries";
import { parseGpuConfig } from "./app";

// From: modal/_utils/blob_utils.py
const maxObjectSizeBytes = 2 * 1024 * 1024; // 2 MiB

// From: client/modal/_functions.py
const maxSystemRetries = 8;

/** Optional parameters for `client.functions.fromName()`. */
export type FunctionFromNameParams = {
  environment?: string;
  createIfMissing?: boolean;
};

/**
 * Service for managing {@link Function_ Function}s.
 *
 * Normally only ever accessed via the client as:
 * ```typescript
 * const modal = new ModalClient();
 * const function = await modal.functions.fromName("my-app", "my-function");
 * ```
 */
export class FunctionService {
  readonly #client: ModalClient;
  constructor(client: ModalClient) {
    this.#client = client;
  }

  /**
   * Reference a {@link Function_ Function} by its name in an App.
   */
  async fromName(
    appName: string,
    name: string,
    params: FunctionFromNameParams = {},
  ): Promise<Function_> {
    if (name.includes(".")) {
      const [clsName, methodName] = name.split(".", 2);
      throw new Error(
        `Cannot retrieve Cls methods using 'functions.fromName()'. Use:\n  const cls = await client.cls.fromName("${appName}", "${clsName}");\n  const instance = await cls.instance();\n  const m = instance.method("${methodName}");`,
      );
    }
    try {
      const resp = await this.#client.cpClient.functionGet({
        appName,
        objectTag: name,
        environmentName: this.#client.environmentName(params.environment),
      });
      this.#client.logger.debug(
        "Retrieved Function",
        "function_id",
        resp.functionId,
        "app_name",
        appName,
        "function_name",
        name,
      );
      return new Function_(
        this.#client,
        resp.functionId,
        undefined,
        resp.handleMetadata,
      );
    } catch (err) {
      if (err instanceof ClientError && err.code === Status.NOT_FOUND)
        throw new NotFoundError(`Function '${appName}/${name}' not found`);
      throw err;
    }
  }
}

/** Simple data structure storing stats for a running {@link Function_ Function}. */
export interface FunctionStats {
  backlog: number;
  numTotalRunners: number;
}

/** Optional parameters for {@link Function_#updateAutoscaler Function_.updateAutoscaler()}. */
export interface FunctionUpdateAutoscalerParams {
  minContainers?: number;
  maxContainers?: number;
  bufferContainers?: number;
  scaledownWindowMs?: number;
}

/** Configuration options for {@link Function_#withOptions Function_.withOptions()}. */
export type FunctionWithOptionsParams = {
  cpu?: number;
  cpuLimit?: number;
  memoryMiB?: number;
  memoryLimitMiB?: number;
  gpu?: string;
  env?: Record<string, string>;
  secrets?: Secret[];
  volumes?: Record<string, Volume>;
  retries?: number | Retries;
  maxContainers?: number;
  bufferContainers?: number;
  scaledownWindowMs?: number;
  timeoutMs?: number;
};

/** Configuration options for {@link Function_#withConcurrency Function_.withConcurrency()}. */
export type FunctionWithConcurrencyParams = {
  maxInputs: number;
  targetInputs?: number;
};

/** Configuration options for {@link Function_#withBatching Function_.withBatching()}. */
export type FunctionWithBatchingParams = {
  maxBatchSize: number;
  waitMs: number;
};

/** Internal data structure holding configuration options for {@link Function_ Function}. */
export type FunctionOptions = FunctionWithOptionsParams & {
  maxConcurrentInputs?: number;
  targetConcurrentInputs?: number;
  batchMaxSize?: number;
  batchWaitMs?: number;
};

/**
 * @internal
 * @hidden
 */
export function mergeServiceOptions(
  base: FunctionOptions | undefined,
  diff: Partial<FunctionOptions>,
): FunctionOptions | undefined {
  const filteredDiff = Object.fromEntries(
    Object.entries(diff).filter(([, value]) => value !== undefined),
  ) as Partial<FunctionOptions>;
  const merged = { ...(base ?? {}), ...filteredDiff } as FunctionOptions;
  return Object.keys(merged).length === 0 ? undefined : merged;
}

/**
 * @internal
 * @hidden
 */
export async function buildFunctionOptionsProto(
  options?: FunctionOptions,
): Promise<FunctionOptionsProto | undefined> {
  if (!options) return undefined;
  const o = options ?? {};

  checkForRenamedParams(o, {
    memory: "memoryMiB",
    memoryLimit: "memoryLimitMiB",
    scaledownWindow: "scaledownWindowMs",
    timeout: "timeoutMs",
  });

  const gpuConfig = parseGpuConfig(o.gpu);

  let milliCpu: number | undefined = undefined;
  let milliCpuMax: number | undefined = undefined;
  if (o.cpu === undefined && o.cpuLimit !== undefined) {
    throw new Error("must also specify cpu when cpuLimit is specified");
  }
  if (o.cpu !== undefined) {
    if (o.cpu <= 0) {
      throw new Error(`cpu (${o.cpu}) must be a positive number`);
    }
    milliCpu = Math.trunc(1000 * o.cpu);
    if (o.cpuLimit !== undefined) {
      if (o.cpuLimit < o.cpu) {
        throw new Error(
          `cpu (${o.cpu}) cannot be higher than cpuLimit (${o.cpuLimit})`,
        );
      }
      milliCpuMax = Math.trunc(1000 * o.cpuLimit);
    }
  }

  let memoryMb: number | undefined = undefined;
  let memoryMbMax: number | undefined = undefined;
  if (o.memoryMiB === undefined && o.memoryLimitMiB !== undefined) {
    throw new Error(
      "must also specify memoryMiB when memoryLimitMiB is specified",
    );
  }
  if (o.memoryMiB !== undefined) {
    if (o.memoryMiB <= 0) {
      throw new Error(`memoryMiB (${o.memoryMiB}) must be a positive number`);
    }
    memoryMb = o.memoryMiB;
    if (o.memoryLimitMiB !== undefined) {
      if (o.memoryLimitMiB < o.memoryMiB) {
        throw new Error(
          `memoryMiB (${o.memoryMiB}) cannot be higher than memoryLimitMiB (${o.memoryLimitMiB})`,
        );
      }
      memoryMbMax = o.memoryLimitMiB;
    }
  }

  const resources =
    milliCpu !== undefined ||
    milliCpuMax !== undefined ||
    memoryMb !== undefined ||
    memoryMbMax !== undefined ||
    gpuConfig
      ? {
          milliCpu,
          milliCpuMax,
          memoryMb,
          memoryMbMax,
          gpuConfig,
        }
      : undefined;

  const secretIds = (o.secrets || []).map((s) => s.secretId);

  const volumeMounts: VolumeMount[] = o.volumes
    ? Object.entries(o.volumes).map(([mountPath, volume]) =>
        volumeToMountProto(mountPath, volume),
      )
    : [];

  const parsedRetries = parseRetries(o.retries);
  const retryPolicy: FunctionRetryPolicy | undefined = parsedRetries
    ? {
        retries: parsedRetries.maxRetries,
        backoffCoefficient: parsedRetries.backoffCoefficient,
        initialDelayMs: parsedRetries.initialDelayMs,
        maxDelayMs: parsedRetries.maxDelayMs,
      }
    : undefined;

  if (o.scaledownWindowMs !== undefined && o.scaledownWindowMs % 1000 !== 0) {
    throw new Error(
      `scaledownWindowMs must be a multiple of 1000ms, got ${o.scaledownWindowMs}`,
    );
  }
  if (o.timeoutMs !== undefined && o.timeoutMs % 1000 !== 0) {
    throw new Error(
      `timeoutMs must be a multiple of 1000ms, got ${o.timeoutMs}`,
    );
  }

  const functionOptions = FunctionOptionsProto.create({
    secretIds,
    replaceSecretIds: secretIds.length > 0,
    replaceVolumeMounts: volumeMounts.length > 0,
    volumeMounts,
    resources,
    retryPolicy,
    concurrencyLimit: o.maxContainers,
    bufferContainers: o.bufferContainers,
    taskIdleTimeoutSecs:
      o.scaledownWindowMs !== undefined
        ? o.scaledownWindowMs / 1000
        : undefined,
    timeoutSecs: o.timeoutMs !== undefined ? o.timeoutMs / 1000 : undefined,
    maxConcurrentInputs: o.maxConcurrentInputs,
    targetConcurrentInputs: o.targetConcurrentInputs,
    batchMaxSize: o.batchMaxSize,
    batchLingerMs: o.batchWaitMs,
  });

  return functionOptions;
}

/**
 * @internal
 * @hidden
 */
export function encodeParameterSet(
  schema: ClassParameterSpec[],
  params: Record<string, any>,
): Uint8Array {
  const encoded: ClassParameterValue[] = [];
  for (const paramSpec of schema) {
    const paramValue = encodeParameter(paramSpec, params[paramSpec.name]);
    encoded.push(paramValue);
  }
  // Sort keys, identical to Python `SerializeToString(deterministic=True)`.
  encoded.sort((a, b) => a.name.localeCompare(b.name));
  return ClassParameterSet.encode({ parameters: encoded }).finish();
}

function encodeParameter(
  paramSpec: ClassParameterSpec,
  value: any,
): ClassParameterValue {
  const name = paramSpec.name;
  const paramType = paramSpec.type;
  const paramValue: ClassParameterValue = { name, type: paramType };

  switch (paramType) {
    case ParameterType.PARAM_TYPE_STRING:
      if (value == null && paramSpec.hasDefault) {
        value = paramSpec.stringDefault ?? "";
      }
      if (typeof value !== "string") {
        throw new Error(`Parameter '${name}' must be a string`);
      }
      paramValue.stringValue = value;
      break;

    case ParameterType.PARAM_TYPE_INT:
      if (value == null && paramSpec.hasDefault) {
        value = paramSpec.intDefault ?? 0;
      }
      if (typeof value !== "number") {
        throw new Error(`Parameter '${name}' must be an integer`);
      }
      paramValue.intValue = value;
      break;

    case ParameterType.PARAM_TYPE_BOOL:
      if (value == null && paramSpec.hasDefault) {
        value = paramSpec.boolDefault ?? false;
      }
      if (typeof value !== "boolean") {
        throw new Error(`Parameter '${name}' must be a boolean`);
      }
      paramValue.boolValue = value;
      break;

    case ParameterType.PARAM_TYPE_BYTES:
      if (value == null && paramSpec.hasDefault) {
        value = paramSpec.bytesDefault ?? new Uint8Array();
      }
      if (!(value instanceof Uint8Array)) {
        throw new Error(`Parameter '${name}' must be a byte array`);
      }
      paramValue.bytesValue = value;
      break;

    default:
      throw new Error(`Unsupported parameter type: ${paramType}`);
  }

  return paramValue;
}

/**
 * @internal
 * @hidden
 */
export async function bindParameters(
  client: ModalClient,
  functionId: string,
  options: FunctionOptions = {},
  schema: ClassParameterSpec[] = [],
  parameters: Record<string, any> = {},
): Promise<FunctionBindParamsResponse> {
  const mergedSecrets = await mergeEnvIntoSecrets(
    client,
    options?.env,
    options?.secrets,
  );

  // FunctionOptions only carries secret IDs, so any locally-created Secrets
  // (and env vars) must be hydrated into server-side Secrets first.
  await hydrateSecrets(client, mergedSecrets);

  const mergedOptions = mergeServiceOptions(options, {
    secrets: mergedSecrets,
    env: undefined, // setting env to undefined just to clarify it's not needed anymore
  });

  const serializedParams = encodeParameterSet(schema, parameters);
  const functionOptions = await buildFunctionOptionsProto(mergedOptions);
  return await client.cpClient.functionBindParams({
    functionId,
    serializedParams,
    functionOptions,
    environmentName: client.environmentName(),
  });
}

/** Represents a deployed Modal Function, which can be invoked remotely. */
export class Function_ {
  readonly functionId: string;
  readonly methodName?: string;
  #client: ModalClient;
  #handleMetadata?: FunctionHandleMetadata;
  #options?: FunctionOptions;

  /** @ignore */
  constructor(
    client: ModalClient,
    functionId: string,
    methodName?: string,
    functionHandleMetadata?: FunctionHandleMetadata,
    options?: FunctionOptions,
  ) {
    this.functionId = functionId;
    this.methodName = methodName;

    this.#client = client;
    this.#handleMetadata = functionHandleMetadata;
    this.#options = options;
  }

  #checkNoWebUrl(fnName: string): void {
    if (this.#handleMetadata?.webUrl) {
      throw new InvalidError(
        `A webhook Function cannot be invoked for remote execution with '.${fnName}'. Invoke this Function via its web url '${this.#handleMetadata.webUrl}' instead.`,
      );
    }
  }

  /** Override the static Function configuration at runtime. */
  withOptions(options: FunctionWithOptionsParams): Function_ {
    return new Function_(
      this.#client,
      this.functionId,
      this.methodName,
      this.#handleMetadata,
      mergeServiceOptions(this.#options, options),
    );
  }

  /** Override the static Function concurrency configuration at runtime. */
  withConcurrency(params: FunctionWithConcurrencyParams): Function_ {
    return new Function_(
      this.#client,
      this.functionId,
      this.methodName,
      this.#handleMetadata,
      mergeServiceOptions(this.#options, {
        maxConcurrentInputs: params.maxInputs,
        targetConcurrentInputs: params.targetInputs,
      }),
    );
  }

  /** Override the static Function batching configuration at runtime. */
  withBatching(params: FunctionWithBatchingParams): Function_ {
    return new Function_(
      this.#client,
      this.functionId,
      this.methodName,
      this.#handleMetadata,
      mergeServiceOptions(this.#options, {
        batchMaxSize: params.maxBatchSize,
        batchWaitMs: params.waitMs,
      }),
    );
  }

  /** Create an instance of the Function with configuration specified by
   * {@link Function_#withOptions withOptions}, {@link Function_#withConcurrency withConcurrency},
   * and/or {@link Function_#withBatching withBatching}. */
  async instance(): Promise<Function_> {
    let newFnId = this.functionId;

    if (this.#options != null && Object.entries(this.#options).length > 0) {
      const boundF = await bindParameters(
        this.#client,
        this.functionId,
        this.#options,
      );

      newFnId = boundF.boundFunctionId;
    }

    return new Function_(
      this.#client,
      newFnId,
      this.methodName,
      this.#handleMetadata,
    );
  }

  // Execute a single input into a remote Function.
  async remote(
    args: any[] = [],
    kwargs: Record<string, any> = {},
  ): Promise<any> {
    this.#client.logger.debug(
      "Executing function call",
      "function_id",
      this.functionId,
    );
    this.#checkNoWebUrl("remote");
    const input = await this.#createInput(args, kwargs);
    const invocation = await this.#createRemoteInvocation(input);
    // TODO(ryan): Add tests for retries.
    let retryCount = 0;
    while (true) {
      try {
        const result = await invocation.awaitOutput();
        this.#client.logger.debug(
          "Function call completed",
          "function_id",
          this.functionId,
        );
        return result;
      } catch (err) {
        if (err instanceof InternalFailure && retryCount <= maxSystemRetries) {
          this.#client.logger.debug(
            "Retrying function call due to internal failure",
            "function_id",
            this.functionId,
            "retry_count",
            retryCount,
          );
          await invocation.retry(retryCount);
          retryCount++;
        } else {
          throw err;
        }
      }
    }
  }

  async #createRemoteInvocation(input: FunctionInput): Promise<Invocation> {
    if (this.#handleMetadata?.inputPlaneUrl) {
      return await InputPlaneInvocation.create(
        this.#client,
        this.#handleMetadata.inputPlaneUrl,
        this.functionId,
        input,
      );
    }

    return await ControlPlaneInvocation.create(
      this.#client,
      this.functionId,
      input,
      FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_SYNC,
    );
  }

  // Spawn a single input into a remote Function.
  async spawn(
    args: any[] = [],
    kwargs: Record<string, any> = {},
  ): Promise<FunctionCall> {
    this.#client.logger.debug(
      "Spawning function call",
      "function_id",
      this.functionId,
    );
    this.#checkNoWebUrl("spawn");
    const input = await this.#createInput(args, kwargs);
    const invocation = await ControlPlaneInvocation.create(
      this.#client,
      this.functionId,
      input,
      FunctionCallInvocationType.FUNCTION_CALL_INVOCATION_TYPE_ASYNC,
    );
    this.#client.logger.debug(
      "Function call spawned",
      "function_id",
      this.functionId,
      "function_call_id",
      invocation.functionCallId,
    );
    return new FunctionCall(this.#client, invocation.functionCallId);
  }

  // Returns statistics about the Function.
  async getCurrentStats(): Promise<FunctionStats> {
    const resp = await this.#client.cpClient.functionGetCurrentStats(
      { functionId: this.functionId },
      { timeoutMs: 10000 },
    );
    return {
      backlog: resp.backlog,
      numTotalRunners: resp.numTotalTasks,
    };
  }

  // Overrides the current autoscaler behavior for this Function.
  async updateAutoscaler(
    params: FunctionUpdateAutoscalerParams,
  ): Promise<void> {
    checkForRenamedParams(params, { scaledownWindow: "scaledownWindowMs" });

    await this.#client.cpClient.functionUpdateSchedulingParams({
      functionId: this.functionId,
      warmPoolSizeOverride: 0, // Deprecated field, always set to 0
      settings: {
        minContainers: params.minContainers,
        maxContainers: params.maxContainers,
        bufferContainers: params.bufferContainers,
        scaledownWindow:
          params.scaledownWindowMs !== undefined
            ? Math.trunc(params.scaledownWindowMs / 1000)
            : undefined,
      },
    });
  }

  /**
   * URL for addressing the Web Function via HTTP.
   * @returns The web URL, or undefined if this is not a Web Function
   */
  async getWebUrl(): Promise<string | undefined> {
    return this.#handleMetadata?.webUrl || undefined;
  }

  async #createInput(
    args: any[] = [],
    kwargs: Record<string, any> = {},
  ): Promise<FunctionInput> {
    const supported_input_formats = this.#handleMetadata?.supportedInputFormats
      ?.length
      ? this.#handleMetadata.supportedInputFormats
      : [DataFormat.DATA_FORMAT_PICKLE];
    if (!supported_input_formats.includes(DataFormat.DATA_FORMAT_CBOR)) {
      // the remote function isn't cbor compatible for inputs
      // so we can error early
      throw new InvalidError(
        "cannot call Modal Function from JS SDK since it was deployed with an incompatible Python SDK version. Redeploy with Modal Python SDK >= 1.2",
      );
    }
    const payload = cborEncode([args, kwargs]);

    let argsBlobId: string | undefined = undefined;
    if (payload.length > maxObjectSizeBytes) {
      argsBlobId = await blobUpload(this.#client.cpClient, payload);
    }

    // Single input sync invocation
    return {
      args: argsBlobId ? undefined : payload,
      argsBlobId,
      dataFormat: DataFormat.DATA_FORMAT_CBOR,
      methodName: this.methodName,
      finalInput: false, // This field isn't specified in the Python client, so it defaults to false.
    };
  }
}
