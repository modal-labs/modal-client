import { ClientError, Status } from "nice-grpc";
import {
  ClassParameterInfo_ParameterSerializationFormat,
  ClassParameterSet,
  ClassParameterSpec,
  ClassParameterValue,
  FunctionHandleMetadata,
  FunctionOptions,
  FunctionRetryPolicy,
  ParameterType,
  VolumeMount,
} from "../proto/modal_proto/api";
import { NotFoundError } from "./errors";
import { getDefaultClient, type ModalClient } from "./client";
import { Function_ } from "./function";
import { parseGpuConfig } from "./app";
import type { Secret } from "./secret";
import { mergeEnvIntoSecrets } from "./secret";
import { Retries, parseRetries } from "./retries";
import type { Volume } from "./volume";
import { checkForRenamedParams } from "./validation";

/** Optional parameters for {@link ClsService#fromName client.cls.fromName()}. */
export type ClsFromNameParams = {
  environment?: string;
  createIfMissing?: boolean;
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

export type ClsWithOptionsParams = {
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

export type ClsWithConcurrencyParams = {
  maxInputs: number;
  targetInputs?: number;
};

export type ClsWithBatchingParams = {
  maxBatchSize: number;
  waitMs: number;
};

type ServiceOptions = ClsWithOptionsParams & {
  maxConcurrentInputs?: number;
  targetConcurrentInputs?: number;
  batchMaxSize?: number;
  batchWaitMs?: number;
};

/** Represents a deployed Modal Cls. */
export class Cls {
  #client: ModalClient;
  #serviceFunctionId: string;
  #serviceFunctionMetadata: FunctionHandleMetadata;
  #serviceOptions?: ServiceOptions;

  /** @ignore */
  constructor(
    client: ModalClient,
    serviceFunctionId: string,
    serviceFunctionMetadata: FunctionHandleMetadata,
    options?: ServiceOptions,
  ) {
    this.#client = client;
    this.#serviceFunctionId = serviceFunctionId;
    this.#serviceFunctionMetadata = serviceFunctionMetadata;
    this.#serviceOptions = options;
  }

  get #schema(): ClassParameterSpec[] {
    return this.#serviceFunctionMetadata.classParameterInfo?.schema ?? [];
  }

  /**
   * @deprecated Use {@link ClsService#fromName client.cls.fromName()} instead.
   */
  static async lookup(
    appName: string,
    name: string,
    params: ClsFromNameParams = {},
  ): Promise<Cls> {
    return getDefaultClient().cls.fromName(appName, name, params);
  }

  /** Create a new instance of the Cls with parameters and/or runtime options. */
  async instance(parameters: Record<string, any> = {}): Promise<ClsInstance> {
    let functionId: string;
    if (this.#schema.length === 0 && this.#serviceOptions === undefined) {
      functionId = this.#serviceFunctionId;
    } else {
      functionId = await this.#bindParameters(parameters);
    }

    const methods = new Map<string, Function_>();
    for (const [name, methodMetadata] of Object.entries(
      this.#serviceFunctionMetadata.methodHandleMetadata,
    )) {
      methods.set(
        name,
        new Function_(this.#client, functionId, name, methodMetadata),
      );
    }
    return new ClsInstance(methods);
  }

  /** Override the static Function configuration at runtime. */
  withOptions(options: ClsWithOptionsParams): Cls {
    const merged = mergeServiceOptions(this.#serviceOptions, options);
    return new Cls(
      this.#client,
      this.#serviceFunctionId,
      this.#serviceFunctionMetadata,
      merged,
    );
  }

  /** Create an instance of the Cls with input concurrency enabled or overridden with new values. */
  withConcurrency(params: ClsWithConcurrencyParams): Cls {
    const merged = mergeServiceOptions(this.#serviceOptions, {
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
    const merged = mergeServiceOptions(this.#serviceOptions, {
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
  async #bindParameters(parameters: Record<string, any>): Promise<string> {
    const mergedSecrets = await mergeEnvIntoSecrets(
      this.#client,
      this.#serviceOptions?.env,
      this.#serviceOptions?.secrets,
    );
    const mergedOptions = mergeServiceOptions(this.#serviceOptions, {
      secrets: mergedSecrets,
      env: undefined, // setting env to undefined just to clarify it's not needed anymore
    });

    const serializedParams = encodeParameterSet(this.#schema, parameters);
    const functionOptions = await buildFunctionOptionsProto(mergedOptions);
    const bindResp = await this.#client.cpClient.functionBindParams({
      functionId: this.#serviceFunctionId,
      serializedParams,
      functionOptions,
      environmentName: this.#client.environmentName(),
    });
    return bindResp.boundFunctionId;
  }
}

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

function mergeServiceOptions(
  base: ServiceOptions | undefined,
  diff: Partial<ServiceOptions>,
): ServiceOptions | undefined {
  const filteredDiff = Object.fromEntries(
    Object.entries(diff).filter(([, value]) => value !== undefined),
  ) as Partial<ServiceOptions>;
  const merged = { ...(base ?? {}), ...filteredDiff } as ServiceOptions;
  return Object.keys(merged).length === 0 ? undefined : merged;
}

async function buildFunctionOptionsProto(
  options?: ServiceOptions,
): Promise<FunctionOptions | undefined> {
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
    ? Object.entries(o.volumes).map(([mountPath, volume]) => ({
        volumeId: volume.volumeId,
        mountPath,
        allowBackgroundCommits: true,
        readOnly: volume.isReadOnly,
      }))
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

  const functionOptions = FunctionOptions.create({
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
