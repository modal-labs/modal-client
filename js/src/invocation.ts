import {
  DataFormat,
  FunctionCallInvocationType,
  FunctionCallType,
  FunctionGetOutputsItem,
  FunctionInput,
  FunctionPutInputsItem,
  FunctionRetryInputsItem,
  GeneratorDone,
  GenericResult,
  GenericResult_GenericStatus,
} from "../proto/modal_proto/api";
import { ModalGrpcClient, type ModalClient } from "./client";
import { FunctionTimeoutError, InternalFailure, RemoteError } from "./errors";
import { cborDecode } from "./serialization";

// From: modal-client/modal/_utils/function_utils.py
const outputsTimeoutMs = 55 * 1000;

/**
 * This abstraction exists so that we can easily send inputs to either the control plane or the input plane.
 * For the control plane, we call the FunctionMap, FunctionRetryInputs, and FunctionGetOutputs RPCs.
 * For the input plane, we call the AttemptStart, AttemptRetry, and AttemptAwait RPCs.
 * For now, we support just the control plane, and will add support for the input plane soon.
 */
export interface Invocation {
  awaitOutput(timeoutMs?: number): Promise<any>;
  retry(retryCount: number): Promise<void>;
}

/**
 * Implementation of Invocation which sends inputs to the control plane.
 */
export class ControlPlaneInvocation implements Invocation {
  private readonly cpClient: ModalGrpcClient;
  readonly functionCallId: string;
  private readonly input?: FunctionInput;
  private readonly functionCallJwt?: string;
  private inputJwt?: string;

  private constructor(
    cpClient: ModalGrpcClient,
    functionCallId: string,
    input?: FunctionInput,
    functionCallJwt?: string,
    inputJwt?: string,
  ) {
    this.cpClient = cpClient;
    this.functionCallId = functionCallId;
    this.input = input;
    this.functionCallJwt = functionCallJwt;
    this.inputJwt = inputJwt;
  }

  static async create(
    client: ModalClient,
    functionId: string,
    input: FunctionInput,
    invocationType: FunctionCallInvocationType,
  ) {
    const functionPutInputsItem = FunctionPutInputsItem.create({
      idx: 0,
      input,
    });

    const functionMapResponse = await client.cpClient.functionMap({
      functionId,
      functionCallType: FunctionCallType.FUNCTION_CALL_TYPE_UNARY,
      functionCallInvocationType: invocationType,
      pipelinedInputs: [functionPutInputsItem],
    });

    return new ControlPlaneInvocation(
      client.cpClient,
      functionMapResponse.functionCallId,
      input,
      functionMapResponse.functionCallJwt,
      functionMapResponse.pipelinedInputs[0].inputJwt,
    );
  }

  static fromFunctionCallId(client: ModalClient, functionCallId: string) {
    return new ControlPlaneInvocation(client.cpClient, functionCallId);
  }

  async awaitOutput(timeoutMs?: number): Promise<any> {
    return await pollFunctionOutput(
      this.cpClient,
      (timeoutMs: number) => this.#getOutput(timeoutMs),
      timeoutMs,
    );
  }

  async #getOutput(
    timeoutMs: number,
  ): Promise<FunctionGetOutputsItem | undefined> {
    const response = await this.cpClient.functionGetOutputs({
      functionCallId: this.functionCallId,
      maxValues: 1,
      timeout: timeoutMs / 1000,
      lastEntryId: "0-0",
      clearOnSuccess: true,
      requestedAt: timeNowSeconds(),
    });
    return response.outputs ? response.outputs[0] : undefined;
  }

  async retry(retryCount: number): Promise<void> {
    // we do not expect this to happen
    if (!this.input) {
      throw new Error("Cannot retry Function invocation - input missing");
    }

    const retryItem: FunctionRetryInputsItem = {
      inputJwt: this.inputJwt!,
      input: this.input,
      retryCount,
    };

    const functionRetryResponse = await this.cpClient.functionRetryInputs({
      functionCallJwt: this.functionCallJwt,
      inputs: [retryItem],
    });
    this.inputJwt = functionRetryResponse.inputJwts[0];
  }
}

/**
 * Implementation of Invocation which sends inputs to the input plane.
 */
export class InputPlaneInvocation implements Invocation {
  private readonly cpClient: ModalGrpcClient;
  private readonly ipClient: ModalGrpcClient;
  private readonly functionId: string;
  private readonly input: FunctionPutInputsItem;
  private attemptToken: string;

  constructor(
    cpClient: ModalGrpcClient,
    ipClient: ModalGrpcClient,
    functionId: string,
    input: FunctionPutInputsItem,
    attemptToken: string,
  ) {
    this.cpClient = cpClient;
    this.ipClient = ipClient;
    this.functionId = functionId;
    this.input = input;
    this.attemptToken = attemptToken;
  }

  static async create(
    client: ModalClient,
    inputPlaneUrl: string,
    functionId: string,
    input: FunctionInput,
  ) {
    const functionPutInputsItem = FunctionPutInputsItem.create({
      idx: 0,
      input,
    });
    const ipClient = client.ipClient(inputPlaneUrl);
    // Single input sync invocation
    const attemptStartResponse = await ipClient.attemptStart({
      functionId,
      input: functionPutInputsItem,
    });
    return new InputPlaneInvocation(
      client.cpClient,
      ipClient,
      functionId,
      functionPutInputsItem,
      attemptStartResponse.attemptToken,
    );
  }

  async awaitOutput(timeoutMs?: number): Promise<any> {
    return await pollFunctionOutput(
      this.cpClient,
      (timeoutMs: number) => this.#getOutput(timeoutMs),
      timeoutMs,
    );
  }

  async #getOutput(
    timeoutMs: number,
  ): Promise<FunctionGetOutputsItem | undefined> {
    const response = await this.ipClient.attemptAwait({
      attemptToken: this.attemptToken,
      requestedAt: timeNowSeconds(),
      timeoutSecs: timeoutMs / 1000,
    });
    return response.output;
  }

  async retry(_retryCount: number): Promise<void> {
    const attemptRetryResponse = await this.ipClient.attemptRetry({
      functionId: this.functionId,
      input: this.input,
      attemptToken: this.attemptToken,
    });
    this.attemptToken = attemptRetryResponse.attemptToken;
  }
}

function timeNowSeconds() {
  return Date.now() / 1e3;
}

/**
 * Signature of a function that fetches a single output using the given timeout. Used by `pollForOutputs` to fetch
 * from either the control plane or the input plane, depending on the implementation.
 */
type GetOutput = (
  timeoutMs: number,
) => Promise<FunctionGetOutputsItem | undefined>;

/***
 * Repeatedly tries to fetch an output using the provided `getOutput` function, and the specified timeout value.
 * We use a timeout value of 55 seconds if the caller does not specify a timeout value, or if the specified timeout
 * value is greater than 55 seconds.
 */
async function pollFunctionOutput(
  cpClient: ModalGrpcClient,
  getOutput: GetOutput,
  timeoutMs?: number,
): Promise<any> {
  const startTime = Date.now();
  let pollTimeoutMs = outputsTimeoutMs;
  if (timeoutMs !== undefined) {
    pollTimeoutMs = Math.min(timeoutMs, outputsTimeoutMs);
  }

  while (true) {
    const output = await getOutput(pollTimeoutMs);
    if (output) {
      return await processResult(cpClient, output.result, output.dataFormat);
    }

    if (timeoutMs !== undefined) {
      const remainingMs = timeoutMs - (Date.now() - startTime);
      if (remainingMs <= 0) {
        const message = `Timeout exceeded: ${timeoutMs}ms`;
        throw new FunctionTimeoutError(message);
      }
      pollTimeoutMs = Math.min(outputsTimeoutMs, remainingMs);
    }
  }
}

async function processResult(
  cpClient: ModalGrpcClient,
  result: GenericResult | undefined,
  dataFormat: DataFormat,
): Promise<unknown> {
  if (!result) {
    throw new Error("Received null result from invocation");
  }

  let data = new Uint8Array();
  if (result.data !== undefined) {
    data = result.data;
  } else if (result.dataBlobId) {
    data = await blobDownload(cpClient, result.dataBlobId);
  }

  switch (result.status) {
    case GenericResult_GenericStatus.GENERIC_STATUS_TIMEOUT:
      throw new FunctionTimeoutError(`Timeout: ${result.exception}`);
    case GenericResult_GenericStatus.GENERIC_STATUS_INTERNAL_FAILURE:
      throw new InternalFailure(`Internal failure: ${result.exception}`);
    case GenericResult_GenericStatus.GENERIC_STATUS_SUCCESS:
      // Proceed to deserialize the data.
      break;
    default:
      // Handle other statuses, e.g., remote error.
      throw new RemoteError(`Remote error: ${result.exception}`);
  }

  return deserializeDataFormat(data, dataFormat);
}

async function blobDownload(
  cpClient: ModalGrpcClient,
  blobId: string,
): Promise<Uint8Array> {
  const resp = await cpClient.blobGet({ blobId });
  const s3resp = await fetch(resp.downloadUrl);
  if (!s3resp.ok) {
    throw new Error(`Failed to download blob: ${s3resp.statusText}`);
  }
  const buf = await s3resp.arrayBuffer();
  return new Uint8Array(buf);
}

function deserializeDataFormat(
  data: Uint8Array | undefined,
  dataFormat: DataFormat,
): unknown {
  if (!data) {
    return null; // No data to deserialize.
  }

  switch (dataFormat) {
    case DataFormat.DATA_FORMAT_PICKLE:
      throw new Error(
        "PICKLE output format is not supported - remote function must return CBOR format",
      );
    case DataFormat.DATA_FORMAT_CBOR:
      return cborDecode(data);
    case DataFormat.DATA_FORMAT_ASGI:
      throw new Error("ASGI data format is not supported in modal-js");
    case DataFormat.DATA_FORMAT_GENERATOR_DONE:
      return GeneratorDone.decode(data);
    default:
      throw new Error(`Unsupported data format: ${dataFormat}`);
  }
}
