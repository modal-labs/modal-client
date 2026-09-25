import type { Metadata } from "nice-grpc";
import { ModalClient } from "../src/client";

export class MockGrpcClient {
  // Map of short RPC name -> FIFO queue of handlers
  private readonly methodHandlerQueues: Map<string, Array<MockHandler>> =
    new Map();

  constructor() {
    return new Proxy(this, {
      get(target, propKey) {
        if (typeof propKey === "string" && !(propKey in target)) {
          return (actualRequest: unknown, options?: MockCallOptions) =>
            target.dispatch(propKey, actualRequest, options);
        }
        return (target as any)[propKey];
      },
    });
  }

  private readonly dispatch = async (
    methodKey: string,
    actualRequest: unknown,
    options?: MockCallOptions,
  ): Promise<unknown> => {
    const queue = this.methodHandlerQueues.get(methodKey) ?? [];
    if (queue.length === 0) {
      throw new Error(
        `Unexpected gRPC call: ${methodKey} with request ${formatValue(actualRequest)}`,
      );
    }
    const handler = queue.shift()!;
    try {
      const response = await handler(actualRequest, options);
      return structuredClone(response);
    } catch (err) {
      // A mock error can carry a `trailer` for callers that read trailing
      // metadata (e.g. grpc-status-details-bin), as the real transport would.
      const trailer = (err as { trailer?: Metadata }).trailer;
      if (trailer != null) options?.onTrailer?.(trailer);
      throw err;
    }
  };

  handleUnary(rpcName: string, handler: MockHandler) {
    const methodKey = rpcToClientMethodName(shortName(rpcName));
    const queue = this.methodHandlerQueues.get(methodKey) ?? [];
    queue.push(handler);
    this.methodHandlerQueues.set(methodKey, queue);
  }

  assertExhausted() {
    const outstanding = Array.from(this.methodHandlerQueues.entries()).filter(
      ([, q]) => q.length > 0,
    );
    if (outstanding.length > 0) {
      const details = outstanding
        .map(([k, q]) => `- ${k}: ${q.length} expectation(s) remaining`)
        .join("\n");
      throw new Error(`Not all expected gRPC calls were made:\n${details}`);
    }
  }
}

export function createMockModalClients(): {
  mockClient: ModalClient;
  mockCpClient: MockGrpcClient;
} {
  const mockCpClient = new MockGrpcClient();
  const mockClient = new ModalClient({
    cpClient: mockCpClient as any,
    tokenId: "test-token-id",
    tokenSecret: "test-token-secret",
  });

  return { mockClient, mockCpClient };
}

type MockCallOptions = {
  onTrailer?: (trailer: Metadata) => void;
  signal?: AbortSignal;
  timeoutMs?: number;
};

type MockHandler = (
  req: unknown,
  options?: MockCallOptions,
) => unknown | Promise<unknown>;

function rpcToClientMethodName(name: string): string {
  return name.length ? name[0].toLowerCase() + name.slice(1) : name;
}

function shortName(method: string): string {
  if (method.startsWith("/")) {
    const idx = method.lastIndexOf("/");
    if (idx >= 0 && idx + 1 < method.length) {
      return method.slice(idx + 1);
    }
  }
  return method;
}

function formatValue(v: unknown): string {
  try {
    return JSON.stringify(v, undefined, 2);
  } catch {
    return String(v);
  }
}
