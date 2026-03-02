import { ModalClient } from "../src/client";

export class MockGrpcClient {
  // Map of short RPC name -> FIFO queue of handlers
  private readonly methodHandlerQueues: Map<
    string,
    Array<(req: unknown) => unknown | Promise<unknown>>
  > = new Map();

  constructor() {
    return new Proxy(this, {
      get(target, propKey) {
        if (typeof propKey === "string" && !(propKey in target)) {
          return (actualRequest: unknown) =>
            target.dispatch(propKey, actualRequest);
        }
        return (target as any)[propKey];
      },
    });
  }

  private readonly dispatch = async (
    methodKey: string,
    actualRequest: unknown,
  ): Promise<unknown> => {
    const queue = this.methodHandlerQueues.get(methodKey) ?? [];
    if (queue.length === 0) {
      throw new Error(
        `Unexpected gRPC call: ${methodKey} with request ${formatValue(actualRequest)}`,
      );
    }
    const handler = queue.shift()!;
    const response = await handler(actualRequest);
    return structuredClone(response);
  };

  handleUnary(
    rpcName: string,
    handler: (req: unknown) => unknown | Promise<unknown>,
  ) {
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
