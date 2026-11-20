import { describe, expect, test, vi } from "vitest";
import { App, ExperimentalOutboundPolicy, Image, ModalClient } from "modal";

function makeClient() {
  const logger = {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  };
  const cpClient = {
    async sandboxCreateV2(_req: any) {
      return {
        sandboxId: "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV",
        taskId: "ta-123",
        tunnels: [],
      };
    },
  };
  const client = new ModalClient({
    cpClient: cpClient as any,
    tokenId: "test-id",
    tokenSecret: "test-secret",
    logger,
  });
  return { client, logger };
}

function warningsMatching(
  logger: ReturnType<typeof makeClient>["logger"],
  text: string,
) {
  return logger.warn.mock.calls
    .map(([message]) => String(message))
    .filter((message) => message.includes(text));
}

describe("Sandbox create parameters that outboundPolicy replaces", () => {
  test.each([
    [
      { outboundCidrAllowlist: ["10.0.0.0/8"] },
      "outboundCidrAllowlist is deprecated and will be removed in a future release",
    ],
    [
      { outboundDomainAllowlist: ["example.com"] },
      "outboundDomainAllowlist is deprecated and will be removed in a future release",
    ],
    [
      { experimentalOutboundPolicy: new ExperimentalOutboundPolicy() },
      "experimentalOutboundPolicy is deprecated and will be removed in version 0.12.0",
    ],
  ])("%o logs a deprecation warning once", async (params, warning) => {
    const { client, logger } = makeClient();
    const image = new Image(client, "im-123", "");
    for (let i = 0; i < 2; i++) {
      await client.sandboxes.create(new App("ap-1234"), image, params);
    }
    expect(warningsMatching(logger, warning)).toHaveLength(1);
  });
});
