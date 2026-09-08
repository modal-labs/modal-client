import { afterEach, describe, expect, test, vi } from "vitest";
import { App, Image, InvalidError, ModalClient } from "modal";
import { ClientError, Status } from "nice-grpc";

const V1_SANDBOX_ID = "sb-nGEijt9WbBMlGrsPH9FOaC";
const V2_SANDBOX_ID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV";

afterEach(() => {
  vi.unstubAllEnvs();
});

function makeRoutingStub(
  opts: { fromNameV2Error?: ClientError; listV2Error?: ClientError } = {},
) {
  const stub = {
    v1Creates: 0,
    v2Creates: 0,
    v1Lookups: 0,
    v2Lookups: 0,
    v1Lists: 0,
    v2Lists: 0,
    listV2Req: undefined as any,
    async sandboxCreate(_req: any) {
      stub.v1Creates++;
      return { sandboxId: V1_SANDBOX_ID };
    },
    async sandboxCreateV2(_req: any) {
      stub.v2Creates++;
      return { sandboxId: V2_SANDBOX_ID, taskId: "ta-v2-123", tunnels: [] };
    },
    async sandboxGetFromName(_req: any) {
      stub.v1Lookups++;
      return { sandboxId: V1_SANDBOX_ID };
    },
    async sandboxGetFromNameV2(_req: any) {
      stub.v2Lookups++;
      if (opts.fromNameV2Error) throw opts.fromNameV2Error;
      return { sandboxId: V2_SANDBOX_ID };
    },
    async sandboxList(_req: any) {
      stub.v1Lists++;
      return { sandboxes: [] };
    },
    async sandboxListV2(req: any) {
      stub.v2Lists++;
      stub.listV2Req = req;
      if (opts.listV2Error) throw opts.listV2Error;
      return { sandboxes: [] };
    },
  };
  return stub;
}

function makeClient(stub: unknown, sandboxV2: boolean): ModalClient {
  vi.stubEnv("MODAL_SANDBOX_V2", sandboxV2 ? "1" : undefined);
  return new ModalClient({
    cpClient: stub as any,
    tokenId: "test-id",
    tokenSecret: "test-secret",
  });
}

const app = new App("ap-1234");

// The sandboxV2 profile flag (MODAL_SANDBOX_V2) routes create() calls
// onto the V2 backend.
describe("MODAL_SANDBOX_V2 routing for create", () => {
  test("routes to the V2 backend when the flag is set", async () => {
    const stub = makeRoutingStub();
    const client = makeClient(stub, true);
    const image = new Image(client, "im-123", "");

    const sb = await client.sandboxes.create(app, image);
    expect(sb.sandboxId).toBe(V2_SANDBOX_ID);
    expect(stub.v2Creates).toBe(1);
    expect(stub.v1Creates).toBe(0);
  });

  test("stays on V1 when a GPU is requested", async () => {
    const stub = makeRoutingStub();
    const client = makeClient(stub, true);
    const image = new Image(client, "im-123", "");

    const sb = await client.sandboxes.create(app, image, { gpu: "T4" });
    expect(sb.sandboxId).toBe(V1_SANDBOX_ID);
    expect(stub.v1Creates).toBe(1);
    expect(stub.v2Creates).toBe(0);
  });

  test("stays on V1 without the flag", async () => {
    const stub = makeRoutingStub();
    const client = makeClient(stub, false);
    const image = new Image(client, "im-123", "");

    const sb = await client.sandboxes.create(app, image);
    expect(sb.sandboxId).toBe(V1_SANDBOX_ID);
    expect(stub.v1Creates).toBe(1);
    expect(stub.v2Creates).toBe(0);
  });
});

describe("MODAL_SANDBOX_V2 routing for fromName", () => {
  test("returns the V2 Sandbox when one matches", async () => {
    const stub = makeRoutingStub();
    const client = makeClient(stub, true);

    const sb = await client.sandboxes.fromName("my-app", "my-sandbox");
    expect(sb.sandboxId).toBe(V2_SANDBOX_ID);
    expect(stub.v2Lookups).toBe(1);
    expect(stub.v1Lookups).toBe(0);
  });

  test("falls back to V1 when V2 has no match", async () => {
    const stub = makeRoutingStub({
      fromNameV2Error: new ClientError(
        "/modal.client.ModalClient/SandboxGetFromNameV2",
        Status.NOT_FOUND,
        "no such sandbox",
      ),
    });
    const client = makeClient(stub, true);

    const sb = await client.sandboxes.fromName("my-app", "my-sandbox");
    expect(sb.sandboxId).toBe(V1_SANDBOX_ID);
    expect(stub.v2Lookups).toBe(1);
    expect(stub.v1Lookups).toBe(1);
  });

  test("propagates V2 errors other than not-found", async () => {
    const stub = makeRoutingStub({
      fromNameV2Error: new ClientError(
        "/modal.client.ModalClient/SandboxGetFromNameV2",
        Status.INTERNAL,
        "boom",
      ),
    });
    const client = makeClient(stub, true);

    await expect(
      client.sandboxes.fromName("my-app", "my-sandbox"),
    ).rejects.toThrow();
    expect(stub.v1Lookups).toBe(0);
  });

  test("stays on V1 without the flag", async () => {
    const stub = makeRoutingStub();
    const client = makeClient(stub, false);

    const sb = await client.sandboxes.fromName("my-app", "my-sandbox");
    expect(sb.sandboxId).toBe(V1_SANDBOX_ID);
    expect(stub.v1Lookups).toBe(1);
    expect(stub.v2Lookups).toBe(0);
  });
});

describe("MODAL_SANDBOX_V2 routing for list", () => {
  test("lists through the V2 backend when the flag is set", async () => {
    const stub = makeRoutingStub();
    const client = makeClient(stub, true);

    let yielded = 0;
    for await (const _ of client.sandboxes.list({
      appId: "ap-1234",
      tags: { env: "prod" },
    })) {
      yielded++;
    }

    expect(yielded).toBe(0);
    expect(stub.v2Lists).toBe(1);
    expect(stub.v1Lists).toBe(0);
    expect(stub.listV2Req.appId).toBe("ap-1234");
    expect(stub.listV2Req.tags).toEqual([{ tagName: "env", tagValue: "prod" }]);
  });

  test("stays on V1 without the flag", async () => {
    const stub = makeRoutingStub();
    const client = makeClient(stub, false);

    let yielded = 0;
    for await (const _ of client.sandboxes.list()) {
      yielded++;
    }

    expect(yielded).toBe(0);
    expect(stub.v1Lists).toBe(1);
    expect(stub.v2Lists).toBe(0);
  });

  test("translates invalid requests to InvalidError", async () => {
    const stub = makeRoutingStub({
      listV2Error: new ClientError(
        "/modal.client.ModalClient/SandboxListV2",
        Status.INVALID_ARGUMENT,
        "invalid app ID",
      ),
    });
    const client = makeClient(stub, true);

    const listing = client.sandboxes.list({ appId: "not-an-app-id" });
    await expect(listing.next()).rejects.toThrow(InvalidError);
  });
});
