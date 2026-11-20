import { afterEach, describe, expect, test, vi } from "vitest";
import {
  App,
  ExperimentalOutboundPolicy,
  Image,
  InvalidError,
  ModalClient,
  OutboundPolicy,
  OutboundRoute,
  Proxy,
  Secret,
} from "modal";
import { OutboundPolicy_Transport } from "../proto/modal_proto/api";
import { buildSandboxCreateRequestProto } from "../src/sandbox";
import { createMockModalClients } from "../test-support/grpc_mock";

afterEach(() => {
  vi.unstubAllEnvs();
});

const anyRoute = OutboundRoute.any();

function policyOf(
  routes: OutboundRoute[],
  allowTlsDecryption: boolean = true,
): OutboundPolicy {
  return new OutboundPolicy({
    allowTlsDecryption,
    routes: [...routes, anyRoute],
  });
}

describe("OutboundRoute", () => {
  test("factories record their match", () => {
    expect(OutboundRoute.tls({ domains: ["a.com"] })._match).toEqual({
      kind: "tls",
      domains: ["a.com"],
    });
    const secret = new Secret("st-1", "s");
    expect(
      OutboundRoute.https({
        domains: ["a.com"],
        headerReplacements: [{ secret, headers: { A: "$K" } }],
      })._match,
    ).toEqual({
      kind: "https",
      domains: ["a.com"],
      headerReplacements: [{ secret, headers: { A: "$K" } }],
    });
    expect(OutboundRoute.https({ domains: ["a.com"] })._match).toEqual({
      kind: "https",
      domains: ["a.com"],
      headerReplacements: [],
    });
    expect(OutboundRoute.ip({ cidrs: ["10.0.0.0/8"] })._match).toEqual({
      kind: "ip",
      cidrs: ["10.0.0.0/8"],
    });
    expect(OutboundRoute.any()._match).toEqual({ kind: "any" });
    expect(OutboundRoute.any()._block).toBe(false);
  });

  test("factories copy their inputs", () => {
    const domains = ["a.com"];
    const headers = { A: "b" };
    const route = OutboundRoute.https({
      domains,
      headerReplacements: [{ headers }],
    });
    domains.push("b.com");
    headers.A = "changed";
    expect(route._match).toEqual({
      kind: "https",
      domains: ["a.com"],
      headerReplacements: [{ secret: undefined, headers: { A: "b" } }],
    });
  });

  test("block() returns a new route", () => {
    const blocked = anyRoute.block();
    expect(blocked).not.toBe(anyRoute);
    expect(anyRoute._block).toBe(false);
    expect(blocked._block).toBe(true);
    expect(blocked._match).toBe(anyRoute._match);
  });

  test("block() twice throws", () => {
    expect(() => OutboundRoute.any().block().block()).toThrow("already blocks");
  });

  test("only the any route supports block()", () => {
    expect(() => OutboundRoute.tls({ domains: ["a.com"] }).block()).toThrow(
      "TLS routes don't support block()",
    );
    expect(() => OutboundRoute.https({ domains: ["a.com"] }).block()).toThrow(
      "HTTPS routes don't support block()",
    );
    expect(() => OutboundRoute.ip({ cidrs: ["10.0.0.0/8"] }).block()).toThrow(
      "IP routes don't support block()",
    );
  });
});

describe("OutboundPolicy._validate", () => {
  test("accepts the documented example", () => {
    const secret = new Secret("st-1", "s");
    const policy = new OutboundPolicy({
      allowTlsDecryption: true,
      routes: [
        OutboundRoute.https({
          domains: ["*.mydomain.com", "mydomain.org"],
          headerReplacements: [
            { secret, headers: { Authorization: "Bearer $API_TOKEN" } },
          ],
        }),
        OutboundRoute.https({ domains: ["api.partner.com"] }),
        OutboundRoute.tls({ domains: ["*.example.com", "*"] }),
        OutboundRoute.ip({ cidrs: ["203.0.113.0/24"] }),
        OutboundRoute.any().block(),
      ],
    });
    policy._validate();
  });

  test("defaults allowTlsDecryption to false", () => {
    expect(new OutboundPolicy({ routes: [anyRoute] })._allowTlsDecryption).toBe(
      false,
    );
  });

  test("copies the routes array", () => {
    const routes = [anyRoute];
    const policy = new OutboundPolicy({ routes });
    routes.push(OutboundRoute.tls({ domains: ["a.com"] }));
    expect(policy._routes).toEqual([anyRoute]);
  });

  test("rejects empty routes", () => {
    expect(() => new OutboundPolicy({ routes: [] })._validate()).toThrow(
      "at least one route",
    );
  });

  test("requires exactly one any route, last", () => {
    const tls = OutboundRoute.tls({ domains: ["a.com"] });
    for (const routes of [
      [tls],
      [anyRoute, tls],
      [anyRoute, anyRoute],
      [tls, anyRoute, anyRoute],
    ]) {
      expect(() => new OutboundPolicy({ routes })._validate()).toThrow(
        "exactly one `OutboundRoute.any()` route, and it must be last",
      );
    }
    new OutboundPolicy({ routes: [anyRoute] })._validate();
  });

  test("rejects domain routes without domains", () => {
    expect(() =>
      policyOf([OutboundRoute.tls({ domains: [] })])._validate(),
    ).toThrow("TLS routes need at least one domain");
    expect(() =>
      policyOf([OutboundRoute.https({ domains: [] })])._validate(),
    ).toThrow("HTTPS routes need at least one domain");
  });

  test("rejects invalid domains", () => {
    for (const domain of ["not a domain", "example.com\n", "*.", "**"]) {
      expect(() =>
        policyOf([
          OutboundRoute.tls({ domains: ["a.com", domain] }),
        ])._validate(),
      ).toThrow("Invalid domain");
      expect(() =>
        policyOf([OutboundRoute.https({ domains: [domain] })])._validate(),
      ).toThrow("Invalid domain");
    }
  });

  test("rejects IP routes without CIDRs", () => {
    expect(() =>
      policyOf([OutboundRoute.ip({ cidrs: [] })])._validate(),
    ).toThrow("IP routes need at least one CIDR");
  });

  test("HTTPS routes require allowTlsDecryption", () => {
    const route = OutboundRoute.https({ domains: ["a.com"] });
    expect(() => policyOf([route], false)._validate()).toThrow(
      "HTTPS routes require an OutboundPolicy with `allowTlsDecryption: true`",
    );
    policyOf([route], true)._validate();
    // TLS routes don't.
    policyOf([OutboundRoute.tls({ domains: ["a.com"] })], false)._validate();
  });

  function httpsWith(
    headers: Record<string, string>,
    secret?: Secret,
  ): OutboundPolicy {
    return policyOf([
      OutboundRoute.https({
        domains: ["a.com"],
        headerReplacements: [{ secret, headers }],
      }),
    ]);
  }

  test("rejects empty headers", () => {
    expect(() => httpsWith({})._validate()).toThrow("at least one header");
  });

  test("rejects invalid header names", () => {
    for (const name of ["bad header", "X-Foo\n", ""]) {
      expect(() => httpsWith({ [name]: "x" })._validate()).toThrow(
        "Invalid header name",
      );
    }
  });

  test("rejects a header set twice in a route, ignoring case", () => {
    expect(() =>
      httpsWith({ Authorization: "a", authorization: "b" })._validate(),
    ).toThrow('Header "authorization" is set more than once');
    expect(() =>
      policyOf([
        OutboundRoute.https({
          domains: ["a.com"],
          headerReplacements: [
            { headers: { "X-Api-Key": "a" } },
            { headers: { "X-Api-Key": "b" } },
          ],
        }),
      ])._validate(),
    ).toThrow('Header "X-Api-Key" is set more than once');
  });

  test("rejects control characters in header values", () => {
    for (const value of ["x\r\ninjected", "x\x00y", "x\x7fy"]) {
      expect(() => httpsWith({ a: value })._validate()).toThrow(
        "control characters",
      );
    }
    httpsWith({ a: "x\ty" })._validate();
  });

  test("$KEY references require a secret", () => {
    expect(() =>
      httpsWith({ Authorization: "Bearer $API_KEY" })._validate(),
    ).toThrow("no `secret` was passed");
    httpsWith({ a: "ca$$h", b: "$5", c: "$" })._validate();
    httpsWith(
      { Authorization: "Bearer $API_KEY" },
      new Secret("st-1", "s"),
    )._validate();
  });

  test("rejects ephemeral secrets", async () => {
    const { mockClient: mc } = createMockModalClients();
    const ephemeral = await mc.secrets.fromObject({ API_KEY: "k" });
    expect(() =>
      httpsWith({ Authorization: "Bearer $API_KEY" }, ephemeral)._validate(),
    ).toThrow("only support named secrets");
  });
});

test("_secrets deduplicates across routes", () => {
  const s1 = new Secret("st-1", "s1");
  const s2 = new Secret("st-2", "s2");
  const policy = policyOf([
    OutboundRoute.https({
      domains: ["a.com"],
      headerReplacements: [
        { secret: s1, headers: { a: "$K" } },
        { headers: { b: "c" } },
      ],
    }),
    OutboundRoute.https({
      domains: ["b.com"],
      headerReplacements: [
        { secret: s1, headers: { a: "$K" } },
        { secret: s2, headers: { a: "$K" } },
      ],
    }),
  ]);
  expect(policy._secrets()).toEqual([s1, s2]);
});

test("_toProto converts every route type", () => {
  const secret = new Secret("st-1", "s");
  const proto = new OutboundPolicy({
    allowTlsDecryption: true,
    routes: [
      OutboundRoute.https({
        domains: ["*.mydomain.com", "mydomain.org"],
        headerReplacements: [
          { secret, headers: { Authorization: "Bearer $API_TOKEN" } },
          { headers: { "X-Static": "v" } },
        ],
      }),
      OutboundRoute.tls({ domains: ["*.example.com"] }),
      OutboundRoute.ip({ cidrs: ["203.0.113.0/24"] }),
      OutboundRoute.any().block(),
    ],
  })._toProto();

  expect(proto.tlsDecryption).toEqual({ allowed: true });
  expect(proto.routes).toHaveLength(4);
  const [https, tls, ip, anyBlock] = proto.routes;

  expect(https.https).toEqual({
    domains: ["*.mydomain.com", "mydomain.org"],
    headerReplacements: [
      { secretId: "st-1", headers: { Authorization: "Bearer $API_TOKEN" } },
      { secretId: "", headers: { "X-Static": "v" } },
    ],
    redirect: undefined,
    transport: OutboundPolicy_Transport.TRANSPORT_UNSPECIFIED,
    ports: [],
  });
  expect(tls.tls).toEqual({
    domains: ["*.example.com"],
    redirect: undefined,
    transport: OutboundPolicy_Transport.TRANSPORT_UNSPECIFIED,
    ports: [],
  });
  expect(ip.ip).toEqual({
    cidrs: ["203.0.113.0/24"],
    transport: OutboundPolicy_Transport.TRANSPORT_UNSPECIFIED,
    ports: [],
    redirect: undefined,
  });
  expect(anyBlock.any).toEqual({ redirect: { block: {} } });
  for (const route of proto.routes) {
    const set = [route.tls, route.https, route.ip, route.any].filter(
      (r) => r !== undefined,
    );
    expect(set).toHaveLength(1);
  }
});

test("_toProto disallows TLS decryption by default", () => {
  const proto = new OutboundPolicy({ routes: [anyRoute] })._toProto();
  expect(proto.tlsDecryption).toEqual({ allowed: false });
  expect(proto.routes).toEqual([{ any: { redirect: undefined } }]);
});

describe("buildSandboxCreateRequestProto with outboundPolicy", () => {
  const policy = policyOf([OutboundRoute.tls({ domains: ["a.com"] })]);

  test("sets outboundPolicy", async () => {
    const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
      outboundPolicy: policy,
      proxy: new Proxy("pr-1"),
    });
    expect(req.definition?.outboundPolicy).toEqual(policy._toProto());
    expect(req.definition?.proxyId).toBe("pr-1");
    expect(req.definition?.experimentalOutboundPolicy).toBeUndefined();
  });

  test("leaves outboundPolicy unset when absent", async () => {
    const req = await buildSandboxCreateRequestProto("app-123", "img-456", {});
    expect(req.definition?.outboundPolicy).toBeUndefined();
  });

  test("validates the policy", async () => {
    await expect(
      buildSandboxCreateRequestProto("app-123", "img-456", {
        outboundPolicy: new OutboundPolicy({ routes: [] }),
      }),
    ).rejects.toThrow("at least one route");
  });

  test.each([
    ["blockNetwork", { blockNetwork: true }, "OutboundRoute.any().block()"],
    [
      "outboundCidrAllowlist",
      { outboundCidrAllowlist: [] },
      "OutboundRoute.ip({ cidrs: [...] })",
    ],
    [
      "outboundDomainAllowlist",
      { outboundDomainAllowlist: ["a.com"] },
      "OutboundRoute.tls({ domains: [...] })",
    ],
    [
      "experimentalOutboundPolicy",
      { experimentalOutboundPolicy: new ExperimentalOutboundPolicy() },
      "OutboundRoute.https(",
    ],
  ])("rejects combining with %s", async (name, extra, suggestion) => {
    const request = buildSandboxCreateRequestProto("app-123", "img-456", {
      outboundPolicy: policy,
      ...extra,
    });
    await expect(request).rejects.toThrow(InvalidError);
    await expect(request).rejects.toThrow(
      `outboundPolicy cannot be used with ${name}; `,
    );
    await expect(request).rejects.toThrow(suggestion);
  });

  test("allows other experimental options", async () => {
    const req = await buildSandboxCreateRequestProto("app-123", "img-456", {
      outboundPolicy: policy,
      experimentalOptions: { enable_docker: true },
    });
    expect(req.definition?.outboundPolicy).toBeDefined();
  });
});

describe("create sends outboundPolicy", () => {
  const app = new App("ap-1234");

  function makeStub() {
    const stub = {
      requests: [] as any[],
      async sandboxCreate(req: any) {
        stub.requests.push(req);
        return { sandboxId: "sb-nGEijt9WbBMlGrsPH9FOaC" };
      },
      async sandboxGetTaskId(_req: any) {
        return { taskId: "ta-123" };
      },
      async sandboxCreateV2(req: any) {
        stub.requests.push(req);
        return {
          sandboxId: "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV",
          taskId: "ta-123",
          tunnels: [],
        };
      },
    };
    return stub;
  }

  function makeClient(stub: unknown, sandboxV2: boolean): ModalClient {
    vi.stubEnv("MODAL_SANDBOX_V2", sandboxV2 ? "1" : "0");
    return new ModalClient({
      cpClient: stub as any,
      tokenId: "test-id",
      tokenSecret: "test-secret",
    });
  }

  test.each([
    ["V1", false],
    ["V2", true],
  ])("on %s", async (_name, sandboxV2) => {
    const stub = makeStub();
    const client = makeClient(stub, sandboxV2);
    const image = new Image(client, "im-123", "");
    const policy = policyOf([
      OutboundRoute.https({
        domains: ["api.example.com"],
        headerReplacements: [
          {
            secret: new Secret("st-1", "s"),
            headers: { Authorization: "Bearer $API_KEY" },
          },
        ],
      }),
      OutboundRoute.ip({ cidrs: ["10.0.0.0/8"] }),
    ]);

    await client.sandboxes.create(app, image, { outboundPolicy: policy });

    expect(stub.requests).toHaveLength(1);
    const def = stub.requests[0].definition;
    expect(def.outboundPolicy).toEqual(policy._toProto());
    expect(
      def.outboundPolicy.routes[0].https.headerReplacements[0].secretId,
    ).toBe("st-1");
    expect(def.outboundPolicy.routes[1].ip.cidrs).toEqual(["10.0.0.0/8"]);
  });

  test.each([
    ["V1", false],
    ["V2", true],
  ])("rejects an invalid policy before any request on %s", async (_n, v2) => {
    const stub = makeStub();
    const client = makeClient(stub, v2);
    const image = new Image(client, "im-123", "");
    const ephemeral = await client.secrets.fromObject({ API_KEY: "k" });
    const policy = policyOf([
      OutboundRoute.https({
        domains: ["api.example.com"],
        headerReplacements: [
          { secret: ephemeral, headers: { Authorization: "Bearer $API_KEY" } },
        ],
      }),
    ]);

    await expect(
      client.sandboxes.create(app, image, { outboundPolicy: policy }),
    ).rejects.toThrow("only support named secrets");
    expect(stub.requests).toHaveLength(0);
  });
});
