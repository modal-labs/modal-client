import { expect, test } from "vitest";
import { InvalidError, OutboundPolicy, Secret } from "modal";
import { templateReferencesKey } from "../src/outbound_policy";

test("templateReferencesKey", () => {
  expect(templateReferencesKey("Bearer $API_KEY")).toBe(true);
  expect(templateReferencesKey("$_PRIVATE")).toBe(true);
  expect(templateReferencesKey("x$y")).toBe(true);
  expect(templateReferencesKey("plain")).toBe(false);
  expect(templateReferencesKey("ca$$h")).toBe(false);
  expect(templateReferencesKey("$$API_KEY")).toBe(false);
  expect(templateReferencesKey("$5")).toBe(false);
  expect(templateReferencesKey("$")).toBe(false);
  expect(templateReferencesKey("ends with $")).toBe(false);
  expect(templateReferencesKey("ca$$h $KEY")).toBe(true);
});

test("withHeaderReplacement builds immutable policy", () => {
  const policy = new OutboundPolicy();
  const updated = policy.withHeaderReplacement({
    domain: "api.example.com",
    headers: { "X-Static": "plain" },
  });

  expect(updated).not.toBe(policy);
  expect(policy._replacements).toEqual([]);
  expect(updated._replacements).toEqual([
    {
      domain: "api.example.com",
      headers: { "X-Static": "plain" },
      secret: undefined,
    },
  ]);
});

test("_toProto fans out replacements", () => {
  const secret = new Secret("st-1", "my-secret");
  const policy = new OutboundPolicy()
    .withHeaderReplacement({
      domain: "api.example.com",
      secret,
      headers: { Authorization: "Bearer $API_KEY", "X-Key-Raw": "$API_KEY" },
    })
    .withHeaderReplacement({
      domain: "*.example.com",
      headers: { "X-Static": "plain" },
    });

  const proto = policy._toProto();
  expect(proto.headerReplacements).toHaveLength(2);

  const [first, second] = proto.headerReplacements;
  expect(first.domain).toBe("api.example.com");
  expect(first.secretId).toBe("st-1");
  expect(first.headers).toEqual({
    Authorization: "Bearer $API_KEY",
    "X-Key-Raw": "$API_KEY",
  });
  expect(second.domain).toBe("*.example.com");
  expect(second.secretId).toBe("");
  expect(second.headers).toEqual({ "X-Static": "plain" });
});

test("_validate rejects invalid domain", () => {
  for (const domain of ["not a domain", "example.com\n"]) {
    const policy = new OutboundPolicy().withHeaderReplacement({
      domain,
      headers: { a: "b" },
    });
    expect(() => policy._validate()).toThrow(InvalidError);
  }
});

test("withHeaderReplacement accepts wildcards", () => {
  const policy = new OutboundPolicy()
    .withHeaderReplacement({ domain: "*.example.com", headers: { a: "b" } })
    .withHeaderReplacement({ domain: "*", headers: { a: "b" } });
  expect(policy._replacements.map((s) => s.domain)).toEqual([
    "*.example.com",
    "*",
  ]);
});

test("_validate rejects empty headers", () => {
  const policy = new OutboundPolicy().withHeaderReplacement({
    domain: "example.com",
    headers: {},
  });
  expect(() => policy._validate()).toThrow("at least one header");
});

test("_validate rejects invalid header name", () => {
  for (const name of ["bad header", "X-Foo\n"]) {
    const policy = new OutboundPolicy().withHeaderReplacement({
      domain: "example.com",
      headers: { [name]: "x" },
    });
    expect(() => policy._validate()).toThrow("Invalid header name");
  }
});

test("_validate rejects control characters in header value", () => {
  for (const value of ["x\r\ninjected", "x\x00y", "x\x7fy"]) {
    const policy = new OutboundPolicy().withHeaderReplacement({
      domain: "example.com",
      headers: { a: value },
    });
    expect(() => policy._validate()).toThrow("control characters");
  }
});

test("_validate allows tab in header value", () => {
  const policy = new OutboundPolicy().withHeaderReplacement({
    domain: "example.com",
    headers: { a: "x\ty" },
  });
  policy._validate();
  expect(policy._replacements[0].headers).toEqual({ a: "x\ty" });
});

test("_validate rejects template without secret", () => {
  const policy = new OutboundPolicy().withHeaderReplacement({
    domain: "example.com",
    headers: { Authorization: "Bearer $API_KEY" },
  });
  expect(() => policy._validate()).toThrow("no `secret` was passed");
});

test("_validate allows escaped dollar without secret", () => {
  const policy = new OutboundPolicy().withHeaderReplacement({
    domain: "example.com",
    headers: { a: "ca$$h", b: "$5", c: "$" },
  });
  policy._validate();
  expect(policy._replacements[0].headers).toEqual({
    a: "ca$$h",
    b: "$5",
    c: "$",
  });
});

test("_validate rejects too many headers", () => {
  const headers: Record<string, string> = {};
  for (let i = 0; i < 26; i++) headers[`X-Header-${i}`] = "x";
  const policy = new OutboundPolicy().withHeaderReplacement({
    domain: "example.com",
    headers,
  });
  expect(() => policy._validate()).toThrow("more than 25 header replacements");

  // The limit counts headers across replacements.
  const full: Record<string, string> = {};
  for (let i = 0; i < 25; i++) full[`X-Header-${i}`] = "x";
  const overLimit = new OutboundPolicy()
    .withHeaderReplacement({ domain: "example.com", headers: full })
    .withHeaderReplacement({ domain: "other.com", headers: { a: "b" } });
  expect(() => overLimit._validate()).toThrow(
    "more than 25 header replacements",
  );
});

test("_secrets deduplicates", () => {
  const secret = new Secret("st-1", "my-secret");
  const policy = new OutboundPolicy()
    .withHeaderReplacement({
      domain: "a.example.com",
      secret,
      headers: { a: "$K" },
    })
    .withHeaderReplacement({
      domain: "b.example.com",
      secret,
      headers: { b: "$K" },
    });
  expect(policy._secrets()).toEqual([secret]);
});

test("_validate rejects ephemeral secrets", async () => {
  const { createMockModalClients } = await import("../test-support/grpc_mock");
  const { mockClient: mc } = createMockModalClients();
  const ephemeral = await mc.secrets.fromObject({ API_KEY: "k" });
  const policy = new OutboundPolicy().withHeaderReplacement({
    domain: "example.com",
    secret: ephemeral,
    headers: { Authorization: "Bearer $API_KEY" },
  });
  expect(() => policy._validate()).toThrow("only support named secrets");
});
