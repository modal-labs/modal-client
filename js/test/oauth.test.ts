import { createPublicKey, generateKeyPairSync } from "node:crypto";
import jwt from "jsonwebtoken";
import { expect, test } from "vitest";
import {
  mintOAuthClientAssertion,
  parseOAuthJwtKey,
  rsaJwkThumbprint,
} from "../src/oauth";

const { privateKey, publicKey } = generateKeyPairSync("rsa", {
  modulusLength: 2048,
});
const privateKeyPem = privateKey
  .export({ type: "pkcs8", format: "pem" })
  .toString();
const publicKeyPem = publicKey.export({ type: "spki", format: "pem" });

test("parseOAuthJwtKey accepts PEM with escaped newlines", () => {
  const escaped = privateKeyPem.replaceAll("\n", "\\n");
  expect(rsaJwkThumbprint(parseOAuthJwtKey(escaped))).toBe(
    rsaJwkThumbprint(parseOAuthJwtKey(privateKeyPem)),
  );
});

test.each([
  ["not-a-key", "must be an unencrypted RSA private key"],
  [
    generateKeyPairSync("ed25519")
      .privateKey.export({ type: "pkcs8", format: "pem" })
      .toString(),
    "must be an unencrypted RSA private key",
  ],
  [
    generateKeyPairSync("rsa", { modulusLength: 1024 })
      .privateKey.export({ type: "pkcs8", format: "pem" })
      .toString(),
    "must be at least 2048 bits",
  ],
])("parseOAuthJwtKey rejects unusable keys", (pem, message) => {
  expect(() => parseOAuthJwtKey(pem)).toThrow(message);
});

test("rsaJwkThumbprint hashes the RFC 7638 required members", () => {
  const key = parseOAuthJwtKey(privateKeyPem);
  const kid = rsaJwkThumbprint(key);
  expect(kid).toMatch(/^[A-Za-z0-9_-]{43}$/);
  // The signer and whoever publishes the JWKS must agree on the identifier.
  expect(rsaJwkThumbprint(createPublicKey(key))).toBe(kid);
});

test("mintOAuthClientAssertion signs claims the server requires", () => {
  const key = parseOAuthJwtKey(privateKeyPem);
  const assertion = mintOAuthClientAssertion("oc-client-id", key);

  const decoded = jwt.decode(assertion, { complete: true });
  expect(decoded?.header.alg).toBe("RS256");
  expect(decoded?.header.kid).toBe(rsaJwkThumbprint(key));

  const claims = jwt.verify(assertion, publicKeyPem, {
    algorithms: ["RS256"],
    audience: "modal-server",
    issuer: "oc-client-id",
  }) as jwt.JwtPayload;
  expect(claims.sub).toBe("oc-client-id");
  expect(claims.jti).toBeTruthy();
  // Longer-lived assertions stop authenticating.
  expect(claims.exp! - claims.iat!).toBe(300);
});

test("mintOAuthClientAssertion mints a fresh jti each time", () => {
  const key = parseOAuthJwtKey(privateKeyPem);
  const first = jwt.decode(mintOAuthClientAssertion("oc-client-id", key));
  const second = jwt.decode(mintOAuthClientAssertion("oc-client-id", key));
  expect((first as jwt.JwtPayload).jti).not.toBe(
    (second as jwt.JwtPayload).jti,
  );
});
