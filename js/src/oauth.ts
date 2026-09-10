// Private-key JWT (RFC 7523) assertions for OAuth client authentication.

import {
  createHash,
  createPrivateKey,
  createPublicKey,
  createSign,
  type KeyObject,
} from "node:crypto";
import { v4 as uuidv4 } from "uuid";
import { InvalidError } from "./errors";

const OAUTH_CLIENT_ASSERTION_AUDIENCE = "modal-server";
const OAUTH_CLIENT_ASSERTION_LIFETIME_SECONDS = 5 * 60;
const MIN_RSA_KEY_SIZE = 2048;

const INVALID_KEY_MESSAGE =
  "oauth_jwt_key must be an unencrypted RSA private key encoded as PEM.";

/**
 * Parses a PEM-encoded RSA private key. Newlines may be escaped as `\n` so the
 * key survives a single-line environment variable.
 *
 * @internal
 * @hidden
 */
export function parseOAuthJwtKey(privateKeyPem: string): KeyObject {
  let key: KeyObject;
  try {
    key = createPrivateKey(privateKeyPem.replaceAll("\\n", "\n"));
  } catch {
    throw new InvalidError(INVALID_KEY_MESSAGE);
  }
  if (key.asymmetricKeyType !== "rsa") {
    throw new InvalidError(INVALID_KEY_MESSAGE);
  }
  if ((key.asymmetricKeyDetails?.modulusLength ?? 0) < MIN_RSA_KEY_SIZE) {
    throw new InvalidError("oauth_jwt_key must be at least 2048 bits.");
  }
  return key;
}

/**
 * Computes the RFC 7638 thumbprint of an RSA key, used as the assertion's `kid`.
 *
 * @internal
 * @hidden
 */
export function rsaJwkThumbprint(key: KeyObject): string {
  const publicKey = key.type === "public" ? key : createPublicKey(key);
  const jwk = publicKey.export({ format: "jwk" }) as {
    e: string;
    n: string;
  };
  // Member order is part of the spec.
  const canonical = JSON.stringify({ e: jwk.e, kty: "RSA", n: jwk.n });
  return createHash("sha256").update(canonical).digest("base64url");
}

function base64UrlJson(value: unknown): string {
  return Buffer.from(JSON.stringify(value), "utf-8").toString("base64url");
}

/**
 * Mints a short-lived RS256 client assertion for the given OAuth client.
 *
 * @internal
 * @hidden
 */
export function mintOAuthClientAssertion(
  clientId: string,
  key: KeyObject,
): string {
  const issuedAt = Math.floor(Date.now() / 1000);
  const header = base64UrlJson({
    alg: "RS256",
    typ: "JWT",
    kid: rsaJwkThumbprint(key),
  });
  const payload = base64UrlJson({
    iss: clientId,
    sub: clientId,
    aud: OAUTH_CLIENT_ASSERTION_AUDIENCE,
    iat: issuedAt,
    exp: issuedAt + OAUTH_CLIENT_ASSERTION_LIFETIME_SECONDS,
    jti: uuidv4(),
  });
  const signingInput = `${header}.${payload}`;
  const signature = createSign("RSA-SHA256")
    .update(signingInput)
    .sign(key)
    .toString("base64url");
  return `${signingInput}.${signature}`;
}
