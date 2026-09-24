/**
 * Immutable configuration for replacing headers in outbound HTTPS traffic.
 */

import {
  OutboundPolicy as OutboundPolicyProto,
  OutboundPolicy_HeaderReplacement,
} from "../proto/modal_proto/api";
import { InvalidError } from "./errors";
import { Secret, secretEnvDictHydrator } from "./secret";

const DOMAIN_RE =
  /^(\*\.)?([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$/;
const HEADER_NAME_RE = /^[A-Za-z0-9!#$%&'*+.^_`|~-]+$/;
const MAX_HEADER_REPLACEMENTS = 25;

function hasInvalidHeaderValueChar(value: string): boolean {
  for (const c of value) {
    const code = c.codePointAt(0)!;
    if ((code < 0x20 && c !== "\t") || code === 0x7f) {
      return true;
    }
  }
  return false;
}

/**
 * Whether a header value template references a `$KEY`.
 * @internal
 */
export function templateReferencesKey(template: string): boolean {
  const isKeyStart = (c: string) => /[a-zA-Z_]/.test(c);
  let i = template.indexOf("$");
  while (i !== -1 && i + 1 < template.length) {
    const c = template[i + 1];
    if (c === "$") {
      i = template.indexOf("$", i + 2);
      continue;
    }
    if (isKeyStart(c)) {
      return true;
    }
    i = template.indexOf("$", i + 1);
  }
  return false;
}

/** @internal */
export type HeaderReplacement = {
  domain: string;
  headers: Record<string, string>;
  secret?: Secret;
};

/**
 * Immutable configuration for replacing headers in outbound HTTPS requests
 * from a Sandbox.
 *
 * EXPERIMENTAL: the API is subject to change.
 *
 * Header values support templating with keys in a replacement's secret: a
 * `$`-prefixed key name in the secret is replaced with the secret value.
 * Literal `$` characters are written `$$`.
 *
 * Secret values never enter the Sandbox: they are resolved and injected into
 * matching requests outside the container.
 *
 * ```ts
 * const secret = await modal.secrets.fromName("api-token");
 *
 * const outboundPolicy = new modal.ExperimentalOutboundPolicy()
 *   // Inject a secret-backed Authorization header.
 *   .withHeaderReplacement({
 *     domain: "example.com",
 *     secret,
 *     headers: { Authorization: "Bearer $API_TOKEN" },
 *   })
 *   // Inject a static header into requests to another domain.
 *   .withHeaderReplacement({
 *     domain: "modal.com",
 *     headers: { "X-Trace-Token": "trace_abcd" },
 *   });
 *
 * const sb = await modal.sandboxes.create(app, image, {
 *   experimentalOutboundPolicy: outboundPolicy,
 * });
 * ```
 */
export class ExperimentalOutboundPolicy {
  /** @internal */
  readonly _replacements: readonly HeaderReplacement[];

  /** @internal */
  constructor(_replacements: readonly HeaderReplacement[] = []) {
    this._replacements = _replacements;
  }

  /**
   * Return a new {@link ExperimentalOutboundPolicy} with an added header replacement.
   *
   * @param params.domain Domain the replacements are scoped to. Supports `*.` wildcard
   *   prefixes (matching the apex domain and subdomains) and a bare `"*"`.
   * @param params.headers Header name -> header value. Values support `$KEY` templates
   *   referencing keys in the replacement's `secret`.
   * @param params.secret Named {@link Secret} (e.g. from `client.secrets.fromName`) whose
   *   keys may be referenced in the header value templates. Static replacements pass no secret.
   */
  withHeaderReplacement(params: {
    domain: string;
    headers: Record<string, string>;
    secret?: Secret;
  }): ExperimentalOutboundPolicy {
    const { domain, headers, secret } = params;
    return new ExperimentalOutboundPolicy([
      ...this._replacements,
      { domain, headers: { ...headers }, secret },
    ]);
  }

  /** Check all replacements, throwing `InvalidError` on the first violation found.
   * @internal
   */
  _validate(): void {
    let total = 0;
    for (const replacement of this._replacements) {
      const { domain, headers, secret } = replacement;
      if (domain !== "*" && !DOMAIN_RE.test(domain)) {
        throw new InvalidError(`Invalid domain: ${JSON.stringify(domain)}`);
      }
      const entries = Object.entries(headers);
      if (entries.length === 0) {
        throw new InvalidError("`headers` must contain at least one header");
      }
      for (const [name, value] of entries) {
        if (!HEADER_NAME_RE.test(name)) {
          throw new InvalidError(
            `Invalid header name: ${JSON.stringify(name)}`,
          );
        }
        if (hasInvalidHeaderValueChar(value)) {
          throw new InvalidError(
            `Header value for ${JSON.stringify(name)} must not contain control characters (except tab)`,
          );
        }
        if (secret === undefined && templateReferencesKey(value)) {
          throw new InvalidError(
            `Header value for ${JSON.stringify(name)} references a secret key, but no \`secret\` was passed`,
          );
        }
      }
      if (secret !== undefined && secretEnvDictHydrator(secret) !== undefined) {
        throw new InvalidError(
          "Outbound policies only support named secrets (e.g. `client.secrets.fromName`); " +
            "ephemeral secrets (e.g. `client.secrets.fromObject`) are not yet supported",
        );
      }
      total += entries.length;
    }
    if (total > MAX_HEADER_REPLACEMENTS) {
      throw new InvalidError(
        `Outbound policy cannot have more than ${MAX_HEADER_REPLACEMENTS} header replacements`,
      );
    }
  }

  /** Deduplicated list of secrets referenced by the policy's replacements.
   * @internal
   */
  _secrets(): Secret[] {
    return [
      ...new Set(
        this._replacements.map((s) => s.secret).filter((s) => s !== undefined),
      ),
    ];
  }

  /** Convert to the wire format. Referenced secrets must be hydrated first.
   * @internal
   */
  _toProto(): OutboundPolicyProto {
    return OutboundPolicyProto.create({
      headerReplacements: this._replacements.map((replacement) =>
        OutboundPolicy_HeaderReplacement.create({
          domain: replacement.domain,
          secretId: replacement.secret?.secretId ?? "",
          headers: replacement.headers,
        }),
      ),
    });
  }
}
