/**
 * Immutable configuration for replacing headers in outbound HTTPS traffic.
 */

import {
  ExperimentalOutboundPolicy as ExperimentalOutboundPolicyProto,
  ExperimentalOutboundPolicy_HeaderReplacement,
} from "../proto/modal_proto/api";
import { InvalidError } from "./errors";
import {
  validateHeaderReplacement,
  validateOutboundDomain,
} from "./outbound_policy";
import { Secret } from "./secret";

const MAX_HEADER_REPLACEMENTS = 25;

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
    for (const { domain, headers, secret } of this._replacements) {
      validateOutboundDomain(domain);
      validateHeaderReplacement(headers, secret);
      total += Object.keys(headers).length;
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
  _toProto(): ExperimentalOutboundPolicyProto {
    return ExperimentalOutboundPolicyProto.create({
      headerReplacements: this._replacements.map((replacement) =>
        ExperimentalOutboundPolicy_HeaderReplacement.create({
          domain: replacement.domain,
          secretId: replacement.secret?.secretId ?? "",
          headers: replacement.headers,
        }),
      ),
    });
  }
}
