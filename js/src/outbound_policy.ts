/**
 * Route-based outbound network policy for Sandboxes.
 */

import {
  OutboundPolicy as OutboundPolicyProto,
  OutboundPolicy_AnyRoute,
  OutboundPolicy_HeaderReplacement,
  OutboundPolicy_HTTPSRoute,
  OutboundPolicy_IPRoute,
  OutboundPolicy_Redirect,
  OutboundPolicy_Redirect_Block,
  OutboundPolicy_Route,
  OutboundPolicy_TLSDecryption,
  OutboundPolicy_TLSRoute,
} from "../proto/modal_proto/api";
import { InvalidError } from "./errors";
import { Secret, secretEnvDictHydrator } from "./secret";

const DOMAIN_RE =
  /^(\*\.)?([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$/;
const HEADER_NAME_RE = /^[A-Za-z0-9!#$%&'*+.^_`|~-]+$/;

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

/**
 * Throw `InvalidError` unless `domain` is `"*"` or a (possibly `*.`-prefixed)
 * domain name.
 * @internal
 * @hidden
 */
export function validateOutboundDomain(domain: string): void {
  if (domain !== "*" && !DOMAIN_RE.test(domain)) {
    throw new InvalidError(`Invalid domain: ${JSON.stringify(domain)}`);
  }
}

/**
 * Throw `InvalidError` unless `headers` is a valid, non-empty set of header
 * value templates for `secret`, and `secret` (if any) is a named secret.
 * @internal
 * @hidden
 */
export function validateHeaderReplacement(
  headers: Record<string, string>,
  secret: Secret | undefined,
): void {
  const entries = Object.entries(headers);
  if (entries.length === 0) {
    throw new InvalidError("`headers` must contain at least one header");
  }
  for (const [name, value] of entries) {
    if (!HEADER_NAME_RE.test(name)) {
      throw new InvalidError(`Invalid header name: ${JSON.stringify(name)}`);
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
}

/**
 * Headers to set on requests matched by an HTTPS {@link OutboundRoute}.
 *
 * Header values support templating with keys in `secret`: a `$`-prefixed key
 * name in the secret (e.g. `$API_TOKEN`) is replaced with the secret value.
 * Literal `$` characters are written `$$`. Secret values never enter the
 * Sandbox.
 */
export interface OutboundHeaderReplacement {
  /**
   * Named {@link Secret} (e.g. from `client.secrets.fromName`) whose keys the
   * header values may reference. Static values need no secret.
   */
  secret?: Secret;
  /** Header name -> header value template. */
  headers: Record<string, string>;
}

type RouteMatch =
  | { kind: "tls"; domains: readonly string[] }
  | {
      kind: "https";
      domains: readonly string[];
      headerReplacements: readonly OutboundHeaderReplacement[];
    }
  | { kind: "ip"; cidrs: readonly string[] }
  | { kind: "any" };

const ROUTE_LABELS: Record<RouteMatch["kind"], string> = {
  tls: "TLS",
  https: "HTTPS",
  ip: "IP",
  any: "OutboundRoute.any()",
};

/**
 * One rule of an {@link OutboundPolicy}: which outbound traffic it matches.
 *
 * Create routes with the static factories {@link OutboundRoute.tls},
 * {@link OutboundRoute.https}, {@link OutboundRoute.ip} and
 * {@link OutboundRoute.any}. Matched traffic goes to its original
 * destination, except that the `any` route can block what it matches with
 * {@link OutboundRoute#block block}. Routes are immutable: `block()` returns a
 * new route.
 */
export class OutboundRoute {
  /** @internal @hidden */
  readonly _match: RouteMatch;
  /** @internal @hidden */
  readonly _block: boolean;

  private constructor(_match: RouteMatch, _block = false) {
    this._match = _match;
    this._block = _block;
  }

  /**
   * Match connections that carry TLS from the first byte to the given
   * domains, by the domain the client requests (SNI), on tcp/443. These
   * connections are never decrypted. Traffic to these domains on other ports
   * or transports isn't matched by this route.
   *
   * @param params.domains Domains to match. A `*.` prefix matches subdomains
   *   at any depth but not the domain itself: `*.example.com` matches
   *   `a.example.com` and `a.b.example.com`, not `example.com`. A bare `"*"`
   *   matches all domains.
   */
  static tls(params: { domains: string[] }): OutboundRoute {
    return new OutboundRoute({ kind: "tls", domains: [...params.domains] });
  }

  /**
   * Match HTTPS connections to the given domains on tcp/443. These
   * connections are always decrypted, so the policy must set
   * `allowTlsDecryption: true`. Requests whose `Host` header doesn't match
   * the requested domain are refused. Traffic to these domains on other ports
   * or transports isn't matched by this route.
   *
   * @param params.domains Domains to match. A `*.` prefix matches subdomains
   *   at any depth but not the domain itself: `*.example.com` matches
   *   `a.example.com` and `a.b.example.com`, not `example.com`. A bare `"*"`
   *   matches all domains.
   * @param params.headerReplacements Headers to set on matched requests, in
   *   public beta. See {@link OutboundHeaderReplacement}. Each header is set on
   *   every matching request, so the request has exactly one header with that
   *   name and the given value. Copies the request already has are removed,
   *   whatever their case, and the header is added if the request has none. A
   *   route can set each header name only once, ignoring case.
   */
  static https(params: {
    domains: string[];
    headerReplacements?: OutboundHeaderReplacement[];
  }): OutboundRoute {
    return new OutboundRoute({
      kind: "https",
      domains: [...params.domains],
      headerReplacements: (params.headerReplacements ?? []).map(
        ({ secret, headers }) => ({ secret, headers: { ...headers } }),
      ),
    });
  }

  /**
   * Match all traffic to the given CIDRs, on any transport and port.
   *
   * @param params.cidrs Destination CIDRs to match, e.g. `"203.0.113.0/24"`.
   */
  static ip(params: { cidrs: string[] }): OutboundRoute {
    return new OutboundRoute({ kind: "ip", cidrs: [...params.cidrs] });
  }

  /**
   * Match all traffic that no earlier route matched. Every policy must end
   * with exactly one `any` route. Without {@link OutboundRoute#block block},
   * the traffic it matches goes to its original destination.
   */
  static any(): OutboundRoute {
    return new OutboundRoute({ kind: "any" });
  }

  /**
   * Return a new route that blocks the traffic it matches. Only
   * {@link OutboundRoute.any} supports `block()`.
   */
  block(): OutboundRoute {
    if (this._match.kind !== "any") {
      throw new InvalidError(
        `${ROUTE_LABELS[this._match.kind]} routes don't support block()`,
      );
    }
    if (this._block) {
      throw new InvalidError(
        "This route already blocks the traffic it matches",
      );
    }
    return new OutboundRoute(this._match, true);
  }

  /** Check the route, throwing `InvalidError` on the first violation found.
   * @internal
   * @hidden
   */
  _validate(allowTlsDecryption: boolean): void {
    const match = this._match;
    const label = ROUTE_LABELS[match.kind];
    switch (match.kind) {
      case "tls":
      case "https": {
        if (match.domains.length === 0) {
          throw new InvalidError(`${label} routes need at least one domain`);
        }
        match.domains.forEach(validateOutboundDomain);
        if (match.kind === "tls") {
          return;
        }
        if (!allowTlsDecryption) {
          throw new InvalidError(
            "HTTPS routes require an OutboundPolicy with `allowTlsDecryption: true`",
          );
        }
        const names = new Set<string>();
        for (const { headers: h, secret } of match.headerReplacements) {
          validateHeaderReplacement(h, secret);
          for (const name of Object.keys(h)) {
            if (names.has(name.toLowerCase())) {
              throw new InvalidError(
                `Header ${JSON.stringify(name)} is set more than once in the route`,
              );
            }
            names.add(name.toLowerCase());
          }
        }
        return;
      }
      case "ip":
        if (match.cidrs.length === 0) {
          throw new InvalidError("IP routes need at least one CIDR");
        }
        return;
      case "any":
        return;
    }
  }

  /** @internal @hidden */
  _toProto(): OutboundPolicy_Route {
    const match = this._match;
    switch (match.kind) {
      case "tls":
        return OutboundPolicy_Route.create({
          tls: OutboundPolicy_TLSRoute.create({
            domains: [...match.domains],
          }),
        });
      case "https":
        return OutboundPolicy_Route.create({
          https: OutboundPolicy_HTTPSRoute.create({
            domains: [...match.domains],
            headerReplacements: match.headerReplacements.map(
              ({ secret, headers }) =>
                OutboundPolicy_HeaderReplacement.create({
                  secretId: secret?.secretId ?? "",
                  headers: { ...headers },
                }),
            ),
          }),
        });
      case "ip":
        return OutboundPolicy_Route.create({
          ip: OutboundPolicy_IPRoute.create({ cidrs: [...match.cidrs] }),
        });
      case "any":
        return OutboundPolicy_Route.create({
          any: OutboundPolicy_AnyRoute.create({
            redirect: this._block
              ? OutboundPolicy_Redirect.create({
                  block: OutboundPolicy_Redirect_Block.create(),
                })
              : undefined,
          }),
        });
    }
  }
}

/**
 * Ordered rules for a Sandbox's outbound traffic.
 *
 * Each connection is handled by the first {@link OutboundRoute} that matches
 * it, and goes to its original destination. The last route must be
 * {@link OutboundRoute.any}, which matches everything earlier routes didn't
 * and either lets it through or, with `block()`, blocks it.
 *
 * ```ts
 * const secret = await modal.secrets.fromName("api-token");
 *
 * const policy = new modal.OutboundPolicy({
 *   allowTlsDecryption: true,
 *   routes: [
 *     modal.OutboundRoute.https({
 *       domains: ["api.example.com"],
 *       headerReplacements: [
 *         { secret, headers: { Authorization: "Bearer $API_TOKEN" } },
 *       ],
 *     }),
 *     modal.OutboundRoute.tls({
 *       domains: ["github.com", "*.github.com", "pypi.org"],
 *     }),
 *     modal.OutboundRoute.ip({ cidrs: ["203.0.113.0/24"] }),
 *     modal.OutboundRoute.any().block(),
 *   ],
 * });
 *
 * const sb = await modal.sandboxes.create(app, image, { outboundPolicy: policy });
 * ```
 */
export class OutboundPolicy {
  /** @internal @hidden */
  readonly _allowTlsDecryption: boolean;
  /** @internal @hidden */
  readonly _routes: readonly OutboundRoute[];

  /**
   * @param params.allowTlsDecryption Allow Modal to decrypt the Sandbox's TLS
   *   traffic on routes that need it, by generating a certificate authority
   *   for the Sandbox and attempting to make the Sandbox trust it. Decrypts
   *   nothing on its own. Required by HTTPS routes. Fixed when the Sandbox is
   *   created. Sandboxes that allow TLS decryption don't support memory
   *   snapshots. Defaults to `false`.
   * @param params.routes Ordered routes; the first route matching a
   *   connection handles it. Must end with exactly one
   *   {@link OutboundRoute.any} route.
   */
  constructor(params: {
    allowTlsDecryption?: boolean;
    routes: OutboundRoute[];
  }) {
    this._allowTlsDecryption = params.allowTlsDecryption ?? false;
    this._routes = [...(params.routes ?? [])];
  }

  /** Check the policy, throwing `InvalidError` on the first violation found.
   * @internal
   * @hidden
   */
  _validate(): void {
    if (this._routes.length === 0) {
      throw new InvalidError("Outbound policy must have at least one route");
    }
    const anyCount = this._routes.filter((r) => r._match.kind === "any").length;
    const last = this._routes[this._routes.length - 1];
    if (anyCount !== 1 || last._match.kind !== "any") {
      throw new InvalidError(
        "Outbound policy must have exactly one `OutboundRoute.any()` route, and it must be last",
      );
    }
    for (const route of this._routes) {
      route._validate(this._allowTlsDecryption);
    }
  }

  /** Deduplicated list of secrets referenced by header replacements.
   * @internal
   * @hidden
   */
  _secrets(): Secret[] {
    const secrets = new Set<Secret>();
    for (const route of this._routes) {
      if (route._match.kind === "https") {
        for (const { secret } of route._match.headerReplacements) {
          if (secret !== undefined) secrets.add(secret);
        }
      }
    }
    return [...secrets];
  }

  /** Convert to the wire format. Referenced secrets must be hydrated first.
   * @internal
   * @hidden
   */
  _toProto(): OutboundPolicyProto {
    return OutboundPolicyProto.create({
      tlsDecryption: OutboundPolicy_TLSDecryption.create({
        allowed: this._allowTlsDecryption,
      }),
      routes: this._routes.map((r) => r._toProto()),
    });
  }
}
