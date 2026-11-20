# Copyright Modal Labs 2026
"""Route-based policy for outbound network traffic from a Sandbox."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from modal_proto import api_pb2

from ._utils.async_utils import synchronize_api
from .exception import InvalidError
from .secret import _Secret
from .types import OutboundHeaderReplacement

_DOMAIN_RE = re.compile(
    r"^(\*\.)?([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\Z"
)
_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]+\Z")


def _is_valid_domain(domain: str) -> bool:
    """Whether `domain` is a domain name, a `*.`-prefixed wildcard domain, or a bare `*`."""
    return domain == "*" or _DOMAIN_RE.match(domain) is not None


def _is_valid_header_name(name: str) -> bool:
    return _HEADER_NAME_RE.match(name) is not None


def _header_value_has_invalid_chars(value: str) -> bool:
    return any((ord(c) < 0x20 and c != "\t") or ord(c) == 0x7F for c in value)


def _template_references_key(template: str) -> bool:
    """Whether a header value template references a `$KEY`."""
    i = template.find("$")
    while i != -1 and i + 1 < len(template):
        c = template[i + 1]
        if c == "$":
            i = template.find("$", i + 2)
            continue
        if c.isascii() and (c.isalpha() or c == "_"):
            return True
        i = template.find("$", i + 1)
    return False


_RouteKind = Literal["tls", "https", "ip", "any"]
_ROUTE_KIND_NAMES = {"tls": "TLS", "https": "HTTPS", "ip": "IP", "any": "OutboundRoute.any()"}

_HEADER_REPLACEMENT_KEYS = frozenset({"headers", "secret"})


@dataclass(frozen=True)
class _RouteHeaderReplacement:
    headers: dict[str, str]
    secret: _Secret | None
    unknown_keys: tuple[str, ...] = ()


def _string_list(value: Sequence[str], arg_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise InvalidError(f"`{arg_name}` must be a list of strings, not a single string")
    try:
        return tuple(value)
    except TypeError:
        raise InvalidError(f"`{arg_name}` must be a list of strings, got {type(value).__name__}") from None


def _normalize_header_replacement(replacement: OutboundHeaderReplacement) -> _RouteHeaderReplacement:
    """Check the shape and types of a header replacement. The values are checked by `_validate`."""
    # Read as a plain mapping: at runtime the argument may not match its annotation.
    raw = cast(Mapping[str, object], replacement)
    if not isinstance(raw, Mapping):
        raise InvalidError("Each header replacement must be a dict with a `headers` key and an optional `secret` key")
    headers = raw.get("headers")
    if headers is not None and not isinstance(headers, Mapping):
        raise InvalidError(f"`headers` must be a dict of header names to values, got {type(headers).__name__}")
    secret = raw.get("secret")
    if secret is not None and not isinstance(secret, _Secret):
        raise InvalidError(f"`secret` must be a `modal.Secret`, got {type(secret).__name__}")
    return _RouteHeaderReplacement(
        headers=dict(headers) if headers is not None else {},
        secret=secret,
        unknown_keys=tuple(sorted(str(k) for k in raw if k not in _HEADER_REPLACEMENT_KEYS)),
    )


class _OutboundRoute:
    """A route in an `OutboundPolicy`: which outbound traffic it matches.

    Create routes with the `OutboundRoute.tls`, `OutboundRoute.https`,
    `OutboundRoute.ip` and `OutboundRoute.any` factories. Traffic a route
    matches goes to its original destination, except that the `any` route can
    block what it matches. Routes are immutable: `block()` returns a new route.

    Examples:
        ```python
        import modal

        # Allow TLS connections to a domain and its subdomains.
        modal.OutboundRoute.tls(domains=["example.com", "*.example.com"])

        # Block everything no earlier route matched.
        modal.OutboundRoute.any().block()
        ```
    """

    _kind: _RouteKind
    _domains: tuple[str, ...]
    _cidrs: tuple[str, ...]
    _header_replacements: tuple[_RouteHeaderReplacement, ...]
    _block: bool

    def __init__(self, *args, **kwargs):
        """mdmd:hidden"""
        raise InvalidError(
            "OutboundRoute has no constructor. Use `OutboundRoute.tls`, `OutboundRoute.https`, "
            "`OutboundRoute.ip` or `OutboundRoute.any` instead."
        )

    @classmethod
    def _new(
        cls,
        kind: _RouteKind,
        *,
        domains: tuple[str, ...] = (),
        cidrs: tuple[str, ...] = (),
        header_replacements: tuple[_RouteHeaderReplacement, ...] = (),
        block: bool = False,
    ) -> _OutboundRoute:
        route = cls.__new__(cls)
        route._kind = kind
        route._domains = domains
        route._cidrs = cidrs
        route._header_replacements = header_replacements
        route._block = block
        return route

    def __repr__(self) -> str:
        # Header replacements are left out: their values can be secret material.
        if self._kind in ("tls", "https"):
            args = f"domains={list(self._domains)!r}"
        elif self._kind == "ip":
            args = f"cidrs={list(self._cidrs)!r}"
        else:
            args = ""
        return f"OutboundRoute.{self._kind}({args})" + (".block()" if self._block else "")

    @classmethod
    def tls(cls, *, domains: Sequence[str]) -> _OutboundRoute:
        """Match connections that carry TLS from the first byte to one of `domains`, on tcp/443.

        Connections are matched by the domain the client requests during the TLS
        handshake (SNI). TLS routes never decrypt traffic: the connection's
        contents are passed through as-is. Traffic to these domains on other
        ports or transports isn't matched by this route.

        Args:
            domains: Domains to match. A `*.` prefix matches subdomains at any
                depth but not the domain itself: `*.example.com` matches
                `a.example.com` and `a.b.example.com`, not `example.com`. A bare
                `"*"` matches all domains.
        """
        return cls._new("tls", domains=_string_list(domains, "domains"))

    @classmethod
    def https(
        cls,
        *,
        domains: Sequence[str],
        header_replacements: Sequence[OutboundHeaderReplacement] | None = None,
    ) -> _OutboundRoute:
        """Match HTTPS connections to one of `domains`, on tcp/443.

        HTTPS routes always decrypt traffic, which requires the policy to be
        created with `allow_tls_decryption=True`, even when the route has no
        header replacements. Requests whose `Host` header doesn't match the domain
        requested for the connection are refused. Traffic to these domains on
        other ports or transports isn't matched by this route.

        Header values support templating with keys in a replacement's `secret`:
        a `$`-prefixed key name in the secret is replaced with the secret value.
        Literal `$` characters are written `$$`. Secret values never enter the
        Sandbox: they are injected into matching requests outside the container.

        Args:
            domains: Domains to match. A `*.` prefix matches subdomains at any
                depth but not the domain itself: `*.example.com` matches
                `a.example.com` and `a.b.example.com`, not `example.com`. A bare
                `"*"` matches all domains.
            header_replacements: Headers to set on matching requests, in public
                beta. Each entry is a dict with a `headers` key (header name ->
                value template) and an optional `secret` key: a named Secret
                (e.g. from `Secret.from_name`) whose keys the header values may
                reference.
                Each header is set on every matching request, so the request
                has exactly one header with that name and the given value.
                Copies the request already has are removed, whatever their
                case, and the header is added if the request has none. A route
                can set each header name only once, ignoring case.

        Examples:
            ```python
            secret = modal.Secret.from_name("api-token")

            route = modal.OutboundRoute.https(
                domains=["api.example.com"],
                header_replacements=[
                    {"secret": secret, "headers": {"Authorization": "Bearer $API_TOKEN"}},
                ],
            )
            ```
        """
        return cls._new(
            "https",
            domains=_string_list(domains, "domains"),
            header_replacements=tuple(_normalize_header_replacement(r) for r in header_replacements or ()),
        )

    @classmethod
    def ip(cls, *, cidrs: Sequence[str]) -> _OutboundRoute:
        """Match all traffic to one of `cidrs`, on any transport and port.

        Args:
            cidrs: CIDR ranges to match, e.g. `"203.0.113.0/24"`.
        """
        return cls._new("ip", cidrs=_string_list(cidrs, "cidrs"))

    @classmethod
    def any(cls) -> _OutboundRoute:
        """Match all traffic that no earlier route in the policy matched.

        Every `OutboundPolicy` needs exactly one `any` route, as its last
        route. Without a redirect, the traffic it matches goes to its original
        destination; call `.block()` on the route to block it instead.
        """
        return cls._new("any")

    def block(self) -> _OutboundRoute:
        """Return a new route that blocks the traffic it matches.

        Only `OutboundRoute.any()` supports `block()`.
        """
        if self._kind != "any":
            raise InvalidError(f"{_ROUTE_KIND_NAMES[self._kind]} routes don't support `block()`")
        if self._block:
            raise InvalidError("This route already blocks the traffic it matches")
        return _OutboundRoute._new(self._kind, block=True)

    def _validate(self, *, allow_tls_decryption: bool) -> None:
        kind_name = _ROUTE_KIND_NAMES[self._kind]
        if self._kind in ("tls", "https"):
            if not self._domains:
                raise InvalidError(f"{kind_name} routes must have at least one domain")
            for domain in self._domains:
                if not isinstance(domain, str) or not _is_valid_domain(domain):
                    raise InvalidError(f"Invalid domain: {domain!r}")
        if self._kind == "https" and not allow_tls_decryption:
            raise InvalidError("HTTPS routes require `allow_tls_decryption=True` on the `OutboundPolicy`")
        if self._kind == "ip":
            if not self._cidrs:
                raise InvalidError("IP routes must have at least one CIDR")
            for cidr in self._cidrs:
                if not isinstance(cidr, str) or not cidr:
                    raise InvalidError(f"Invalid CIDR: {cidr!r}")
        header_names: set[str] = set()
        for replacement in self._header_replacements:
            if replacement.unknown_keys:
                raise InvalidError(
                    f"Unknown key(s) in header replacement: {', '.join(replacement.unknown_keys)}; "
                    "expected `headers` and optionally `secret`"
                )
            if not replacement.headers:
                raise InvalidError("`headers` must contain at least one header")
            secret = replacement.secret
            for name, value in replacement.headers.items():
                if not isinstance(name, str) or not _is_valid_header_name(name):
                    raise InvalidError(f"Invalid header name: {name!r}")
                if name.lower() in header_names:
                    raise InvalidError(f"Header {name!r} is set more than once in the route")
                header_names.add(name.lower())
                if not isinstance(value, str):
                    raise InvalidError(f"Header value for {name!r} must be a string")
                if _header_value_has_invalid_chars(value):
                    raise InvalidError(f"Header value for {name!r} must not contain control characters (except tab)")
                if secret is None and _template_references_key(value):
                    raise InvalidError(f"Header value for {name!r} references a secret key, but no `secret` was passed")
            if secret is not None and secret._is_ephemeral:
                raise InvalidError(
                    "Outbound policies only support named secrets (e.g. `Secret.from_name`); "
                    "ephemeral secrets (e.g. `Secret.from_dict`) are not yet supported"
                )


class _OutboundPolicy:
    """Routes that control where outbound network traffic from a Sandbox goes.

    A policy is an ordered list of `OutboundRoute`s. Each outbound connection is
    handled by the first route that matches it, and goes to its original
    destination. The last route must be `OutboundRoute.any()`, which matches
    everything no earlier route matched and either lets it through or, with
    `.block()`, blocks it.

    Examples:
        ```python
        import modal

        secret = modal.Secret.from_name("api-token")

        policy = modal.OutboundPolicy(
            allow_tls_decryption=True,
            routes=[
                # Inject a secret-backed header into HTTPS requests.
                modal.OutboundRoute.https(
                    domains=["api.example.com"],
                    header_replacements=[{"secret": secret, "headers": {"Authorization": "Bearer $API_TOKEN"}}],
                ),
                # Allow TLS connections to these domains.
                modal.OutboundRoute.tls(domains=["github.com", "*.github.com", "pypi.org"]),
                # Allow all traffic to a CIDR range.
                modal.OutboundRoute.ip(cidrs=["203.0.113.0/24"]),
                # Block everything else.
                modal.OutboundRoute.any().block(),
            ],
        )

        sb = modal.Sandbox.create(app=app, outbound_policy=policy)
        ```
    """

    _allow_tls_decryption: bool
    _routes: tuple[_OutboundRoute, ...]

    def __init__(self, *, allow_tls_decryption: bool = False, routes: Sequence[_OutboundRoute]):
        """Create an outbound policy.

        Args:
            allow_tls_decryption: Allow Modal to decrypt the Sandbox's TLS traffic on
                routes that need it, by generating a certificate authority for the
                Sandbox and attempting to make the Sandbox trust it. Decrypts nothing
                on its own. Required by HTTPS routes. Fixed when the Sandbox is created.
                Sandboxes that allow TLS decryption don't support memory snapshots.
            routes: Ordered routes; the first route that matches a connection
                handles it. Must end with `OutboundRoute.any()`.
        """
        self._allow_tls_decryption = allow_tls_decryption
        self._routes = tuple(routes)

    def _validate(self) -> None:
        """Check the policy, raising `InvalidError` on the first violation found."""
        if not isinstance(self._allow_tls_decryption, bool):
            raise InvalidError("`allow_tls_decryption` must be a bool")
        if not self._routes:
            raise InvalidError("`routes` must contain at least one route")
        for route in self._routes:
            if not isinstance(route, _OutboundRoute):
                raise InvalidError(f"`routes` must contain only `OutboundRoute`s, got {type(route).__name__}")
        any_routes = [i for i, route in enumerate(self._routes) if route._kind == "any"]
        if not any_routes:
            raise InvalidError("The last route must be `OutboundRoute.any()`")
        if len(any_routes) > 1:
            raise InvalidError("`routes` can contain only one `OutboundRoute.any()`")
        if any_routes[0] != len(self._routes) - 1:
            raise InvalidError("`OutboundRoute.any()` must be the last route")
        for route in self._routes:
            route._validate(allow_tls_decryption=self._allow_tls_decryption)

    def _secrets(self) -> list[_Secret]:
        """Deduplicated list of secrets referenced by header replacements."""
        return list(
            dict.fromkeys(
                r.secret for route in self._routes for r in route._header_replacements if r.secret is not None
            )
        )

    def _to_proto(self) -> api_pb2.OutboundPolicy:
        """Convert to the wire format. Referenced secrets must be hydrated first."""
        P = api_pb2.OutboundPolicy

        # A local helper keeps nested proto types out of the module's generated type stub.
        def route_proto(route: _OutboundRoute) -> api_pb2.OutboundPolicy.Route:
            if route._kind == "tls":
                return P.Route(tls=P.TLSRoute(domains=route._domains))
            if route._kind == "https":
                header_replacements = [
                    P.HeaderReplacement(
                        secret_id=r.secret.object_id if r.secret is not None else "",
                        headers=r.headers,
                    )
                    for r in route._header_replacements
                ]
                return P.Route(https=P.HTTPSRoute(domains=route._domains, header_replacements=header_replacements))
            if route._kind == "ip":
                return P.Route(ip=P.IPRoute(cidrs=route._cidrs))
            redirect = P.Redirect(block=P.Redirect.Block()) if route._block else None
            return P.Route(any=P.AnyRoute(redirect=redirect))

        return P(
            tls_decryption=P.TLSDecryption(allowed=self._allow_tls_decryption),
            routes=[route_proto(route) for route in self._routes],
        )


def _validate_outbound_policy_compatibility(
    outbound_policy: _OutboundPolicy | None,
    *,
    block_network: bool,
    outbound_cidr_allowlist: Sequence[str] | None,
    outbound_domain_allowlist: Sequence[str] | None,
    experimental_outbound_policy: Any,
) -> None:
    """Reject Sandbox options that conflict with `outbound_policy`."""
    if outbound_policy is None:
        return
    if block_network:
        raise InvalidError(
            "`outbound_policy` cannot be used when `block_network` is enabled; "
            "end the policy's routes with `OutboundRoute.any().block()` instead"
        )
    if outbound_cidr_allowlist is not None:
        raise InvalidError(
            "`outbound_policy` cannot be used with `outbound_cidr_allowlist`; "
            "use an `OutboundRoute.ip(cidrs=[...])` route, followed by `OutboundRoute.any().block()`, instead"
        )
    if outbound_domain_allowlist is not None:
        raise InvalidError(
            "`outbound_policy` cannot be used with `outbound_domain_allowlist`; "
            "use an `OutboundRoute.tls(domains=[...])` route, followed by `OutboundRoute.any().block()`, instead"
        )
    if experimental_outbound_policy is not None:
        raise InvalidError(
            "`outbound_policy` cannot be used with `_experimental_outbound_policy`; "
            "use `OutboundRoute.https(domains=[...], header_replacements=[...])` routes "
            "in a policy with `allow_tls_decryption=True` instead"
        )


OutboundRoute = synchronize_api(_OutboundRoute)
OutboundPolicy = synchronize_api(_OutboundPolicy)
