# Copyright Modal Labs 2026
"""Immutable configuration for replacing headers in outbound HTTPS traffic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from modal_proto import api_pb2

from ._outbound_policy import (
    _header_value_has_invalid_chars,
    _is_valid_domain,
    _is_valid_header_name,
    _template_references_key,
)
from ._utils.async_utils import synchronize_api
from .exception import InvalidError
from .secret import _Secret

_MAX_HEADER_REPLACEMENTS: int = 25


@dataclass(frozen=True)
class _HeaderReplacement:
    domain: str
    headers: dict[str, str]
    secret: _Secret | None


class _OutboundPolicy:
    """Immutable configuration for replacing headers in outbound HTTPS requests
    from a Sandbox.

    Deprecated, and removed in modal 1.7.0: use `modal.OutboundPolicy` with
    `modal.OutboundRoute.https(..., header_replacements=...)` routes instead.

    Header values support templating with keys in a replacement's secret: a
    `$`-prefixed key name in the secret is replaced with the secret value.
    Literal `$` characters are written `$$`.

    Secret values never enter the Sandbox: they are resolved and injected into
    matching requests outside the container.

    Examples:
        ```python
        import modal
        import modal.experimental

        secret = modal.Secret.from_name("api-token")

        outbound_policy = (
            modal.experimental.OutboundPolicy()
            # Inject a secret-backed Authorization header.
            .with_header_replacement(
                domain="example.com",
                secret=secret,
                headers={"Authorization": "Bearer $API_TOKEN"},
            )
            # Inject a static header into requests to another domain.
            .with_header_replacement(
                domain="modal.com",
                headers={"X-Trace-Token": "trace_abcd"},
            )
        )

        sb = modal.Sandbox.create(_experimental_outbound_policy=outbound_policy)
        ```
    """

    _replacements: tuple[_HeaderReplacement, ...]

    def __init__(self, _replacements: tuple[_HeaderReplacement, ...] = ()):
        self._replacements = _replacements

    def with_header_replacement(
        self,
        *,
        domain: str,
        headers: Mapping[str, str],
        secret: _Secret | None = None,
    ) -> _OutboundPolicy:
        """Return a new `OutboundPolicy` with an added header replacement.

        Args:
            domain: Domain the replacements are scoped to. Supports `*.` wildcard
                prefixes (matching the apex domain and subdomains) and a bare `"*"`.
            headers: Header name -> header value. Values support `$KEY` templates
                referencing keys in the replacement's `secret`.
            secret: Named Secret (e.g. from `Secret.from_name`) whose keys may be
                referenced in the header value templates. Static replacements pass no
                secret.
        """
        replacement = _HeaderReplacement(domain=domain, headers=dict(headers), secret=secret)
        return _OutboundPolicy(self._replacements + (replacement,))

    def _validate(self) -> None:
        """Check all replacements, raising `InvalidError` on the first violation found."""
        total = 0
        for replacement in self._replacements:
            if not _is_valid_domain(replacement.domain):
                raise InvalidError(f"Invalid domain: {replacement.domain!r}")
            if not replacement.headers:
                raise InvalidError("`headers` must contain at least one header")
            for name, value in replacement.headers.items():
                if not _is_valid_header_name(name):
                    raise InvalidError(f"Invalid header name: {name!r}")
                if _header_value_has_invalid_chars(value):
                    raise InvalidError(f"Header value for {name!r} must not contain control characters (except tab)")
                if replacement.secret is None and _template_references_key(value):
                    raise InvalidError(f"Header value for {name!r} references a secret key, but no `secret` was passed")
            if replacement.secret is not None and replacement.secret._is_ephemeral:
                raise InvalidError(
                    "Outbound policies only support named secrets (e.g. `Secret.from_name`); "
                    "ephemeral secrets (e.g. `Secret.from_dict`) are not yet supported"
                )
            total += len(replacement.headers)
        if total > _MAX_HEADER_REPLACEMENTS:
            raise InvalidError(f"Outbound policy cannot have more than {_MAX_HEADER_REPLACEMENTS} header replacements")

    def _secrets(self) -> list[_Secret]:
        """Deduplicated list of secrets referenced by the policy's replacements."""
        return list(dict.fromkeys(r.secret for r in self._replacements if r.secret is not None))

    def _to_proto(self) -> api_pb2.ExperimentalOutboundPolicy:
        """Convert to the wire format. Referenced secrets must be hydrated first."""
        return api_pb2.ExperimentalOutboundPolicy(
            header_replacements=[
                api_pb2.ExperimentalOutboundPolicy.HeaderReplacement(
                    domain=replacement.domain,
                    secret_id=(replacement.secret.object_id if replacement.secret is not None else ""),
                    headers=replacement.headers,
                )
                for replacement in self._replacements
            ]
        )


def _validate_compatible_network_access(
    outbound_policy: _OutboundPolicy | None,
    block_network: bool,
    outbound_domain_allowlist: Sequence[str] | None,
) -> None:
    """Reject combinations of network configuration and outbound policy that the server does not support."""
    if outbound_policy is None:
        return
    if block_network:
        raise InvalidError("`_experimental_outbound_policy` cannot be used when `block_network` is enabled")
    if outbound_domain_allowlist:
        raise InvalidError("`_experimental_outbound_policy` cannot be used with `outbound_domain_allowlist`")


# The public interface lives under `modal.experimental` (re-exported there), but
# the synchronized class is defined here so that the generated type stub for this
# module declares it as a class.
OutboundPolicy = synchronize_api(_OutboundPolicy)
