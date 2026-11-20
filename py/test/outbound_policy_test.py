# Copyright Modal Labs 2026
import pytest
from typing import Any

import modal
from modal._outbound_policy import _OutboundPolicy, _OutboundRoute
from modal.exception import InvalidError
from modal.secret import _Secret
from modal_proto import api_pb2

P = api_pb2.OutboundPolicy


def _policy(*routes: _OutboundRoute, allow_tls_decryption: bool = False) -> _OutboundPolicy:
    return _OutboundPolicy(allow_tls_decryption=allow_tls_decryption, routes=[*routes, _OutboundRoute.any()])


def test_public_types_exported():
    assert "OutboundPolicy" in modal.__all__
    assert "OutboundRoute" in modal.__all__
    # The experimental header-replacement policy is a distinct type.
    assert modal.OutboundPolicy is not modal.experimental.OutboundPolicy


def test_route_has_no_constructor():
    with pytest.raises(InvalidError, match="OutboundRoute has no constructor"):
        modal.OutboundRoute()
    with pytest.raises(InvalidError, match="OutboundRoute has no constructor"):
        _OutboundRoute("any")


def test_factories():
    tls = _OutboundRoute.tls(domains=["*.example.com", "example.org"])
    assert tls._kind == "tls"
    assert tls._domains == ("*.example.com", "example.org")
    assert tls._block is False

    secret: Any = _Secret.from_name("my-secret")
    https = _OutboundRoute.https(
        domains=["api.example.com"],
        header_replacements=[{"secret": secret, "headers": {"Authorization": "Bearer $TOKEN"}}],
    )
    assert https._kind == "https"
    assert https._domains == ("api.example.com",)
    (replacement,) = https._header_replacements
    assert replacement.headers == {"Authorization": "Bearer $TOKEN"}
    assert replacement.secret is secret

    ip = _OutboundRoute.ip(cidrs=["203.0.113.10/32"])
    assert ip._kind == "ip"
    assert ip._cidrs == ("203.0.113.10/32",)

    assert _OutboundRoute.any()._kind == "any"


def test_factories_copy_inputs():
    domains = ["example.com"]
    headers = {"X-Static": "a"}
    route = _OutboundRoute.https(domains=domains, header_replacements=[{"headers": headers}])
    domains.append("other.com")
    headers["X-Other"] = "b"
    assert route._domains == ("example.com",)
    assert route._header_replacements[0].headers == {"X-Static": "a"}


def test_factories_reject_single_string():
    with pytest.raises(InvalidError, match="`domains` must be a list"):
        _OutboundRoute.tls(domains="example.com")
    with pytest.raises(InvalidError, match="`cidrs` must be a list"):
        _OutboundRoute.ip(cidrs="10.0.0.0/8")


def test_factories_reject_values_that_are_not_lists():
    with pytest.raises(InvalidError, match="`domains` must be a list of strings, got NoneType"):
        _OutboundRoute.tls(domains=None)  # type: ignore[arg-type]
    with pytest.raises(InvalidError, match="`cidrs` must be a list of strings, got int"):
        _OutboundRoute.ip(cidrs=10)  # type: ignore[arg-type]


def test_header_replacement_headers_must_be_a_dict():
    with pytest.raises(InvalidError, match="`headers` must be a dict of header names to values, got str"):
        _OutboundRoute.https(
            domains=["example.com"],
            header_replacements=[{"headers": "Authorization: x"}],  # type: ignore[typeddict-item]
        )


def test_route_repr_shows_what_the_route_matches():
    assert repr(_OutboundRoute.tls(domains=["example.com"])) == "OutboundRoute.tls(domains=['example.com'])"
    assert repr(_OutboundRoute.ip(cidrs=["10.0.0.0/8"])) == "OutboundRoute.ip(cidrs=['10.0.0.0/8'])"
    assert repr(_OutboundRoute.any().block()) == "OutboundRoute.any().block()"
    https = _OutboundRoute.https(
        domains=["example.com"], header_replacements=[{"headers": {"Authorization": "Bearer token"}}]
    )
    assert repr(https) == "OutboundRoute.https(domains=['example.com'])"


def test_block_returns_new_route():
    route = _OutboundRoute.any()
    blocked = route.block()
    assert blocked is not route
    assert route._block is False
    assert blocked._block is True
    _OutboundPolicy(routes=[route])._validate()
    _OutboundPolicy(routes=[blocked])._validate()


def test_block_twice_rejected():
    with pytest.raises(InvalidError, match="already blocks"):
        _OutboundRoute.any().block().block()


def test_block_only_on_any_route():
    for route in [
        _OutboundRoute.tls(domains=["example.com"]),
        _OutboundRoute.https(domains=["example.com"]),
        _OutboundRoute.ip(cidrs=["10.0.0.0/8"]),
    ]:
        with pytest.raises(InvalidError, match="don't support `block\\(\\)`"):
            route.block()


def test_validate_requires_routes():
    with pytest.raises(InvalidError, match="at least one route"):
        _OutboundPolicy(routes=[])._validate()


def test_validate_requires_any_route_last():
    with pytest.raises(InvalidError, match="last route must be `OutboundRoute.any\\(\\)`"):
        _OutboundPolicy(routes=[_OutboundRoute.tls(domains=["example.com"])])._validate()

    with pytest.raises(InvalidError, match="only one `OutboundRoute.any\\(\\)`"):
        _OutboundPolicy(routes=[_OutboundRoute.any(), _OutboundRoute.any().block()])._validate()

    with pytest.raises(InvalidError, match="must be the last route"):
        _OutboundPolicy(routes=[_OutboundRoute.any(), _OutboundRoute.tls(domains=["example.com"])])._validate()


def test_validate_rejects_non_route():
    with pytest.raises(InvalidError, match="only `OutboundRoute`s"):
        _OutboundPolicy(routes=["example.com", _OutboundRoute.any()])._validate()  # type: ignore[list-item]


@pytest.mark.parametrize("factory", [_OutboundRoute.tls, _OutboundRoute.https])
def test_validate_domains(factory):
    with pytest.raises(InvalidError, match="at least one domain"):
        _policy(factory(domains=[]), allow_tls_decryption=True)._validate()
    for bad in ["not a domain", "example.com\n", "*example.com", ""]:
        with pytest.raises(InvalidError, match="Invalid domain"):
            _policy(factory(domains=[bad]), allow_tls_decryption=True)._validate()
    _policy(factory(domains=["*", "*.example.com", "api.example.com"]), allow_tls_decryption=True)._validate()


def test_validate_ip_route():
    with pytest.raises(InvalidError, match="at least one CIDR"):
        _policy(_OutboundRoute.ip(cidrs=[]))._validate()
    _policy(_OutboundRoute.ip(cidrs=["10.0.0.0/8"]))._validate()


def test_validate_https_requires_tls_decryption():
    route = _OutboundRoute.https(domains=["example.com"])
    with pytest.raises(InvalidError, match="HTTPS routes require `allow_tls_decryption=True`"):
        _policy(route)._validate()
    _policy(route, allow_tls_decryption=True)._validate()
    # TLS routes are never decrypted and don't need allow_tls_decryption.
    _policy(_OutboundRoute.tls(domains=["example.com"]))._validate()


def _https(*replacements) -> _OutboundPolicy:
    return _policy(
        _OutboundRoute.https(domains=["example.com"], header_replacements=list(replacements)), allow_tls_decryption=True
    )


def test_validate_header_replacement_rejects_unknown_keys():
    with pytest.raises(InvalidError, match="Unknown key\\(s\\) in header replacement: domain"):
        _https({"headers": {"a": "b"}, "domain": "example.com"})._validate()


def test_validate_header_replacement_rejects_empty_headers():
    with pytest.raises(InvalidError, match="at least one header"):
        _https({"headers": {}})._validate()
    with pytest.raises(InvalidError, match="at least one header"):
        _https({})._validate()


def test_validate_header_replacement_rejects_invalid_header_name():
    for name in ["bad header", "X-Foo\n", ""]:
        with pytest.raises(InvalidError, match="Invalid header name"):
            _https({"headers": {name: "x"}})._validate()


def test_validate_header_replacement_rejects_duplicate_header_names():
    with pytest.raises(InvalidError, match="'authorization' is set more than once"):
        _https({"headers": {"Authorization": "a", "authorization": "b"}})._validate()
    with pytest.raises(InvalidError, match="'X-Api-Key' is set more than once"):
        _https({"headers": {"X-Api-Key": "a"}}, {"headers": {"X-Api-Key": "b"}})._validate()


def test_validate_header_replacement_rejects_control_characters():
    for value in ["x\r\ninjected", "x\x00y", "x\x7fy"]:
        with pytest.raises(InvalidError, match="control characters"):
            _https({"headers": {"a": value}})._validate()
    _https({"headers": {"a": "x\ty"}})._validate()


def test_validate_header_replacement_template_requires_secret():
    with pytest.raises(InvalidError, match="no `secret` was passed"):
        _https({"headers": {"Authorization": "Bearer $API_KEY"}})._validate()
    # `$$` is a literal `$`, and `$` not followed by a key name is not a template.
    _https({"headers": {"a": "ca$$h", "b": "$5", "c": "$"}})._validate()
    _https({"secret": _Secret.from_name("s"), "headers": {"Authorization": "Bearer $API_KEY"}})._validate()


def test_validate_header_replacement_rejects_ephemeral_secret():
    with pytest.raises(InvalidError, match="only support named secrets"):
        _https({"secret": _Secret.from_dict({"API_KEY": "k"}), "headers": {"a": "$API_KEY"}})._validate()


def test_https_route_rejects_a_secret_that_is_not_a_secret():
    # Rejected when the route is built, not only when the policy is used.
    with pytest.raises(InvalidError, match="`secret` must be a `modal.Secret`, got str"):
        _https({"secret": "my-secret", "headers": {"a": "b"}})


def test_secrets_deduplicated():
    secret: Any = _Secret.from_name("my-secret")
    other_secret: Any = _Secret.from_name("other-secret")
    policy = _OutboundPolicy(
        allow_tls_decryption=True,
        routes=[
            _OutboundRoute.https(
                domains=["a.example.com"],
                header_replacements=[{"secret": secret, "headers": {"a": "$K"}}, {"headers": {"b": "c"}}],
            ),
            _OutboundRoute.https(
                domains=["b.example.com"],
                header_replacements=[
                    {"secret": secret, "headers": {"a": "$K"}},
                    {"secret": other_secret, "headers": {"b": "$K"}},
                ],
            ),
            _OutboundRoute.ip(cidrs=["10.0.0.0/8"]),
            _OutboundRoute.any(),
        ],
    )
    assert policy._secrets() == [secret, other_secret]


def test_to_proto():
    secret: Any = _Secret.from_name("my-secret")
    secret._hydrate("st-123", None, None)  # type: ignore[arg-type]

    policy = _OutboundPolicy(
        allow_tls_decryption=True,
        routes=[
            _OutboundRoute.https(
                domains=["*.mydomain.com", "mydomain.org"],
                header_replacements=[
                    {"secret": secret, "headers": {"Authorization": "Bearer $API_TOKEN"}},
                    {"headers": {"X-Static": "plain"}},
                ],
            ),
            _OutboundRoute.https(domains=["api.partner.com"]),
            _OutboundRoute.tls(domains=["*.example.com"]),
            _OutboundRoute.ip(cidrs=["203.0.113.0/24"]),
            _OutboundRoute.any().block(),
        ],
    )
    policy._validate()

    expected = P(
        tls_decryption=P.TLSDecryption(allowed=True),
        routes=[
            P.Route(
                https=P.HTTPSRoute(
                    domains=["*.mydomain.com", "mydomain.org"],
                    header_replacements=[
                        P.HeaderReplacement(secret_id="st-123", headers={"Authorization": "Bearer $API_TOKEN"}),
                        P.HeaderReplacement(secret_id="", headers={"X-Static": "plain"}),
                    ],
                )
            ),
            P.Route(https=P.HTTPSRoute(domains=["api.partner.com"])),
            P.Route(tls=P.TLSRoute(domains=["*.example.com"])),
            P.Route(ip=P.IPRoute(cidrs=["203.0.113.0/24"])),
            P.Route(any=P.AnyRoute(redirect=P.Redirect(block=P.Redirect.Block()))),
        ],
    )
    assert policy._to_proto() == expected

    proto = policy._to_proto()
    # Routes leave redirect, transport and ports unset, so they take their route type's defaults.
    assert not proto.routes[2].tls.HasField("redirect")
    assert proto.routes[3].ip.transport == P.TRANSPORT_UNSPECIFIED
    assert len(proto.routes[3].ip.ports) == 0


def test_to_proto_tls_decryption_disallowed_by_default():
    proto = _OutboundPolicy(routes=[_OutboundRoute.any()])._to_proto()
    assert proto.HasField("tls_decryption")
    assert proto.tls_decryption.allowed is False
    assert proto.routes[0].WhichOneof("route") == "any"
    assert not proto.routes[0].any.HasField("redirect")
