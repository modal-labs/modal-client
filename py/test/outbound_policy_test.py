# Copyright Modal Labs 2026
import pytest

from modal.exception import InvalidError
from modal.outbound_policy import _OutboundPolicy, _template_references_key
from modal.secret import _Secret


def test_template_references_key():
    assert _template_references_key("Bearer $API_KEY") is True
    assert _template_references_key("$_PRIVATE") is True
    assert _template_references_key("x$y") is True
    assert _template_references_key("plain") is False
    assert _template_references_key("ca$$h") is False
    assert _template_references_key("$$API_KEY") is False
    assert _template_references_key("$5") is False
    assert _template_references_key("$") is False
    assert _template_references_key("ends with $") is False
    assert _template_references_key("ca$$h $KEY") is True


def test_with_header_replacement_builds_immutable_policy():
    policy = _OutboundPolicy()
    updated = policy.with_header_replacement(domain="api.example.com", headers={"X-Static": "plain"})

    assert policy is not updated
    assert policy._replacements == ()
    assert len(updated._replacements) == 1

    replacement = updated._replacements[0]
    assert replacement.domain == "api.example.com"
    assert replacement.headers == {"X-Static": "plain"}
    assert replacement.secret is None


def test_to_proto_fans_out_replacements():
    policy = (
        _OutboundPolicy()
        .with_header_replacement(
            domain="api.example.com",
            headers={"X-First": "1", "X-Second": "2"},
        )
        .with_header_replacement(domain="*.example.com", headers={"X-Static": "plain"})
    )
    proto = policy._to_proto()
    assert len(proto.header_replacements) == 2

    first, second = proto.header_replacements
    assert first.domain == "api.example.com"
    assert first.secret_id == ""
    assert dict(first.headers) == {"X-First": "1", "X-Second": "2"}
    assert second.domain == "*.example.com"
    assert dict(second.headers) == {"X-Static": "plain"}


def test_validate_rejects_invalid_domain():
    with pytest.raises(InvalidError, match="Invalid domain"):
        _OutboundPolicy().with_header_replacement(domain="not a domain", headers={"a": "b"})._validate()

    with pytest.raises(InvalidError, match="Invalid domain"):
        _OutboundPolicy().with_header_replacement(domain="example.com\n", headers={"a": "b"})._validate()


def test_with_header_replacement_accepts_wildcards():
    policy = (
        _OutboundPolicy()
        .with_header_replacement(domain="*.example.com", headers={"a": "b"})
        .with_header_replacement(domain="*", headers={"a": "b"})
    )
    assert [s.domain for s in policy._replacements] == ["*.example.com", "*"]


def test_validate_rejects_empty_headers():
    with pytest.raises(InvalidError, match="at least one header"):
        _OutboundPolicy().with_header_replacement(domain="example.com", headers={})._validate()


def test_validate_rejects_invalid_header_name():
    with pytest.raises(InvalidError, match="Invalid header name"):
        _OutboundPolicy().with_header_replacement(domain="example.com", headers={"bad header": "x"})._validate()

    with pytest.raises(InvalidError, match="Invalid header name"):
        _OutboundPolicy().with_header_replacement(domain="example.com", headers={"X-Foo\n": "x"})._validate()


def test_validate_rejects_control_characters_in_value():
    for value in ["x\r\ninjected", "x\x00y", "x\x7fy"]:
        with pytest.raises(InvalidError, match="control characters"):
            _OutboundPolicy().with_header_replacement(domain="example.com", headers={"a": value})._validate()


def test_validate_allows_tab_in_value():
    _OutboundPolicy().with_header_replacement(domain="example.com", headers={"a": "x\ty"})._validate()


def test_validate_rejects_template_without_secret():
    with pytest.raises(InvalidError, match="no `secret` was passed"):
        _OutboundPolicy().with_header_replacement(
            domain="example.com", headers={"Authorization": "Bearer $API_KEY"}
        )._validate()


def test_validate_allows_escaped_dollar_without_secret():
    policy = _OutboundPolicy().with_header_replacement(
        domain="example.com", headers={"a": "ca$$h", "b": "$5", "c": "$"}
    )
    policy._validate()
    assert policy._replacements[0].headers == {"a": "ca$$h", "b": "$5", "c": "$"}


def test_validate_rejects_ephemeral_secret():
    with pytest.raises(InvalidError, match="only support named secrets"):
        _OutboundPolicy().with_header_replacement(
            domain="example.com",
            secret=_Secret.from_dict({"API_KEY": "k"}),
            headers={"Authorization": "Bearer $API_KEY"},
        )._validate()


def test_validate_rejects_too_many_headers():
    with pytest.raises(InvalidError, match="more than 25 header replacements"):
        _OutboundPolicy().with_header_replacement(
            domain="example.com", headers={f"X-Header-{i}": "x" for i in range(26)}
        )._validate()

    # The limit counts headers across replacements.
    policy = (
        _OutboundPolicy()
        .with_header_replacement(domain="example.com", headers={f"X-Header-{i}": "x" for i in range(25)})
        .with_header_replacement(domain="other.com", headers={"a": "b"})
    )
    with pytest.raises(InvalidError, match="more than 25 header replacements"):
        policy._validate()


def test_secrets_deduplicates():
    secret = _Secret.from_name("my-secret")
    policy = (
        _OutboundPolicy()
        .with_header_replacement(domain="a.example.com", secret=secret, headers={"a": "$K"})
        .with_header_replacement(domain="b.example.com", secret=secret, headers={"b": "$K"})
    )
    assert policy._secrets() == [secret]
