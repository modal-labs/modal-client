# Copyright Modal Labs 2026
import json
import pytest

from modal._utils.sandbox_fs_utils import (
    ErrorPayload,
    translate_exec_errors,
    translate_exec_unexpected_error,
    try_parse_error_payload,
)
from modal.exception import InternalError, NotFoundError, SandboxFilesystemError


def test_translate_exec_unexpected_error_includes_backend_error_code():
    exc = InternalError("Failed to start exec command (Error code: ABCD1234)")
    result = translate_exec_unexpected_error("read_bytes", "/tmp/missing.txt", exc)
    assert isinstance(result, SandboxFilesystemError)
    assert "Error code: ABCD1234" in str(result)


def test_translate_exec_errors_converts_sandbox_unavailable_error():
    with pytest.raises(NotFoundError, match="Sandbox is unavailable"):
        with translate_exec_errors("read_bytes", "/tmp/file.txt"):
            raise NotFoundError("sandbox not found")


def test_translate_exec_errors_converts_general_modal_error():
    with pytest.raises(SandboxFilesystemError, match="unexpected error"):
        with translate_exec_errors("read_bytes", "/tmp/file.txt"):
            raise InternalError("something broke")


def test_try_parse_error_payload_returns_payload_for_valid_json():
    stderr = json.dumps({"error_kind": "NotFound", "message": "file not found"})
    result = try_parse_error_payload(stderr)
    assert result == ErrorPayload(error_kind="NotFound", message="file not found")


def test_try_parse_error_payload_accepts_bytes():
    stderr = json.dumps({"error_kind": "PermissionDenied", "message": "access denied"}).encode("utf-8")
    result = try_parse_error_payload(stderr)
    assert result == ErrorPayload(error_kind="PermissionDenied", message="access denied")


def test_try_parse_error_payload_returns_none_for_empty_stderr():
    assert try_parse_error_payload("") is None
    assert try_parse_error_payload(b"") is None


def test_try_parse_error_payload_returns_none_for_non_json():
    assert try_parse_error_payload("not json at all") is None


def test_try_parse_error_payload_returns_none_for_non_dict_json():
    assert try_parse_error_payload(json.dumps([1, 2, 3])) is None


def test_try_parse_error_payload_returns_none_for_missing_error_kind():
    assert try_parse_error_payload(json.dumps({"message": "oops"})) is None


def test_try_parse_error_payload_returns_none_for_non_string_error_kind():
    assert try_parse_error_payload(json.dumps({"error_kind": 42, "message": "oops"})) is None


def test_try_parse_error_payload_returns_none_for_missing_message():
    assert try_parse_error_payload(json.dumps({"error_kind": "NotFound"})) is None


def test_try_parse_error_payload_returns_none_for_non_string_message():
    assert try_parse_error_payload(json.dumps({"error_kind": "NotFound", "message": 123})) is None


def test_try_parse_error_payload_returns_none_for_blank_message():
    assert try_parse_error_payload(json.dumps({"error_kind": "NotFound", "message": "  "})) is None
