# Copyright Modal Labs 2026
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import NoReturn, Optional, Union

from ..config import logger
from ..exception import (
    ConnectionError,
    Error as ModalError,
    InvalidError,
    NotFoundError,
    SandboxFilesystemDirectoryNotEmptyError,
    SandboxFilesystemError,
    SandboxFilesystemFileTooLargeError,
    SandboxFilesystemIsADirectoryError,
    SandboxFilesystemNotADirectoryError,
    SandboxFilesystemNotFoundError,
    SandboxFilesystemPathAlreadyExistsError,
    SandboxFilesystemPermissionError,
    ServiceError,
)

_EXEC_SANDBOX_UNAVAILABLE_ERROR_TYPES = (NotFoundError, ServiceError, ConnectionError)


@dataclass
class ErrorPayload:
    error_kind: str
    message: str
    detail: str = ""


def _stderr_to_text(stderr: Union[str, bytes]) -> str:
    stderr_bytes = stderr if isinstance(stderr, bytes) else stderr.encode("utf-8")
    return stderr_bytes.decode("utf-8", errors="replace").strip()


def try_parse_error_payload(stderr: Union[str, bytes]) -> Optional[ErrorPayload]:
    stderr_text = _stderr_to_text(stderr)
    if not stderr_text:
        return None
    try:
        payload = json.loads(stderr_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    error_kind = payload.get("error_kind")
    message = payload.get("message")
    if not isinstance(error_kind, str):
        return None
    if not isinstance(message, str) or not message.strip():
        return None
    detail = payload.get("detail", "")
    if not isinstance(detail, str):
        detail = ""
    return ErrorPayload(error_kind=error_kind, message=message, detail=detail)


def raise_read_file_error(returncode: int, stderr: Union[str, bytes], remote_path: str) -> NoReturn:
    if payload := try_parse_error_payload(stderr):
        logger.debug(
            f"sandbox-fs-tools read error: path={remote_path}, "
            f"error_kind={payload.error_kind}, message={payload.message}, detail={payload.detail}"
        )
        if payload.error_kind == "NotFound":
            raise SandboxFilesystemNotFoundError(f"{payload.message}: {remote_path}")
        if payload.error_kind == "IsDirectory":
            raise SandboxFilesystemIsADirectoryError(f"{payload.message}: {remote_path}")
        if payload.error_kind == "PermissionDenied":
            raise SandboxFilesystemPermissionError(f"{payload.message}: {remote_path}")
        if payload.error_kind == "FileTooLarge":
            raise SandboxFilesystemFileTooLargeError(f"{payload.message}: {remote_path}")
        raise SandboxFilesystemError(payload.message)

    if stderr_text := _stderr_to_text(stderr):
        logger.debug(f"Unstructured modal-sandbox-fs-tools stderr: {stderr_text}")
    raise SandboxFilesystemError(f"Operation on '{remote_path}' failed with exit code {returncode}")


def _extract_support_error_code(exc: Exception) -> Optional[str]:
    if match := re.search(r"Error code:\s*([A-Z0-9]{8})", str(exc)):
        return match.group(1)
    return None


def _translate_exec_sandbox_unavailable_error(operation: str, path: str, exc: Exception) -> Exception:
    logger.debug(
        f"Sandbox filesystem control-plane error for operation={operation}, path={path}: {type(exc).__name__}: {exc}"
    )
    return NotFoundError("The Sandbox is unavailable. This Sandbox may have already shut down.")


def translate_exec_unexpected_error(operation: str, path: str, exc: Exception) -> Exception:
    """Translate an unexpected exec-level error into a generic user-facing error.

    This is less than ideal — it discards the original exception and returns a
    generic message — but it's necessary to avoid surfacing "call to exec()
    failed"-style messages to users who shouldn't be aware that the filesystem
    API is implemented in terms of Sandbox.exec().
    """
    error_code = _extract_support_error_code(exc)
    logger.debug(
        f"Unexpected sandbox filesystem exec error for operation={operation}, path={path}: {type(exc).__name__}: {exc}"
    )
    support_suffix = (
        f"please contact support@modal.com (Error code: {error_code})"
        if error_code
        else "please contact support@modal.com"
    )
    return SandboxFilesystemError(f"An unexpected error occurred, {support_suffix}")


def raise_write_file_error(returncode: int, stderr: Union[str, bytes], remote_path: str) -> NoReturn:
    if payload := try_parse_error_payload(stderr):
        logger.debug(
            f"sandbox-fs-tools write error: path={remote_path}, "
            f"error_kind={payload.error_kind}, message={payload.message}, detail={payload.detail}"
        )
        if payload.error_kind == "NotDirectory" or payload.error_kind == "AlreadyExists":
            raise SandboxFilesystemNotADirectoryError(f"{payload.message}: {remote_path}")
        if payload.error_kind == "IsDirectory":
            raise SandboxFilesystemIsADirectoryError(f"{payload.message}: {remote_path}")
        if payload.error_kind == "PermissionDenied":
            raise SandboxFilesystemPermissionError(f"{payload.message}: {remote_path}")
        raise SandboxFilesystemError(payload.message)

    if stderr_text := _stderr_to_text(stderr):
        logger.debug(f"Unstructured modal-sandbox-fs-tools stderr: {stderr_text}")
    raise SandboxFilesystemError(f"Operation on '{remote_path}' failed with exit code {returncode}")


def make_write_file_command(remote_path: str) -> str:
    """Build the JSON command string for a WriteFile operation.

    The returned JSON must match the `Command` enum in the modal-sandbox-fs-tools
    Rust crate (crates/modal-sandbox-fs-tools/src/lib.rs). Treat changes to
    this schema like protobuf changes: fields must not be removed or renamed,
    only added with backwards-compatible defaults.
    """
    return json.dumps({"WriteFile": {"path": remote_path}})


def make_read_file_command(remote_path: str) -> str:
    """Build the JSON command string for a ReadFile operation.

    The returned JSON must match the `Command` enum in the modal-sandbox-fs-tools
    Rust crate (crates/modal-sandbox-fs-tools/src/lib.rs). Treat changes to
    this schema like protobuf changes: fields must not be removed or renamed,
    only added with backwards-compatible defaults.
    """
    return json.dumps({"ReadFile": {"path": remote_path}})


def make_remove_command(remote_path: str, recursive: bool) -> str:
    """Build the JSON command string for a Remove operation.

    The returned JSON must match the `Command` enum in the modal-sandbox-fs-tools
    Rust crate (crates/modal-sandbox-fs-tools/src/lib.rs). Treat changes to
    this schema like protobuf changes: fields must not be removed or renamed,
    only added with backwards-compatible defaults.
    """
    return json.dumps({"Remove": {"path": remote_path, "recursive": recursive}})


def raise_remove_error(returncode: int, stderr: Union[str, bytes], remote_path: str) -> NoReturn:
    if payload := try_parse_error_payload(stderr):
        logger.debug(
            f"sandbox-fs-tools remove error: path={remote_path}, "
            f"error_kind={payload.error_kind}, message={payload.message}, detail={payload.detail}"
        )
        if payload.error_kind == "NotFound":
            raise SandboxFilesystemNotFoundError(f"{payload.message}: {remote_path}")
        if payload.error_kind == "DirectoryNotEmpty":
            raise SandboxFilesystemDirectoryNotEmptyError(f"{payload.message}: {remote_path}")
        if payload.error_kind == "NotSupported":
            raise InvalidError(
                f"{payload.message}: {remote_path} - this operation is not supported for CloudBucketMounts"
            )
        if payload.error_kind == "PermissionDenied":
            raise SandboxFilesystemPermissionError(f"{payload.message}: {remote_path}")
        raise SandboxFilesystemError(payload.message)

    if stderr_text := _stderr_to_text(stderr):
        logger.debug(f"Unstructured modal-sandbox-fs-tools stderr: {stderr_text}")
    raise SandboxFilesystemError(f"Operation on '{remote_path}' failed with exit code {returncode}")


def make_make_directory_command(remote_path: str, create_parents: bool) -> str:
    """Build the JSON command string for a MakeDirectory operation.

    The returned JSON must match the `Command` enum in the modal-sandbox-fs-tools
    Rust crate (crates/modal-sandbox-fs-tools/src/lib.rs). Treat changes to
    this schema like protobuf changes: fields must not be removed or renamed,
    only added with backwards-compatible defaults.
    """
    return json.dumps({"MakeDirectory": {"path": remote_path, "parents": create_parents}})


def raise_make_directory_error(returncode: int, stderr: Union[str, bytes], remote_path: str) -> NoReturn:
    if payload := try_parse_error_payload(stderr):
        logger.debug(
            f"sandbox-fs-tools make_directory error: path={remote_path}, "
            f"error_kind={payload.error_kind}, message={payload.message}, detail={payload.detail}"
        )
        if payload.error_kind == "NotFound":
            raise SandboxFilesystemNotFoundError(f"{payload.message}: {remote_path}")
        if payload.error_kind == "PathAlreadyExists":
            raise SandboxFilesystemPathAlreadyExistsError(f"{payload.message}: {remote_path}")
        if payload.error_kind == "NotDirectory":
            raise SandboxFilesystemNotADirectoryError(f"{payload.message}: {remote_path}")
        if payload.error_kind == "PermissionDenied":
            raise SandboxFilesystemPermissionError(f"{payload.message}: {remote_path}")
        if payload.error_kind == "NotSupported":
            raise InvalidError(
                f"{payload.message}: {remote_path} - this operation is not supported for CloudBucketMounts"
            )
        raise SandboxFilesystemError(payload.message)

    if stderr_text := _stderr_to_text(stderr):
        logger.debug(f"Unstructured modal-sandbox-fs-tools stderr: {stderr_text}")
    raise SandboxFilesystemError(f"Operation on '{remote_path}' failed with exit code {returncode}")


def validate_absolute_remote_path(remote_path: str, operation: str) -> None:
    if not PurePosixPath(remote_path).is_absolute():
        raise InvalidError(f"Sandbox.filesystem.{operation}() currently only supports absolute remote_path values")


@contextmanager
def translate_exec_errors(operation: str, remote_path: str):
    """Translate exec-level exceptions into user-facing errors."""
    try:
        yield
    except _EXEC_SANDBOX_UNAVAILABLE_ERROR_TYPES as exc:
        raise _translate_exec_sandbox_unavailable_error(operation, remote_path, exc) from None
    except ModalError as exc:
        raise translate_exec_unexpected_error(operation, remote_path, exc) from None
