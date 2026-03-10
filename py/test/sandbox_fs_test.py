# Copyright Modal Labs 2026
import pathlib
import pytest
import random

from modal import App, Sandbox
from modal.exception import (
    InvalidError,
    SandboxFilesystemIsADirectoryError,
    SandboxFilesystemNotADirectoryError,
    SandboxFilesystemNotFoundError,
)

# The auto-generated .pyi stub for io_streams only includes synchronicity-wrapped
# types, so mypy cannot see this plain module-level constant. We suppress the
# error here rather than defining a separate constant we'd have to keep in sync.
from modal.io_streams import TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE  # type: ignore[attr-defined]

from .supports.skip import skip_windows

skip_non_subprocess = skip_windows("Needs subprocess support")

_MOCK_SANDBOX_FS_TOOLS_PATH = pathlib.Path(__file__).parent / "supports" / "mock_sandbox_fs_tools.py"


def _random_bytes(size: int, *, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(size))


@pytest.fixture
def sandbox_fs_tools(monkeypatch, tmp_path):
    """Points the sandbox filesystem tools path at the mock script.

    The mock exec backend ("router") runs commands as local subprocesses,
    so this script is executed directly on the host. It emulates the Rust
    binary's behavior for ReadFile, including structured JSON error
    payloads on stderr.

    We wrap the Python script in a shell script that explicitly calls the
    current test interpreter (sys.executable). This avoids inheriting the
    Bazel PYTHONPATH when the subprocess runs via the shebang's `env python3`,
    which would otherwise cause import hook version mismatches (e.g. ddtrace).
    """
    import sys

    wrapper = tmp_path / "mock_sandbox_fs_tools"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{_MOCK_SANDBOX_FS_TOOLS_PATH}" "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setattr("modal.sandbox_fs._SANDBOX_FS_TOOLS_PATH", str(wrapper))


@pytest.fixture
def app(client):
    app_ = App()
    with app_.run(client=client):
        yield app_


@pytest.fixture
def sandbox(app):
    sb = None
    try:
        sb = Sandbox.create(app=app)
        yield sb
    finally:
        if sb:
            sb.terminate()


# ---------------------------------------------------------------------------
# read_text
# ---------------------------------------------------------------------------


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_read_text_returns_expected_text(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "read-text.txt")
    text_payload = "hello from read_text\nsnowman: ☃\n"
    (tmp_path / "read-text.txt").write_text(text_payload, encoding="utf-8")

    assert sandbox.filesystem.read_text(remote_path) == text_payload


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_read_text_returns_empty_string_for_empty_file(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "empty.txt")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    assert sandbox.filesystem.read_text(remote_path) == ""


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_read_text_errors_on_relative_remote_path(servicer, client, exec_backend, sandbox):
    with pytest.raises(InvalidError, match="absolute remote_path values"):
        sandbox.filesystem.read_text("relative/path.txt")


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_read_text_errors_when_remote_path_missing(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "read-text-missing.txt")

    with pytest.raises(SandboxFilesystemNotFoundError):
        sandbox.filesystem.read_text(remote_path)


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_read_text_errors_when_remote_path_is_directory(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_dir = str(tmp_path / "read-text-dir")
    (tmp_path / "read-text-dir").mkdir()

    with pytest.raises(SandboxFilesystemIsADirectoryError):
        sandbox.filesystem.read_text(remote_dir)


# ---------------------------------------------------------------------------
# read_bytes
# ---------------------------------------------------------------------------


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_read_bytes_returns_expected_bytes(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "read-bytes.bin")
    payload = b"\x00\x01\x02binary\x00payload\xff"
    (tmp_path / "read-bytes.bin").write_bytes(payload)

    assert sandbox.filesystem.read_bytes(remote_path) == payload


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_read_bytes_returns_empty_bytes_for_empty_file(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "empty.bin")
    (tmp_path / "empty.bin").write_bytes(b"")

    assert sandbox.filesystem.read_bytes(remote_path) == b""


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_read_bytes_errors_on_relative_remote_path(servicer, client, exec_backend, sandbox):
    with pytest.raises(InvalidError, match="absolute remote_path values"):
        sandbox.filesystem.read_bytes("relative/path.bin")


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_read_bytes_errors_when_remote_path_missing(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "read-bytes-missing.bin")

    with pytest.raises(SandboxFilesystemNotFoundError):
        sandbox.filesystem.read_bytes(remote_path)


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_read_bytes_errors_when_remote_path_is_directory(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_dir = str(tmp_path / "read-bytes-dir")
    (tmp_path / "read-bytes-dir").mkdir()

    with pytest.raises(SandboxFilesystemIsADirectoryError):
        sandbox.filesystem.read_bytes(remote_dir)


# ---------------------------------------------------------------------------
# copy_to_local
# ---------------------------------------------------------------------------


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_to_local_writes_file_to_correct_local_location(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    payload = _random_bytes(1024, seed=1)
    src = tmp_path / "source.bin"
    src.write_bytes(payload)
    local_path = tmp_path / "nested" / "copied.bin"

    sandbox.filesystem.copy_to_local(str(src), local_path)

    assert local_path.exists()
    assert local_path.read_bytes() == payload


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_to_local_creates_parent_directories_if_needed(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    payload = _random_bytes(2048, seed=2)
    src = tmp_path / "source-parent.bin"
    src.write_bytes(payload)
    local_path = tmp_path / "deep" / "nested" / "path" / "copied.bin"

    sandbox.filesystem.copy_to_local(str(src), local_path)

    assert local_path.read_bytes() == payload


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_to_local_copies_correct_contents_when_file_is_empty(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    src = tmp_path / "empty.bin"
    src.write_bytes(b"")
    local_path = tmp_path / "empty-out.bin"

    sandbox.filesystem.copy_to_local(str(src), local_path)

    assert local_path.read_bytes() == b""


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_to_local_overwrites_existing_local_file(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    payload = _random_bytes(4096, seed=5)
    src = tmp_path / "overwrite-source.bin"
    src.write_bytes(payload)
    local_path = tmp_path / "overwrite.bin"
    local_path.write_bytes(b"old-data")

    sandbox.filesystem.copy_to_local(str(src), local_path)

    assert local_path.read_bytes() == payload


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_to_local_preserves_existing_file_on_remote_error(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    local_path = tmp_path / "existing.bin"
    local_path.write_bytes(b"stable-content")
    remote_path = str(tmp_path / "missing.bin")

    with pytest.raises(SandboxFilesystemNotFoundError):
        sandbox.filesystem.copy_to_local(remote_path, local_path)

    assert local_path.read_bytes() == b"stable-content"


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_to_local_errors_on_relative_remote_path(servicer, client, exec_backend, sandbox, tmp_path):
    with pytest.raises(InvalidError, match="absolute remote_path values"):
        sandbox.filesystem.copy_to_local("relative/path.bin", tmp_path / "ignored.bin")


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_to_local_errors_if_remote_does_not_exist(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "missing.bin")
    local_path = tmp_path / "missing-out.bin"

    with pytest.raises(SandboxFilesystemNotFoundError):
        sandbox.filesystem.copy_to_local(remote_path, local_path)


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_to_local_errors_when_local_path_is_directory(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    payload = _random_bytes(512, seed=6)
    src = tmp_path / "local-dir-source.bin"
    src.write_bytes(payload)
    local_dir = tmp_path / "local-dir"
    local_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises((IsADirectoryError, OSError)):
        sandbox.filesystem.copy_to_local(str(src), local_dir)

    # Verify no temp files were leaked in the target directory.
    leftover = list(local_dir.parent.glob(".modal-sandbox-fs-tmp-*"))
    assert leftover == [], f"temp file leaked: {leftover}"


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_to_local_errors_when_remote_path_is_directory(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_dir = str(tmp_path / "remote-dir")
    (tmp_path / "remote-dir").mkdir()

    with pytest.raises(SandboxFilesystemIsADirectoryError):
        sandbox.filesystem.copy_to_local(remote_dir, tmp_path / "unused.bin")


# ---------------------------------------------------------------------------
# write_bytes
# ---------------------------------------------------------------------------


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_bytes_writes_expected_bytes(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "write-bytes.bin")
    payload = b"\x00\x01\x02binary\x00payload\xff"

    sandbox.filesystem.write_bytes(payload, remote_path)

    assert (tmp_path / "write-bytes.bin").read_bytes() == payload


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_bytes_writes_empty_data(servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools):
    remote_path = str(tmp_path / "write-empty.bin")

    sandbox.filesystem.write_bytes(b"", remote_path)

    assert (tmp_path / "write-empty.bin").read_bytes() == b""


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_bytes_creates_parent_directories(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "nested" / "deep" / "write.bin")
    payload = _random_bytes(1024, seed=20)

    sandbox.filesystem.write_bytes(payload, remote_path)

    assert (tmp_path / "nested" / "deep" / "write.bin").read_bytes() == payload


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_bytes_overwrites_existing_file(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "overwrite.bin")
    (tmp_path / "overwrite.bin").write_bytes(b"old-data")

    sandbox.filesystem.write_bytes(b"new-data", remote_path)

    assert (tmp_path / "overwrite.bin").read_bytes() == b"new-data"


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_bytes_accepts_bytearray(servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools):
    remote_path = str(tmp_path / "write-bytearray.bin")
    payload = bytearray(b"\x00\x01\x02bytearray\xff")

    sandbox.filesystem.write_bytes(payload, remote_path)

    assert (tmp_path / "write-bytearray.bin").read_bytes() == payload


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_bytes_accepts_memoryview(servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools):
    remote_path = str(tmp_path / "write-memoryview.bin")
    raw = b"\x00\x01\x02memoryview\xff"

    sandbox.filesystem.write_bytes(memoryview(raw), remote_path)

    assert (tmp_path / "write-memoryview.bin").read_bytes() == raw


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_bytes_handles_data_exceeding_stdin_buffer_limit(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "write-large.bin")
    payload = _random_bytes(TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE + 1024, seed=40)

    sandbox.filesystem.write_bytes(payload, remote_path)

    assert (tmp_path / "write-large.bin").read_bytes() == payload


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_bytes_errors_on_relative_remote_path(servicer, client, exec_backend, sandbox):
    with pytest.raises(InvalidError, match="absolute remote_path values"):
        sandbox.filesystem.write_bytes(b"payload", "relative/path.bin")


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_bytes_errors_on_unsupported_data_type(servicer, client, exec_backend, sandbox, tmp_path):
    with pytest.raises(TypeError):
        sandbox.filesystem.write_bytes("not-bytes", str(tmp_path / "unused.bin"))  # type: ignore[arg-type]


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_bytes_errors_when_remote_path_is_directory(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_dir = str(tmp_path / "write-dir")
    (tmp_path / "write-dir").mkdir()

    with pytest.raises(SandboxFilesystemIsADirectoryError):
        sandbox.filesystem.write_bytes(b"payload", remote_dir)


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_bytes_errors_when_parent_is_a_file(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"I am a file")
    remote_path = str(blocker / "child.txt")

    with pytest.raises(SandboxFilesystemNotADirectoryError):
        sandbox.filesystem.write_bytes(b"payload", remote_path)


# ---------------------------------------------------------------------------
# write_text
# ---------------------------------------------------------------------------


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_text_writes_expected_text(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "write-text.txt")
    text_payload = "hello from write_text\nsnowman: ☃\n"

    sandbox.filesystem.write_text(text_payload, remote_path)

    assert (tmp_path / "write-text.txt").read_bytes() == text_payload.encode("utf-8")


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_text_writes_empty_string(servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools):
    remote_path = str(tmp_path / "write-empty.txt")

    sandbox.filesystem.write_text("", remote_path)

    assert (tmp_path / "write-empty.txt").read_bytes() == b""


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_text_creates_parent_directories(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "nested" / "deep" / "write.txt")

    sandbox.filesystem.write_text("nested text", remote_path)

    assert (tmp_path / "nested" / "deep" / "write.txt").read_text(encoding="utf-8") == "nested text"


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_text_overwrites_existing_file(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "overwrite.txt")
    (tmp_path / "overwrite.txt").write_text("old-data", encoding="utf-8")

    sandbox.filesystem.write_text("new-data", remote_path)

    assert (tmp_path / "overwrite.txt").read_text(encoding="utf-8") == "new-data"


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_text_handles_data_exceeding_stdin_buffer_limit(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_path = str(tmp_path / "write-large.txt")
    # Generate text larger than the stdin buffer limit
    text_payload = "x" * (TASK_COMMAND_ROUTER_MAX_BUFFER_SIZE + 1024)

    sandbox.filesystem.write_text(text_payload, remote_path)

    assert (tmp_path / "write-large.txt").read_bytes() == text_payload.encode("utf-8")


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_text_errors_on_relative_remote_path(servicer, client, exec_backend, sandbox):
    with pytest.raises(InvalidError, match="absolute remote_path values"):
        sandbox.filesystem.write_text("data", "relative/path.txt")


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_text_errors_on_non_string_data(servicer, client, exec_backend, sandbox, tmp_path):
    with pytest.raises(TypeError):
        sandbox.filesystem.write_text(b"not-text", str(tmp_path / "unused.txt"))  # type: ignore[arg-type]


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_text_errors_when_remote_path_is_directory(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    remote_dir = str(tmp_path / "write-text-dir")
    (tmp_path / "write-text-dir").mkdir()

    with pytest.raises(SandboxFilesystemIsADirectoryError):
        sandbox.filesystem.write_text("data", remote_dir)


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_write_text_errors_when_parent_is_a_file(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    blocker = tmp_path / "blocker-text"
    blocker.write_bytes(b"I am a file")
    remote_path = str(blocker / "child.txt")

    with pytest.raises(SandboxFilesystemNotADirectoryError):
        sandbox.filesystem.write_text("data", remote_path)


# ---------------------------------------------------------------------------
# copy_from_local
# ---------------------------------------------------------------------------


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_from_local_writes_file_to_correct_remote_location(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    payload = _random_bytes(8192, seed=30)
    src = tmp_path / "source.bin"
    src.write_bytes(payload)
    remote_path = str(tmp_path / "copied.bin")

    sandbox.filesystem.copy_from_local(src, remote_path)

    assert (tmp_path / "copied.bin").read_bytes() == payload


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_from_local_copies_text_file(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    src = tmp_path / "source.txt"
    src.write_text("text content", encoding="utf-8")
    remote_path = str(tmp_path / "copied.txt")

    sandbox.filesystem.copy_from_local(src, remote_path)

    assert (tmp_path / "copied.txt").read_text(encoding="utf-8") == "text content"


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_from_local_copies_empty_file(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    src = tmp_path / "empty.bin"
    src.write_bytes(b"")
    remote_path = str(tmp_path / "copied-empty.bin")

    sandbox.filesystem.copy_from_local(src, remote_path)

    assert (tmp_path / "copied-empty.bin").read_bytes() == b""


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_from_local_errors_on_relative_remote_path(servicer, client, exec_backend, sandbox, tmp_path):
    src = tmp_path / "source.bin"
    src.write_bytes(b"data")

    with pytest.raises(InvalidError, match="absolute remote_path values"):
        sandbox.filesystem.copy_from_local(src, "relative/path.bin")


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_from_local_errors_when_local_path_missing(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    missing_local = tmp_path / "missing-local.bin"
    remote_path = str(tmp_path / "unused.bin")

    with pytest.raises(FileNotFoundError):
        sandbox.filesystem.copy_from_local(missing_local, remote_path)


@skip_non_subprocess
@pytest.mark.parametrize("exec_backend", ["router"], indirect=True)
def test_sandbox_fs_copy_from_local_errors_when_local_path_is_directory(
    servicer, client, exec_backend, sandbox, tmp_path, sandbox_fs_tools
):
    local_dir = tmp_path / "source-dir"
    local_dir.mkdir()
    remote_path = str(tmp_path / "unused.bin")

    with pytest.raises(IsADirectoryError):
        sandbox.filesystem.copy_from_local(local_dir, remote_path)
