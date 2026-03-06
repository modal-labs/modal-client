# Copyright Modal Labs 2026
import pathlib
import pytest
import random

from modal import App, Sandbox
from modal.exception import (
    InvalidError,
    SandboxFilesystemIsADirectoryError,
    SandboxFilesystemNotFoundError,
)

from .supports.skip import skip_windows

skip_non_subprocess = skip_windows("Needs subprocess support")

_MOCK_SANDBOX_FS_TOOLS_PATH = pathlib.Path(__file__).parent / "supports" / "mock_sandbox_fs_tools.py"


def _random_bytes(size: int, *, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(size))


@pytest.fixture
def sandbox_fs_tools(monkeypatch):
    """Points the sandbox filesystem tools path at the mock script.

    The mock exec backend ("router") runs commands as local subprocesses,
    so this script is executed directly on the host. It emulates the Rust
    binary's behavior for ReadFile, including structured JSON error
    payloads on stderr.
    """
    _MOCK_SANDBOX_FS_TOOLS_PATH.chmod(0o755)
    monkeypatch.setattr("modal.sandbox_fs._SANDBOX_FS_TOOLS_PATH", str(_MOCK_SANDBOX_FS_TOOLS_PATH))


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
