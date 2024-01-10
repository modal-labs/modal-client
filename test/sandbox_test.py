# Copyright Modal Labs 2022

import hashlib
import platform
import pytest
import time
from pathlib import Path

from modal import Image, Mount, NetworkFileSystem, Sandbox, Secret, Stub
from modal.exception import InvalidError

stub = Stub()


skip_non_linux = pytest.mark.skipif(platform.system() != "Linux", reason="sandbox mock uses subprocess")


@skip_non_linux
def test_spawn_sandbox(client, servicer):
    with stub.run(client=client):
        sb = stub.spawn_sandbox("bash", "-c", "echo bye >&2 && sleep 1 && echo hi && exit 42", timeout=600)

        assert sb.poll() is None

        t0 = time.time()
        sb.wait()
        # Test that we actually waited for the sandbox to finish.
        assert time.time() - t0 > 0.3

        assert sb.stdout.read() == "hi\n"
        assert sb.stderr.read() == "bye\n"

        assert sb.returncode == 42
        assert sb.poll() == 42


@skip_non_linux
def test_sandbox_mount(client, servicer, tmpdir):
    tmpdir.join("a.py").write(b"foo")

    with stub.run(client=client):
        sb = stub.spawn_sandbox(
            "echo",
            "hi",
            mounts=[Mount.from_local_dir(Path(tmpdir), remote_path="/m")],
        )
        sb.wait()

    sha = hashlib.sha256(b"foo").hexdigest()
    assert servicer.files_sha2data[sha]["data"] == b"foo"


@skip_non_linux
def test_sandbox_image(client, servicer, tmpdir):
    tmpdir.join("a.py").write(b"foo")

    with stub.run(client=client):
        sb = stub.spawn_sandbox("echo", "hi", image=Image.debian_slim().pip_install("foo", "bar", "potato"))
        sb.wait()

    idx = max(servicer.images.keys())
    last_image = servicer.images[idx]

    assert all(c in last_image.dockerfile_commands[-1] for c in ["foo", "bar", "potato"])


@skip_non_linux
def test_sandbox_secret(client, servicer, tmpdir):
    with stub.run(client=client):
        sb = stub.spawn_sandbox("echo", "$FOO", secrets=[Secret.from_dict({"FOO": "BAR"})])
        sb.wait()

    assert len(servicer.sandbox_defs[0].secret_ids) == 1


@skip_non_linux
def test_sandbox_nfs(client, servicer, tmpdir):
    with stub.run(client=client):
        nfs = NetworkFileSystem.new()

        with pytest.raises(InvalidError):
            stub.spawn_sandbox("echo", "foo > /cache/a.txt", network_file_systems={"/": nfs})

        stub.spawn_sandbox("echo", "foo > /cache/a.txt", network_file_systems={"/cache": nfs})

    assert len(servicer.sandbox_defs[0].nfs_mounts) == 1


@skip_non_linux
def test_sandbox_from_id(client, servicer):
    with stub.run(client=client):
        sb = stub.spawn_sandbox("bash", "-c", "echo foo && exit 42", timeout=600)
        sb.wait()

    sb2 = Sandbox.from_id(sb.object_id, client=client)
    assert sb2.stdout.read() == "foo\n"
    assert sb2.returncode == 42


@skip_non_linux
def test_sandbox_terminate(client, servicer):
    with stub.run(client=client):
        sb = stub.spawn_sandbox("bash", "-c", "sleep 10000")

        sb.terminate()

        assert sb.returncode != 0
