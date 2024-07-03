# Copyright Modal Labs 2022

import hashlib
import platform
import pytest
import time
from pathlib import Path

from modal import App, Image, Mount, NetworkFileSystem, Sandbox, Secret
from modal.exception import InvalidError

app = App()


skip_non_linux = pytest.mark.skipif(platform.system() != "Linux", reason="sandbox mock uses subprocess")


@skip_non_linux
def test_spawn_sandbox(client, servicer):
    with app.run(client=client):
        sb = app.spawn_sandbox("bash", "-c", "echo bye >&2 && sleep 1 && echo hi && exit 42", timeout=600)

        assert sb.poll() is None

        t0 = time.time()
        sb.wait()
        # Test that we actually waited for the sandbox to finish.
        assert time.time() - t0 > 0.3

        assert sb.stdout.read() == "hi\n"
        assert sb.stderr.read() == "bye\n"
        # read a second time
        assert sb.stdout.read() == ""
        assert sb.stderr.read() == ""

        assert sb.returncode == 42
        assert sb.poll() == 42


@skip_non_linux
def test_sandbox_mount(client, servicer, tmpdir):
    tmpdir.join("a.py").write(b"foo")

    with app.run(client=client):
        sb = app.spawn_sandbox(
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

    with app.run(client=client):
        sb = app.spawn_sandbox("echo", "hi", image=Image.debian_slim().pip_install("foo", "bar", "potato"))
        sb.wait()

    idx = max(servicer.images.keys())
    last_image = servicer.images[idx]

    assert all(c in last_image.dockerfile_commands[-1] for c in ["foo", "bar", "potato"])


@skip_non_linux
def test_sandbox_secret(client, servicer, tmpdir):
    with app.run(client=client):
        sb = app.spawn_sandbox("echo", "$FOO", secrets=[Secret.from_dict({"FOO": "BAR"})])
        sb.wait()

    assert len(servicer.sandbox_defs[0].secret_ids) == 1


@skip_non_linux
def test_sandbox_nfs(client, servicer, tmpdir):
    with app.run(client=client):
        with NetworkFileSystem.ephemeral(client=client) as nfs:
            with pytest.raises(InvalidError):
                app.spawn_sandbox("echo", "foo > /cache/a.txt", network_file_systems={"/": nfs})

            app.spawn_sandbox("echo", "foo > /cache/a.txt", network_file_systems={"/cache": nfs})

    assert len(servicer.sandbox_defs[0].nfs_mounts) == 1


@skip_non_linux
def test_sandbox_from_id(client, servicer):
    with app.run(client=client):
        sb = app.spawn_sandbox("bash", "-c", "echo foo && exit 42", timeout=600)
        sb.wait()

    sb2 = Sandbox.from_id(sb.object_id, client=client)
    assert sb2.stdout.read() == "foo\n"
    assert sb2.returncode == 42


@skip_non_linux
def test_sandbox_terminate(client, servicer):
    with app.run(client=client):
        sb = app.spawn_sandbox("bash", "-c", "sleep 10000")
        sb.terminate()

        assert sb.returncode != 0


@skip_non_linux
@pytest.mark.asyncio
async def test_sandbox_stdin_async(client, servicer):
    async with app.run.aio(client=client):
        sb = app.spawn_sandbox("bash", "-c", "while read line; do echo $line; done && exit 13")

        sb.stdin.write(b"foo\n")
        sb.stdin.write(b"bar\n")

        sb.stdin.write_eof()

        await sb.stdin.drain.aio()

        sb.wait()

        assert sb.stdout.read() == "foo\nbar\n"
        assert sb.returncode == 13


@skip_non_linux
def test_sandbox_stdin(client, servicer):
    with app.run(client=client):
        sb = app.spawn_sandbox("bash", "-c", "while read line; do echo $line; done && exit 13")

        sb.stdin.write(b"foo\n")
        sb.stdin.write(b"bar\n")

        sb.stdin.write_eof()

        sb.stdin.drain()

        sb.wait()

        assert sb.stdout.read() == "foo\nbar\n"
        assert sb.returncode == 13


@skip_non_linux
def test_sandbox_stdin_invalid_write(client, servicer):
    with app.run(client=client):
        sb = app.spawn_sandbox("bash", "-c", "echo foo")
        with pytest.raises(TypeError):
            sb.stdin.write("foo\n")  # type: ignore


@skip_non_linux
def test_sandbox_stdin_write_after_eof(client, servicer):
    with app.run(client=client):
        sb = app.spawn_sandbox("bash", "-c", "echo foo")
        sb.stdin.write_eof()
        with pytest.raises(EOFError):
            sb.stdin.write(b"foo")


@skip_non_linux
@pytest.mark.asyncio
async def test_sandbox_async_for(client, servicer):
    async with app.run.aio(client=client):
        sb = app.spawn_sandbox("bash", "-c", "echo hello && echo world && echo bye >&2")

        out = ""

        async for message in sb.stdout:
            out += message
        assert out == "hello\nworld\n"

        # test streaming stdout a second time
        out2 = ""
        async for message in sb.stdout:
            out2 += message
        assert out2 == ""

        err = ""
        async for message in sb.stderr:
            err += message

        assert err == "bye\n"

        # test reading after receiving EOF
        assert sb.stdout.read() == ""
        assert sb.stderr.read() == ""


@skip_non_linux
def test_appless_sandbox(client, servicer):
    # Appless dependencies
    image = Image.debian_slim().pip_install("xyz")
    secret = Secret.from_dict({"FOO": "bar"})
    mount = Mount.from_local_file(__file__, "/xyz")

    # Create sandbox
    sb = Sandbox.create(
        "bash", "-c", "echo bye >&2 && echo hi", image=image, secrets=[secret], mounts=[mount], client=client
    )
    assert sb.stdout.read() == "hi\n"
    assert sb.stderr.read() == "bye\n"

    # Make sure ids got assigned
    assert image.object_id == "im-2"
    assert secret.object_id == "st-0"
    assert mount.object_id == "mo-1"
