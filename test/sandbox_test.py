# Copyright Modal Labs 2022

import hashlib
import platform
import pytest
import time
from pathlib import Path

from modal import App, Image, Mount, NetworkFileSystem, Sandbox, Secret
from modal.exception import DeprecationError, InvalidError

app = App()


skip_non_linux = pytest.mark.skipif(platform.system() != "Linux", reason="sandbox mock uses subprocess")


@skip_non_linux
def test_sandbox(client, servicer):
    sb = Sandbox.create("bash", "-c", "echo bye >&2 && sleep 1 && echo hi && exit 42", timeout=600, client=client)

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

    sb = Sandbox.create("echo", "hi", mounts=[Mount.from_local_dir(Path(tmpdir), remote_path="/m")], client=client)
    sb.wait()

    sha = hashlib.sha256(b"foo").hexdigest()
    assert servicer.files_sha2data[sha]["data"] == b"foo"


@skip_non_linux
def test_sandbox_image(client, servicer, tmpdir):
    tmpdir.join("a.py").write(b"foo")

    sb = Sandbox.create("echo", "hi", image=Image.debian_slim().pip_install("foo", "bar", "potato"), client=client)
    sb.wait()

    idx = max(servicer.images.keys())
    last_image = servicer.images[idx]

    assert all(c in last_image.dockerfile_commands[-1] for c in ["foo", "bar", "potato"])


@skip_non_linux
def test_sandbox_secret(client, servicer, tmpdir):
    sb = Sandbox.create("echo", "$FOO", secrets=[Secret.from_dict({"FOO": "BAR"})], client=client)
    sb.wait()

    assert len(servicer.sandbox_defs[0].secret_ids) == 1


@skip_non_linux
def test_sandbox_nfs(client, servicer, tmpdir):
    with NetworkFileSystem.ephemeral(client=client) as nfs:
        with pytest.raises(InvalidError):
            Sandbox.create("echo", "foo > /cache/a.txt", network_file_systems={"/": nfs}, client=client)

        Sandbox.create("echo", "foo > /cache/a.txt", network_file_systems={"/cache": nfs}, client=client)

    assert len(servicer.sandbox_defs[0].nfs_mounts) == 1


@skip_non_linux
def test_sandbox_from_id(client, servicer):
    sb = Sandbox.create("bash", "-c", "echo foo && exit 42", timeout=600, client=client)
    sb.wait()

    sb2 = Sandbox.from_id(sb.object_id, client=client)
    assert sb2.stdout.read() == "foo\n"
    assert sb2.returncode == 42


@skip_non_linux
def test_sandbox_terminate(client, servicer):
    sb = Sandbox.create("bash", "-c", "sleep 10000", client=client)
    sb.terminate()

    assert sb.returncode != 0


@skip_non_linux
@pytest.mark.asyncio
async def test_sandbox_stdin_async(client, servicer):
    sb = await Sandbox.create.aio("bash", "-c", "while read line; do echo $line; done && exit 13", client=client)

    sb.stdin.write(b"foo\n")
    sb.stdin.write(b"bar\n")

    sb.stdin.write_eof()

    await sb.stdin.drain.aio()

    await sb.wait.aio()

    assert await sb.stdout.read.aio() == "foo\nbar\n"
    assert sb.returncode == 13


@skip_non_linux
def test_sandbox_stdin(client, servicer):
    sb = Sandbox.create("bash", "-c", "while read line; do echo $line; done && exit 13", client=client)

    sb.stdin.write(b"foo\n")
    sb.stdin.write(b"bar\n")

    sb.stdin.write_eof()

    sb.stdin.drain()

    sb.wait()

    assert sb.stdout.read() == "foo\nbar\n"
    assert sb.returncode == 13


@skip_non_linux
def test_sandbox_stdin_invalid_write(client, servicer):
    sb = Sandbox.create("bash", "-c", "echo foo", client=client)
    with pytest.raises(TypeError):
        sb.stdin.write("foo\n")  # type: ignore


@skip_non_linux
def test_sandbox_stdin_write_after_eof(client, servicer):
    sb = Sandbox.create("bash", "-c", "echo foo", client=client)
    sb.stdin.write_eof()
    with pytest.raises(EOFError):
        sb.stdin.write(b"foo")


@skip_non_linux
@pytest.mark.asyncio
async def test_sandbox_async_for(client, servicer):
    sb = await Sandbox.create.aio("bash", "-c", "echo hello && echo world && echo bye >&2", client=client)

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
    assert await sb.stdout.read.aio() == ""
    assert await sb.stderr.read.aio() == ""


@skip_non_linux
def test_app_sandbox(client, servicer):
    image = Image.debian_slim().pip_install("xyz")
    secret = Secret.from_dict({"FOO": "bar"})
    mount = Mount.from_local_file(__file__, "/xyz")

    with app.run(client):
        # Create sandbox
        with pytest.warns(DeprecationError):
            sb = app.spawn_sandbox(
                "bash", "-c", "echo bye >&2 && echo hi", image=image, secrets=[secret], mounts=[mount]
            )
            assert sb.stdout.read() == "hi\n"
            assert sb.stderr.read() == "bye\n"
