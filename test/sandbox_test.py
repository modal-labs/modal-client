# Copyright Modal Labs 2022

import hashlib
import platform
import pytest
import time
from pathlib import Path

from modal import App, Image, Mount, NetworkFileSystem, Sandbox, Secret
from modal.exception import DeprecationError, InvalidError
from modal_proto import api_pb2

skip_non_linux = pytest.mark.skipif(platform.system() != "Linux", reason="sandbox mock uses subprocess")


@pytest.fixture
def app(client):
    app = App()
    with app.run(client):
        yield app


@skip_non_linux
def test_sandbox(app, servicer):
    sb = Sandbox.create("bash", "-c", "echo bye >&2 && sleep 1 && echo hi && exit 42", timeout=600, app=app)

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
def test_sandbox_mount(app, servicer, tmpdir):
    tmpdir.join("a.py").write(b"foo")

    sb = Sandbox.create("echo", "hi", mounts=[Mount.from_local_dir(Path(tmpdir), remote_path="/m")], app=app)
    sb.wait()

    sha = hashlib.sha256(b"foo").hexdigest()
    assert servicer.files_sha2data[sha]["data"] == b"foo"


@skip_non_linux
def test_sandbox_image(app, servicer, tmpdir):
    tmpdir.join("a.py").write(b"foo")

    sb = Sandbox.create("echo", "hi", image=Image.debian_slim().pip_install("foo", "bar", "potato"), app=app)
    sb.wait()

    idx = max(servicer.images.keys())
    last_image = servicer.images[idx]

    assert all(c in last_image.dockerfile_commands[-1] for c in ["foo", "bar", "potato"])


@skip_non_linux
def test_sandbox_secret(app, servicer, tmpdir):
    sb = Sandbox.create("echo", "$FOO", secrets=[Secret.from_dict({"FOO": "BAR"})], app=app)
    sb.wait()

    assert len(servicer.sandbox_defs[0].secret_ids) == 1


@skip_non_linux
def test_sandbox_nfs(client, app, servicer, tmpdir):
    with NetworkFileSystem.ephemeral(client=client) as nfs:
        with pytest.raises(InvalidError):
            Sandbox.create("echo", "foo > /cache/a.txt", network_file_systems={"/": nfs}, app=app)

        Sandbox.create("echo", "foo > /cache/a.txt", network_file_systems={"/cache": nfs}, app=app)

    assert len(servicer.sandbox_defs[0].nfs_mounts) == 1


@skip_non_linux
def test_sandbox_from_id(app, client, servicer):
    sb = Sandbox.create("bash", "-c", "echo foo && exit 42", timeout=600, app=app)
    sb.wait()

    sb2 = Sandbox.from_id(sb.object_id, client=client)
    assert sb2.stdout.read() == "foo\n"
    assert sb2.returncode == 42


@skip_non_linux
def test_sandbox_terminate(app, servicer):
    sb = Sandbox.create("bash", "-c", "sleep 10000", app=app)
    sb.terminate()

    assert sb.returncode != 0


@skip_non_linux
@pytest.mark.asyncio
async def test_sandbox_stdin_async(app, servicer):
    sb = await Sandbox.create.aio("bash", "-c", "while read line; do echo $line; done && exit 13", app=app)

    sb.stdin.write(b"foo\n")
    sb.stdin.write(b"bar\n")

    sb.stdin.write_eof()

    await sb.stdin.drain.aio()

    await sb.wait.aio()

    assert await sb.stdout.read.aio() == "foo\nbar\n"
    assert sb.returncode == 13


@skip_non_linux
def test_sandbox_stdin(app, servicer):
    sb = Sandbox.create("bash", "-c", "while read line; do echo $line; done && exit 13", app=app)

    sb.stdin.write(b"foo\n")
    sb.stdin.write(b"bar\n")

    sb.stdin.write_eof()

    sb.stdin.drain()

    sb.wait()

    assert sb.stdout.read() == "foo\nbar\n"
    assert sb.returncode == 13


@skip_non_linux
def test_sandbox_stdin_invalid_write(app, servicer):
    sb = Sandbox.create("bash", "-c", "echo foo", app=app)
    with pytest.raises(TypeError):
        sb.stdin.write("foo\n")  # type: ignore


@skip_non_linux
def test_sandbox_stdin_write_after_eof(app, servicer):
    sb = Sandbox.create("bash", "-c", "echo foo", app=app)
    sb.stdin.write_eof()
    with pytest.raises(EOFError):
        sb.stdin.write(b"foo")


@skip_non_linux
@pytest.mark.asyncio
async def test_sandbox_async_for(app, servicer):
    sb = await Sandbox.create.aio("bash", "-c", "echo hello && echo world && echo bye >&2", app=app)

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

    app = App()
    with app.run(client):
        # Create sandbox
        with pytest.warns(DeprecationError):
            sb = app.spawn_sandbox(
                "bash", "-c", "echo bye >&2 && echo hi", image=image, secrets=[secret], mounts=[mount]
            )

        sb = Sandbox.create(
            "bash", "-c", "echo bye >&2 && echo hi", image=image, secrets=[secret], mounts=[mount], app=app
        )
        sb.wait()
        assert sb.stderr.read() == "bye\n"
        assert sb.stdout.read() == "hi\n"


@skip_non_linux
def test_sandbox_exec(app, servicer):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("bash", "-c", "while read line; do echo $line; done")

    cp.stdin.write(b"foo\n")
    cp.stdin.write(b"bar\n")
    cp.stdin.write_eof()
    cp.stdin.drain()

    assert cp.stdout.read() == "foo\nbar\n"


@skip_non_linux
def test_sandbox_exec_wait(app, servicer):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("bash", "-c", "sleep 0.5 && exit 42")

    assert cp.poll() is None

    t0 = time.time()
    assert cp.wait() == 42
    assert time.time() - t0 > 0.2

    assert cp.poll() == 42


@skip_non_linux
def test_sandbox_on_app_lookup(client, servicer):
    app = App.lookup("my-app", create_if_missing=True, client=client)
    sb = Sandbox.create("echo", "hi", app=app)
    sb.wait()
    assert sb.stdout.read() == "hi\n"
    assert servicer.sandbox_app_id == app.app_id


@skip_non_linux
def test_sandbox_list_env(app, client, servicer):
    sb = Sandbox.create("bash", "-c", "sleep 10000", app=app)
    assert len(list(Sandbox.list(client=client))) == 1
    sb.terminate()
    assert not list(Sandbox.list(client=client))


@skip_non_linux
def test_sandbox_list_app(client, servicer):
    image = Image.debian_slim().pip_install("xyz")
    secret = Secret.from_dict({"FOO": "bar"})
    mount = Mount.from_local_file(__file__, "/xyz")

    app = App()

    with app.run(client):
        # Create sandbox
        sb = Sandbox.create("bash", "-c", "sleep 10000", image=image, secrets=[secret], mounts=[mount], app=app)
        assert len(list(Sandbox.list(app_id=app.app_id, client=client))) == 1
        sb.terminate()
        assert not list(Sandbox.list(app_id=app.app_id, client=client))


@skip_non_linux
def test_sandbox_list_tags(app, client, servicer):
    sb = Sandbox.create("bash", "-c", "sleep 10000", app=app)
    sb.set_tags({"foo": "bar", "baz": "qux"}, client=client)
    assert len(list(Sandbox.list(tags={"foo": "bar"}, client=client))) == 1
    assert not list(Sandbox.list(tags={"foo": "notbar"}, client=client))
    sb.terminate()
    assert not list(Sandbox.list(tags={"baz": "qux"}, client=client))


@skip_non_linux
def test_sandbox_network_access(app, servicer):
    with pytest.raises(InvalidError):
        Sandbox.create("echo", "test", block_network=True, cidr_allowlist=["10.0.0.0/8"], app=app)

    # Test that blocking works
    sb = Sandbox.create("echo", "test", block_network=True, app=app)
    assert (
        servicer.sandbox_defs[0].network_access.network_access_type == api_pb2.NetworkAccess.NetworkAccessType.BLOCKED
    )
    assert len(servicer.sandbox_defs[0].network_access.allowed_cidrs) == 0
    sb.terminate()

    # Test that allowlisting works
    sb = Sandbox.create("echo", "test", block_network=False, cidr_allowlist=["10.0.0.0/8"], app=app)
    assert (
        servicer.sandbox_defs[1].network_access.network_access_type == api_pb2.NetworkAccess.NetworkAccessType.ALLOWLIST
    )
    assert len(servicer.sandbox_defs[1].network_access.allowed_cidrs) == 1
    assert servicer.sandbox_defs[1].network_access.allowed_cidrs[0] == "10.0.0.0/8"
    sb.terminate()

    # Test that no rules means allow all
    sb = Sandbox.create("echo", "test", block_network=False, app=app)
    assert servicer.sandbox_defs[2].network_access.network_access_type == api_pb2.NetworkAccess.NetworkAccessType.OPEN
    assert len(servicer.sandbox_defs[2].network_access.allowed_cidrs) == 0
    sb.terminate()


@skip_non_linux
def test_sandbox_no_entrypoint(app, servicer):
    sb = Sandbox.create(app=app)

    p = sb.exec("echo", "hi")
    p.wait()
    assert p.returncode == 0
    assert p.stdout.read() == "hi\n"

    sb.terminate()


@skip_non_linux
def test_sandbox_gpu_fallbacks_support(client, servicer):
    with pytest.raises(InvalidError, match="do not support"):
        Sandbox.create(client=client, gpu=["t4", "a100"])  # type: ignore
