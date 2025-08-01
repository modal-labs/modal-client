# Copyright Modal Labs 2022

import hashlib
import pytest
import time
from pathlib import Path
from unittest import mock

from modal import App, Image, NetworkFileSystem, Proxy, Sandbox, SandboxSnapshot, Secret, Volume
from modal.exception import InvalidError
from modal.stream_type import StreamType
from modal_proto import api_pb2

from .supports.skip import skip_windows

skip_non_subprocess = skip_windows("Needs subprocess support")


@pytest.fixture
def app(client):
    app = App()
    with app.run(client=client):
        yield app


@skip_non_subprocess
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


@skip_non_subprocess
def test_sandbox_mount_layer(app, servicer, tmpdir):
    tmpdir.join("a.py").write(b"foo")

    sb = Sandbox.create("echo", "hi", image=Image.debian_slim().add_local_dir(Path(tmpdir), remote_path="/m"), app=app)
    sb.wait()

    sha = hashlib.sha256(b"foo").hexdigest()
    assert servicer.files_sha2data[sha]["data"] == b"foo"


@skip_non_subprocess
def test_sandbox_image(app, servicer, tmpdir):
    tmpdir.join("a.py").write(b"foo")

    sb = Sandbox.create("echo", "hi", image=Image.debian_slim().pip_install("foo", "bar", "potato"), app=app)
    sb.wait()

    idx = max(servicer.images.keys())
    last_image = servicer.images[idx]

    assert all(c in last_image.dockerfile_commands[-1] for c in ["foo", "bar", "potato"])


@skip_non_subprocess
def test_sandbox_secret(app, servicer, tmpdir):
    sb = Sandbox.create("echo", "$FOO", secrets=[Secret.from_dict({"FOO": "BAR"})], app=app)
    sb.wait()

    assert len(servicer.sandbox_defs[0].secret_ids) == 1


@skip_non_subprocess
def test_sandbox_nfs(client, app, servicer, tmpdir):
    with NetworkFileSystem.ephemeral(client=client) as nfs:
        with pytest.raises(InvalidError):
            Sandbox.create("echo", "foo > /cache/a.txt", network_file_systems={"/": nfs}, app=app)

        Sandbox.create("echo", "foo > /cache/a.txt", network_file_systems={"/cache": nfs}, app=app)

    assert len(servicer.sandbox_defs[0].nfs_mounts) == 1


@skip_non_subprocess
def test_sandbox_from_id(app, client, servicer):
    sb = Sandbox.create("bash", "-c", "echo foo && exit 42", timeout=600, app=app)
    sb.wait()

    sb2 = Sandbox.from_id(sb.object_id, client=client)
    assert sb2.stdout.read() == "foo\n"
    assert sb2.returncode == 42


@skip_non_subprocess
def test_sandbox_terminate(app, servicer):
    sb = Sandbox.create("bash", "-c", "sleep 10000", app=app)
    sb.terminate()

    assert sb.returncode != 0


@skip_non_subprocess
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


@skip_non_subprocess
def test_sandbox_stdin(app, servicer):
    sb = Sandbox.create("bash", "-c", "while read line; do echo $line; done && exit 13", app=app)

    sb.stdin.write(b"foo\n")
    sb.stdin.write(b"bar\n")

    sb.stdin.write_eof()

    sb.stdin.drain()

    sb.wait()

    assert sb.stdout.read() == "foo\nbar\n"
    assert sb.returncode == 13


@skip_non_subprocess
def test_sandbox_stdin_write_str(app, servicer):
    sb = Sandbox.create("bash", "-c", "while read line; do echo $line; done && exit 13", app=app)

    sb.stdin.write("foo\n")
    sb.stdin.write("bar\n")

    sb.stdin.write_eof()

    sb.stdin.drain()

    sb.wait()

    assert sb.stdout.read() == "foo\nbar\n"
    assert sb.returncode == 13


@skip_non_subprocess
def test_sandbox_stdin_write_after_terminate(app, servicer):
    sb = Sandbox.create("bash", "-c", "echo foo", app=app)
    sb.wait()
    with pytest.raises(ValueError):
        sb.stdin.write(b"foo")
        sb.stdin.drain()


@skip_non_subprocess
def test_sandbox_stdin_write_after_eof(app, servicer):
    sb = Sandbox.create(app=app)
    sb.stdin.write_eof()
    with pytest.raises(ValueError):
        sb.stdin.write(b"foo")
    sb.terminate()


@skip_non_subprocess
def test_sandbox_stdout(app, servicer):
    """Test that reads from sandboxes are fully line-buffered, i.e.,
    that we don't read partial lines or multiple lines at once."""

    # normal sequence of reads
    sb = Sandbox.create("bash", "-c", "for i in $(seq 1 3); do echo foo $i; done", app=app)
    out = []
    for line in sb.stdout:
        out.append(line)
    assert out == ["foo 1\n", "foo 2\n", "foo 3\n"]

    # multiple newlines
    sb = Sandbox.create("bash", "-c", "echo 'foo 1\nfoo 2\nfoo 3'", app=app)
    out = []
    for line in sb.stdout:
        out.append(line)
    assert out == ["foo 1\n", "foo 2\n", "foo 3\n"]

    # partial lines
    sb = Sandbox.create("sleep", "infinity", app=app)
    cp = sb.exec("bash", "-c", "while read line; do echo $line; done")

    cp.stdin.write(b"foo 1\n")
    cp.stdin.write(b"foo 2")
    cp.stdin.write(b"foo 3\n")
    cp.stdin.write_eof()
    cp.stdin.drain()

    assert cp.stdout.read() == "foo 1\nfoo 2foo 3\n"


@skip_non_subprocess
def test_sandbox_stdout_next(app, servicer):
    """Test that we can iterate on a StreamReader directly, without a call to __aiter__()."""

    # normal sequence of reads
    sb = Sandbox.create("bash", "-c", "for i in $(seq 1 3); do echo foo $i; done", app=app)
    line = next(sb.stdout)
    assert line == "foo 1\n"
    line = next(sb.stdout)
    assert line == "foo 2\n"
    line = next(sb.stdout)
    assert line == "foo 3\n"


@skip_non_subprocess
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


@skip_non_subprocess
def test_sandbox_exec_stdout_bytes_mode(app, servicer):
    """Test that the stream reader works in bytes mode."""

    sb = Sandbox.create(app=app)

    p = sb.exec("echo", "foo", text=False)
    assert p.stdout.read() == b"foo\n"

    p = sb.exec("echo", "foo", text=False)
    for line in p.stdout:
        assert line == b"foo\n"


@skip_non_subprocess
def test_app_sandbox(client, servicer):
    image = Image.debian_slim().pip_install("xyz").add_local_file(__file__, remote_path="/xyz")
    secret = Secret.from_dict({"FOO": "bar"})

    with pytest.raises(InvalidError, match="require an App"):
        Sandbox.create("bash", "-c", "echo bye >&2 && echo hi", image=image, secrets=[secret])

    app = App()
    with app.run(client=client):
        # Create sandbox
        sb = Sandbox.create("bash", "-c", "echo bye >&2 && echo hi", image=image, secrets=[secret], app=app)
        sb.wait()
        assert sb.stderr.read() == "bye\n"
        assert sb.stdout.read() == "hi\n"


@skip_non_subprocess
def test_sandbox_exec(app, servicer):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("bash", "-c", "while read line; do echo $line; done")
    assert str(cp) == "ContainerProcess(process_id='container_exec_id')"

    cp.stdin.write(b"foo\n")
    cp.stdin.write(b"bar\n")
    cp.stdin.write_eof()
    cp.stdin.drain()

    assert cp.stdout.read() == "foo\nbar\n"


@skip_non_subprocess
def test_sandbox_exec_wait(app, servicer):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("bash", "-c", "sleep 0.5 && exit 42")

    assert cp.poll() is None

    t0 = time.time()
    assert cp.wait() == 42
    assert time.time() - t0 > 0.2

    assert cp.poll() == 42


@mock.patch("modal.sandbox.CONTAINER_EXEC_TIMEOUT_BUFFER", 0)
@skip_non_subprocess
def test_sandbox_exec_wait_timeout(app, servicer):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("sleep", "999", timeout=1)
    t0 = time.monotonic()
    assert cp.wait() == -1
    assert 0.8 < time.monotonic() - t0 <= 1.2


@mock.patch("modal.sandbox.CONTAINER_EXEC_TIMEOUT_BUFFER", 0)
@skip_non_subprocess
def test_sandbox_exec_poll_timeout(app, servicer):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("sleep", "999", timeout=1)
    assert not cp.poll()
    time.sleep(1.2)
    assert cp.poll() == -1


@mock.patch("modal.sandbox.CONTAINER_EXEC_TIMEOUT_BUFFER", 0)
@skip_non_subprocess
def test_sandbox_exec_output_timeout(app, servicer):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("sh", "-c", "echo hi; sleep 999", timeout=1)
    t1 = time.monotonic()
    assert cp.stdout.read() == "hi\n"
    assert 1 < time.monotonic() - t1 < 2.0
    assert cp.wait() == -1


@skip_non_subprocess
def test_sandbox_exec_output_double_read(app, servicer):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("sh", "-c", "echo hi")
    assert cp.stdout.read() == "hi\n"
    assert cp.stdout.read() == ""
    assert cp.wait() == 0


@skip_non_subprocess
def test_sandbox_create_and_exec_with_bad_args(app, servicer):
    too_big = 130_000
    single_arg_size = too_big // 10
    too_big_args = ["a" * single_arg_size for _ in range(10)]
    with pytest.raises(InvalidError):
        Sandbox.create(*too_big_args, app=app)

    sb = Sandbox.create("sleep", "infinity", app=app)
    with pytest.raises(InvalidError):
        sb.exec("echo", 1)  # type: ignore

    with pytest.raises(InvalidError):
        sb.exec(*too_big_args)


@skip_non_subprocess
def test_sandbox_on_app_lookup(client, servicer):
    app = App.lookup("my-app", create_if_missing=True, client=client)
    sb = Sandbox.create("echo", "hi", app=app)
    sb.wait()
    assert sb.stdout.read() == "hi\n"
    assert servicer.sandbox_app_id == app.app_id


@skip_non_subprocess
def test_sandbox_list_env(app, client, servicer):
    sb = Sandbox.create("bash", "-c", "sleep 10000", app=app)
    assert len(list(Sandbox.list(client=client))) == 1
    sb.terminate()
    sb.wait(raise_on_termination=False)
    assert not list(Sandbox.list(client=client))


@skip_non_subprocess
def test_sandbox_list_app(client, servicer):
    image = Image.debian_slim().pip_install("xyz").add_local_file(__file__, "/xyz")
    secret = Secret.from_dict({"FOO": "bar"})

    app = App()

    with app.run(client=client):
        # Create sandbox
        sb = Sandbox.create("bash", "-c", "sleep 10000", image=image, secrets=[secret], app=app)
        assert len(list(Sandbox.list(app_id=app.app_id, client=client))) == 1
        sb.terminate()
        sb.wait(raise_on_termination=False)
        assert not list(Sandbox.list(app_id=app.app_id, client=client))


@skip_non_subprocess
def test_sandbox_list_tags(app, client, servicer):
    sb = Sandbox.create("bash", "-c", "sleep 10000", app=app)
    sb.set_tags({"foo": "bar", "baz": "qux"}, client=client)
    assert len(list(Sandbox.list(tags={"foo": "bar"}, client=client))) == 1
    assert not list(Sandbox.list(tags={"foo": "notbar"}, client=client))
    sb.terminate()
    sb.wait(raise_on_termination=False)
    assert not list(Sandbox.list(tags={"baz": "qux"}, client=client))


@skip_non_subprocess
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


@skip_non_subprocess
def test_sandbox_no_entrypoint(app, servicer):
    sb = Sandbox.create(app=app)

    p = sb.exec("echo", "hi")
    p.wait()
    assert p.returncode == 0
    assert p.stdout.read() == "hi\n"

    sb.terminate()


@skip_non_subprocess
def test_sandbox_gpu_fallbacks_support(client, servicer):
    with pytest.raises(InvalidError, match="do not support"):
        Sandbox.create(client=client, gpu=["t4", "a100"])  # type: ignore


@skip_non_subprocess
def test_sandbox_exec_stdout(app, servicer, capsys):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("bash", "-c", "echo hi", stdout=StreamType.STDOUT)
    cp.wait()

    assert capsys.readouterr().out == "hi\n"

    with pytest.raises(InvalidError):
        cp.stdout.read()


@skip_non_subprocess
def test_sandbox_snapshot(app, client, servicer):
    sb = Sandbox.create(app=app, _experimental_enable_snapshot=True)
    sandbox_snapshot = sb._experimental_snapshot()
    snapshot_id = sandbox_snapshot.object_id
    assert snapshot_id == "sn-123"
    sb.terminate()

    sandbox_snapshot = SandboxSnapshot.from_id(snapshot_id, client=client)
    assert sandbox_snapshot.object_id == snapshot_id

    sb = Sandbox._experimental_from_snapshot(sandbox_snapshot, client=client)
    sb.terminate()


@skip_non_subprocess
def test_sandbox_snapshot_fs(app, servicer):
    sb = Sandbox.create(app=app)
    image = sb.snapshot_filesystem()
    sb.terminate()

    sb2 = Sandbox.create(image=image, app=app)
    sb2.terminate()

    assert image.object_id == "im-123"
    assert servicer.sandbox_defs[1].image_id == "im-123"


@skip_non_subprocess
def test_sandbox_cpu_request(app, servicer):
    _ = Sandbox.create(cpu=2.0, app=app)

    assert servicer.sandbox_defs[0].resources.milli_cpu == 2000
    assert servicer.sandbox_defs[0].resources.milli_cpu_max == 0


@skip_non_subprocess
def test_sandbox_cpu_limit(app, servicer):
    _ = Sandbox.create(cpu=(2, 4), app=app)

    assert servicer.sandbox_defs[0].resources.milli_cpu == 2000
    assert servicer.sandbox_defs[0].resources.milli_cpu_max == 4000


@skip_non_subprocess
def test_sandbox_proxy(app, servicer):
    _ = Sandbox.create(proxy=Proxy.from_name("my-proxy"), app=app)

    assert servicer.sandbox_defs[0].proxy_id == "pr-123"


def test_sandbox_list_sets_correct_returncode_for_running(client, servicer):
    with servicer.intercept() as ctx:
        # test generic status
        ctx.add_response(
            "SandboxList",
            api_pb2.SandboxListResponse(
                sandboxes=[
                    api_pb2.SandboxInfo(
                        id="sb-123",
                        task_info=api_pb2.TaskInfo(
                            result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_UNSPECIFIED)
                        ),
                    )
                ]
            ),
        )
        ctx.add_response(
            "SandboxList", api_pb2.SandboxListResponse(sandboxes=[])
        )  # list will loop for older sandboxes until no more arrive
        (list_result,) = list(Sandbox.list(client=client))
    assert list_result.returncode is None


def test_sandbox_list_sets_correct_returncode_for_stopped(client, servicer):
    with servicer.intercept() as ctx:
        # test generic status
        ctx.add_response(
            "SandboxList",
            api_pb2.SandboxListResponse(
                sandboxes=[
                    api_pb2.SandboxInfo(
                        id="sb-123",
                        task_info=api_pb2.TaskInfo(
                            result=api_pb2.GenericResult(
                                status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS, exitcode=0
                            )
                        ),
                    )
                ]
            ),
        )
        ctx.add_response(
            "SandboxList", api_pb2.SandboxListResponse(sandboxes=[])
        )  # list will loop for older sandboxes until no more arrive
        (list_result,) = list(Sandbox.list(client=client))
    assert list_result.returncode == 0


@pytest.mark.parametrize("read_only", [True, False])
@skip_non_subprocess
def test_sandbox_volume(app, servicer, read_only):
    volume = Volume.from_name("my-volume", create_if_missing=True)

    if read_only:
        volume = volume.read_only()

    with servicer.intercept() as ctx:
        Sandbox.create(
            "bash",
            "-c",
            "echo bye >&2 && sleep 1 && echo hi && exit 42",
            timeout=600,
            app=app,
            volumes={"/mnt": volume},
        )
        req = ctx.pop_request("SandboxCreate")
        assert req.definition.volume_mounts[0].read_only == read_only
