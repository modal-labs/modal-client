# Copyright Modal Labs 2022
import hashlib
import inspect
import pytest
import time
import typing
import uuid
from pathlib import Path
from unittest import mock

from grpclib import GRPCError, Status

import modal
from modal import (
    App,
    Image,
    NetworkFileSystem,
    Proxy,
    Sandbox,
    SandboxSnapshot,
    Secret,
    Volume,
)
from modal._utils.async_utils import synchronizer
from modal.exception import AlreadyExistsError, ConflictError, DeprecationError, InvalidError, TimeoutError
from modal.sandbox import SandboxVersion, SidecarContainer, _get_sandbox_version
from modal.stream_type import StreamType
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from .supports.skip import skip_windows

skip_non_subprocess = skip_windows("Needs subprocess support")


@pytest.fixture
def app(client):
    app = App()
    with app.run(client=client):
        yield app


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"port": "8080"}, "expects an integer"),
        ({"port": 0}, "expects `port` in \\[1, 65535\\]"),
        ({"port": 65536}, "expects `port` in \\[1, 65535\\]"),
        ({"port": 8080, "interval_ms": "100"}, "expects an integer"),
        ({"port": 8080, "interval_ms": 0}, "expects `interval_ms` > 0"),
    ],
)
def test_probe_tcp_bad_values_raise_invalid_error(kwargs, match):
    with pytest.raises(InvalidError, match=match):
        modal.Probe.with_tcp(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((), {}, "requires at least one argument"),
        (("echo", 1), {}, "expects all arguments to be strings"),
        (("echo",), {"interval_ms": "100"}, "expects an integer"),
        (("echo",), {"interval_ms": 0}, "expects `interval_ms` > 0"),
    ],
)
def test_probe_exec_bad_values(args, kwargs, match):
    with pytest.raises(InvalidError, match=match):
        modal.Probe.with_exec(*args, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "Probe must be created with"),
        (
            {"tcp_port": 8080, "exec_argv": ("echo", "ok")},
            "Probe must be created with",
        ),
    ],
)
def test_probe_rejects_invalid_raw_configuration(kwargs, match):
    with pytest.raises(InvalidError, match=match):
        modal.Probe(**kwargs)


@pytest.mark.parametrize("port", ["8080", 0, 65536, -1, 8080.0])
def test_create_connect_token_bad_port_raises(app, servicer, port):
    sb = Sandbox.create(app=app)
    with pytest.raises(InvalidError, match="port must be between 1 and 65535"):
        sb.create_connect_token(port=port)


@pytest.mark.parametrize("port", [1, 8080, 9000, 65535])
def test_create_connect_token_sends_port(app, servicer, port):
    sb = Sandbox.create(app=app)
    with servicer.intercept() as ctx:
        creds = sb.create_connect_token(port=port)
        req = ctx.pop_request("SandboxCreateConnectToken")

    assert req.sandbox_id == sb.object_id
    assert req.port == port
    assert creds.token == f"token-{port}"


def test_create_connect_token_defaults_to_8080(app, servicer):
    sb = Sandbox.create(app=app)
    with servicer.intercept() as ctx:
        creds = sb.create_connect_token()
        req = ctx.pop_request("SandboxCreateConnectToken")

    assert req.port == 8080
    assert creds.token == "token-8080"


@skip_non_subprocess
def test_sandbox(app, servicer, sandbox_subprocess):
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


def test_sandbox_duplicate_volumes(client, servicer):
    """Test that duplicate volumes are detected in sandboxes."""
    app = App()
    vol_a = Volume.from_name("test-vol", create_if_missing=True)
    vol_b = Volume.from_name("test-vol", create_if_missing=True)

    # These are different Python objects but represent the same volume
    assert vol_a is not vol_b

    # Should raise an error when creating sandbox with duplicate volumes
    with pytest.raises(InvalidError, match="same.*[Vv]olume.*multiple"):
        with app.run(client=client):
            Sandbox.create(
                "bash",
                "-c",
                "echo hi",
                timeout=600,
                volumes={"/a": vol_a, "/b": vol_b},
                app=app,
            )


def test_sandbox_mount_layer(app, servicer, tmpdir):
    tmpdir.join("a.py").write(b"foo")

    sb = Sandbox.create("echo", "hi", image=Image.debian_slim().add_local_dir(Path(tmpdir), remote_path="/m"), app=app)
    sb.wait()

    sha = hashlib.sha256(b"foo").hexdigest()
    assert servicer.files_sha2data[sha]["data"] == b"foo"


def test_sandbox_image(app, servicer, tmpdir):
    tmpdir.join("a.py").write(b"foo")

    sb = Sandbox.create("echo", "hi", image=Image.debian_slim().pip_install("foo", "bar", "potato"), app=app)
    sb.wait()

    idx = max(servicer.images.keys())
    last_image = servicer.images[idx]

    assert all(c in last_image.dockerfile_commands[-1] for c in ["foo", "bar", "potato"])


def test_sandbox_secret(app, servicer, tmpdir):
    sb = Sandbox.create("echo", "$FOO", secrets=[Secret.from_dict({"FOO": "BAR"})], app=app)
    sb.wait()

    assert len(servicer.sandbox_defs[0].secret_ids) == 1


@pytest.mark.parametrize("sandbox_version", [SandboxVersion.V1, SandboxVersion.V2], ids=["v1", "v2"])
def test_sandbox_create_hydrates_app_id(app, sandbox_version):
    if sandbox_version == SandboxVersion.V1:
        sb = Sandbox.create("echo", "hello", app=app)
    else:
        sb = Sandbox._experimental_create("echo", "hello", app=app)

    assert synchronizer._translate_in(sb)._app_id == app.app_id


@pytest.mark.parametrize("sandbox_version", [SandboxVersion.V1, SandboxVersion.V2], ids=["v1", "v2"])
def test_sandbox_initialize_from_other_preserves_app_id_and_version(app, sandbox_version):
    if sandbox_version == SandboxVersion.V1:
        source = Sandbox.create("echo", "hello", app=app)
    else:
        source = Sandbox._experimental_create("echo", "hello", app=app)

    source_impl = synchronizer._translate_in(source)
    target_impl = type(source_impl).__new__(type(source_impl))  # type: ignore
    target_impl._init("Sandbox initialized from another Sandbox")
    target_impl._initialize_from_other(source_impl)

    assert target_impl._app_id == source_impl._app_id == app.app_id
    assert target_impl._is_v2 is source_impl._is_v2 is (sandbox_version == SandboxVersion.V2)


def test_sandbox_nfs(client, app, servicer, tmpdir):
    with NetworkFileSystem.ephemeral(client=client) as nfs:
        with pytest.raises(InvalidError):
            Sandbox.create("echo", "foo > /cache/a.txt", network_file_systems={"/": nfs}, app=app)

        Sandbox.create("echo", "foo > /cache/a.txt", network_file_systems={"/cache": nfs}, app=app)

    assert len(servicer.sandbox_defs[0].nfs_mounts) == 1


@skip_non_subprocess
def test_sandbox_from_id(app, client, servicer, sandbox_subprocess):
    sb = Sandbox.create("bash", "-c", "echo foo && exit 42", timeout=600, app=app)
    sb.wait()

    sb2 = Sandbox.from_id(sb.object_id, client=client)
    assert sb2.stdout.read() == "foo\n"
    assert sb2.returncode == 42


@pytest.mark.parametrize(
    ("sandbox_id", "expected_version"),
    [
        ("sb-nGEijt9WbBMlGrsPH9FOaC", SandboxVersion.V1),
        ("sb-01ARZ3NDEKTSV4RRFFQ69G5FAV", SandboxVersion.V2),
    ],
)
def test_get_sandbox_version(sandbox_id, expected_version):
    assert _get_sandbox_version(sandbox_id) == expected_version


@pytest.mark.parametrize(
    "sandbox_id",
    [
        "sb-123",
        "sb-nGEijt9WbBMlGrsPH9FOa_",
        "sb-81ARZ3NDEKTSV4RRFFQ69G5FAV",
        "sb-01arz3ndektsv4rrffq69g5fav",
        "fu-01ARZ3NDEKTSV4RRFFQ69G5FAV",
    ],
)
def test_get_sandbox_version_rejects_invalid_id(sandbox_id):
    with pytest.raises(InvalidError, match="Invalid Sandbox ID"):
        _get_sandbox_version(sandbox_id)


def test_sandbox_from_id_routes_v1(client, servicer):
    sandbox_id = "sb-nGEijt9WbBMlGrsPH9FOaC"
    servicer.sandbox_app_id = "ap-from-id-v1"

    with servicer.intercept() as ctx:
        sb = Sandbox.from_id(sandbox_id, client=client)

    assert sb.returncode == 0
    assert synchronizer._translate_in(sb)._app_id == "ap-from-id-v1"
    (wait_req,) = ctx.get_requests("SandboxWait")
    assert wait_req.sandbox_id == sandbox_id
    assert ctx.get_requests("SandboxWaitV2") == []


def test_sandbox_from_id_routes_v2(client, servicer):
    sandbox_id = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV"
    servicer.sandbox_app_id = "ap-from-id-v2"

    with servicer.intercept() as ctx:
        sb = Sandbox.from_id(sandbox_id, client=client)
        exit_code = sb.terminate(wait=True)

    assert exit_code == 137
    assert synchronizer._translate_in(sb)._app_id == "ap-from-id-v2"
    assert ctx.get_requests("SandboxWait") == []

    wait_reqs = ctx.get_requests("SandboxWaitV2")
    assert [req.sandbox_id for req in wait_reqs] == [sandbox_id, sandbox_id]
    assert wait_reqs[0].timeout == 0

    (terminate_req,) = ctx.get_requests("SandboxTerminateV2")
    assert terminate_req.sandbox_id == sandbox_id


@pytest.mark.parametrize("sandbox_version", [SandboxVersion.V1, SandboxVersion.V2], ids=["v1", "v2"])
def test_sandbox_from_name_hydrates_app_id(app, client, servicer, sandbox_version):
    servicer.sandbox_app_id = "ap-from-name"

    with servicer.intercept() as ctx:
        if sandbox_version == SandboxVersion.V1:
            sb = Sandbox.from_name("my-app", "my-sandbox", client=client)
            method_name = "SandboxGetFromName"
            app_id = "ap-from-name"
        else:
            sb = Sandbox._experimental_create("bash", "-c", "sleep 100", app=app, name="my-sandbox")
            sb._experimental_set_name("my-sandbox")
            sb = Sandbox._experimental_from_name("my-app", "my-sandbox", client=client)
            method_name = "SandboxGetFromNameV2"
            app_id = app.app_id

    assert synchronizer._translate_in(sb)._app_id == app_id
    (request,) = ctx.get_requests(method_name)
    assert request.app_name == "my-app"
    assert request.sandbox_name == "my-sandbox"


def test_sandbox_from_id_rejects_invalid_id(client, servicer):
    with servicer.intercept() as ctx:
        with pytest.raises(InvalidError, match="Invalid Sandbox ID"):
            Sandbox.from_id("sb-123", client=client)

    assert ctx.get_requests("SandboxWait") == []
    assert ctx.get_requests("SandboxWaitV2") == []


# Two distinct valid V2 (ULID) Sandbox IDs, used to exercise name reservations
# across separate sandboxes.
_V2_SANDBOX_ID = "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV"
_V2_SANDBOX_ID_2 = "sb-01ARZ3NDEKTSV4RRFFQ69G5FBV"


def test_experimental_set_name_rejects_v1(app, servicer):
    sb = Sandbox.create("bash", "-c", "sleep 100", app=app)

    with servicer.intercept() as ctx:
        with pytest.raises(InvalidError, match="only supported for V2 sandboxes"):
            sb._experimental_set_name("my-sandbox")

    assert ctx.get_requests("SandboxSetName") == []


def test_experimental_set_name_rejects_invalid_name(app, servicer):
    sb = Sandbox._experimental_create("bash", "-c", "sleep 100", app=app)

    with servicer.intercept() as ctx:
        with pytest.raises(InvalidError, match="Invalid Sandbox name"):
            sb._experimental_set_name("bad name")

    # Validation happens client-side, so no RPC should be sent.
    assert ctx.get_requests("SandboxSetName") == []


def test_experimental_set_name_twice_conflicts(client, servicer):
    sb = Sandbox.from_id(_V2_SANDBOX_ID, client=client)
    sb._experimental_set_name("first")

    with pytest.raises(ConflictError, match="already has a name"):
        sb._experimental_set_name("second")


def test_experimental_set_name_duplicate_conflicts(client, servicer):
    first = Sandbox.from_id(_V2_SANDBOX_ID, client=client)
    first._experimental_set_name("shared")

    second = Sandbox.from_id(_V2_SANDBOX_ID_2, client=client)
    with pytest.raises(AlreadyExistsError, match="already in use"):
        second._experimental_set_name("shared")


def test_experimental_set_name_then_from_name_roundtrip(client, servicer):
    sb = Sandbox.from_id(_V2_SANDBOX_ID, client=client)
    sb._experimental_set_name("round-trip")

    resolved = Sandbox._experimental_from_name("my-app", "round-trip", client=client)
    assert resolved.object_id == sb.object_id


def test_sandbox_terminate(app, servicer):
    sb = Sandbox.create("bash", "-c", "sleep 10000", app=app)
    sb.terminate()

    assert sb.returncode != 0


@skip_non_subprocess
@pytest.mark.asyncio
async def test_sandbox_stdin_async(app, servicer, sandbox_subprocess):
    sb = await Sandbox.create.aio("bash", "-c", "while read line; do echo $line; done && exit 13", app=app)

    sb.stdin.write(b"foo\n")
    sb.stdin.write(b"bar\n")

    sb.stdin.write_eof()

    await sb.stdin.drain.aio()

    await sb.wait.aio()

    assert await sb.stdout.read.aio() == "foo\nbar\n"
    assert sb.returncode == 13


@skip_non_subprocess
def test_sandbox_stdin(app, servicer, sandbox_subprocess):
    sb = Sandbox.create("bash", "-c", "while read line; do echo $line; done && exit 13", app=app)

    sb.stdin.write(b"foo\n")
    sb.stdin.write(b"bar\n")

    sb.stdin.write_eof()

    sb.stdin.drain()

    sb.wait()

    assert sb.stdout.read() == "foo\nbar\n"
    assert sb.returncode == 13


@skip_non_subprocess
def test_sandbox_stdin_write_str(app, servicer, sandbox_subprocess):
    sb = Sandbox.create("bash", "-c", "while read line; do echo $line; done && exit 13", app=app)

    sb.stdin.write("foo\n")
    sb.stdin.write("bar\n")

    sb.stdin.write_eof()

    sb.stdin.drain()

    sb.wait()

    assert sb.stdout.read() == "foo\nbar\n"
    assert sb.returncode == 13


@skip_non_subprocess
def test_sandbox_stdin_write_after_terminate(app, servicer, sandbox_subprocess):
    sb = Sandbox.create("bash", "-c", "echo foo", app=app)
    sb.wait()
    with pytest.raises(ValueError):
        sb.stdin.write(b"foo")
        sb.stdin.drain()


@skip_non_subprocess
def test_sandbox_stdin_write_after_eof(app, servicer, sandbox_subprocess):
    sb = Sandbox.create(app=app)
    sb.stdin.write_eof()
    with pytest.raises(ValueError):
        sb.stdin.write(b"foo")
    sb.terminate()


@skip_non_subprocess
def test_sandbox_stdout(app, servicer, sandbox_subprocess):
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
def test_sandbox_stdout_next(app, servicer, sandbox_subprocess):
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
async def test_sandbox_async_for(app, servicer, sandbox_subprocess):
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
def test_app_sandbox(client, servicer, sandbox_subprocess):
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
    # Accept either wrapper repr or impl repr depending on backend
    s = str(cp)
    assert s.startswith("ContainerProcess(process_id=")

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


@skip_non_subprocess
def test_sandbox_exec_wait_timeout(app, servicer):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("sleep", "999", timeout=1)
    t0 = time.monotonic()
    assert cp.wait() == -1
    assert 0.8 < time.monotonic() - t0 <= 1.2


@skip_non_subprocess
def test_sandbox_exec_poll_timeout(app, servicer):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("sleep", "999", timeout=1)
    assert not cp.poll()
    time.sleep(1.2)
    assert cp.poll() == -1


@skip_non_subprocess
def test_sandbox_exec_output_timeout(app, servicer):
    sb = Sandbox.create("sleep", "infinity", app=app)

    t1 = time.monotonic()
    cp = sb.exec("sh", "-c", "echo hi; sleep 999", timeout=1)
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
def test_sandbox_on_app_lookup(client, servicer, sandbox_subprocess):
    app = App.lookup("my-app", create_if_missing=True, client=client)
    sb = Sandbox.create("echo", "hi", app=app)
    sb.wait()
    assert sb.stdout.read() == "hi\n"
    assert servicer.sandbox_app_id == app.app_id


def test_sandbox_list_env(app, client, servicer):
    sb = Sandbox.create("bash", "-c", "sleep 10000", app=app)
    assert len(list(Sandbox.list(client=client))) == 1
    sb.terminate()
    sb.wait(raise_on_termination=False)
    assert not list(Sandbox.list(client=client))


def test_sandbox_list_app(client, servicer):
    image = Image.debian_slim().pip_install("xyz").add_local_file(__file__, "/xyz")
    secret = Secret.from_dict({"FOO": "bar"})

    app = App()

    with app.run(client=client):
        # Create sandbox
        sb = Sandbox.create("bash", "-c", "sleep 10000", image=image, secrets=[secret], app=app)
        (listed_sandbox,) = list(Sandbox.list(app_id=app.app_id, client=client))
        assert synchronizer._translate_in(listed_sandbox)._app_id == app.app_id
        sb.terminate()
        sb.wait(raise_on_termination=False)
        assert not list(Sandbox.list(app_id=app.app_id, client=client))


def test_sandbox_list_tags(app, client, servicer):
    sb = Sandbox.create("bash", "-c", "sleep 10000", app=app)
    assert sb.get_tags() == {}
    sb.set_tags({"foo": "bar", "baz": "qux"})
    assert sb.get_tags() == {"foo": "bar", "baz": "qux"}

    assert len(list(Sandbox.list(tags={"foo": "bar"}, client=client))) == 1
    assert not list(Sandbox.list(tags={"foo": "notbar"}, client=client))
    sb.terminate()
    sb.wait(raise_on_termination=False)
    assert not list(Sandbox.list(tags={"baz": "qux"}, client=client))


@pytest.mark.parametrize("app_id", [None, ""])
def test_sandbox_experimental_list_requires_app_id(app_id, client, servicer):
    # Outside a Modal container there is no default App to deduce, so a
    # non-empty app_id is required. An empty string must not silently widen
    # the listing to the whole environment.
    with pytest.raises(InvalidError, match="requires an `app_id`"):
        list(Sandbox._experimental_list(app_id=app_id, client=client))


def test_sandbox_experimental_list_app(client, servicer):
    image = Image.debian_slim().pip_install("xyz").add_local_file(__file__, "/xyz")
    secret = Secret.from_dict({"FOO": "bar"})

    app = App()

    with app.run(client=client):
        sb = Sandbox.create("bash", "-c", "sleep 10000", image=image, secrets=[secret], app=app)
        (listed_sandbox,) = list(Sandbox._experimental_list(app_id=app.app_id, client=client))
        assert synchronizer._translate_in(listed_sandbox)._app_id == app.app_id
        sb.terminate()
        sb.wait(raise_on_termination=False)
        assert not list(Sandbox._experimental_list(app_id=app.app_id, client=client))


def test_sandbox_experimental_list_tags(client, servicer):
    app = App()
    with app.run(client=client):
        sb = Sandbox._experimental_create("bash", "-c", "sleep 10000", app=app)
        sb.set_tags({"env": "prod", "team": "infra"})

        def experimental_list(tags):
            return [s.object_id for s in Sandbox._experimental_list(app_id=app.app_id, tags=tags, client=client)]

        assert experimental_list({"env": "prod"}) == [sb.object_id]
        assert experimental_list({"env": "prod", "team": "infra"}) == [sb.object_id]
        assert experimental_list({"env": "staging"}) == []


def test_sandbox_create_with_tags(app, client, servicer):
    tags = {"env": "prod", "team": "core"}
    with servicer.intercept() as ctx:
        sb = Sandbox.create("bash", "-c", "sleep 10000", app=app, tags=tags)

    request: api_pb2.SandboxCreateRequest = ctx.pop_request("SandboxCreate")
    assert {tag.tag_name: tag.tag_value for tag in request.tags} == tags

    assert sb.get_tags() == tags
    sb.terminate()


def test_experimental_sandbox_get_set_tags(client, servicer):
    app = App()
    with app.run(client=client):
        sb = Sandbox._experimental_create("bash", "-c", "sleep 10000", app=app)
        assert sb.get_tags() == {}

        sb.set_tags({"env": "prod", "team": "infra"})
        assert sb.get_tags() == {"env": "prod", "team": "infra"}

        # Setting tags replaces the whole set: "team" is dropped.
        sb.set_tags({"env": "staging"})
        assert sb.get_tags() == {"env": "staging"}

        # An empty dict clears all tags.
        sb.set_tags({})
        assert sb.get_tags() == {}


def test_sandbox_network_access(app, servicer):
    with pytest.raises(InvalidError):
        Sandbox.create("echo", "test", block_network=True, outbound_cidr_allowlist=["10.0.0.0/8"], app=app)

    # Test that blocking works
    sb = Sandbox.create("echo", "test", block_network=True, app=app)
    assert (
        servicer.sandbox_defs[0].network_access.network_access_type == api_pb2.NetworkAccess.NetworkAccessType.BLOCKED
    )
    assert len(servicer.sandbox_defs[0].network_access.allowed_cidrs) == 0
    sb.terminate()

    # Test that allowlisting works via outbound_cidr_allowlist
    sb = Sandbox.create("echo", "test", block_network=False, outbound_cidr_allowlist=["10.0.0.0/8"], app=app)
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

    # Test backward compat: deprecated cidr_allowlist still works with a warning
    with pytest.warns(DeprecationError):
        sb = Sandbox.create("echo", "test", block_network=False, cidr_allowlist=["10.0.0.0/8"], app=app)
    assert (
        servicer.sandbox_defs[3].network_access.network_access_type == api_pb2.NetworkAccess.NetworkAccessType.ALLOWLIST
    )
    assert servicer.sandbox_defs[3].network_access.allowed_cidrs[0] == "10.0.0.0/8"
    sb.terminate()

    # Test that passing both raises an error
    with pytest.raises(InvalidError, match="Cannot specify both"):
        Sandbox.create("echo", "test", cidr_allowlist=["10.0.0.0/8"], outbound_cidr_allowlist=["10.0.0.0/8"], app=app)


def test_sandbox_outbound_domain_allowlist(app, servicer):
    # Cannot combine with block_network
    with pytest.raises(InvalidError, match="`outbound_domain_allowlist` cannot be used when `block_network`"):
        Sandbox.create("echo", "test", block_network=True, outbound_domain_allowlist=["example.com"], app=app)

    # Domain allowlist maps to an ALLOWLIST with allowed_domains set.
    sb = Sandbox.create("echo", "test", outbound_domain_allowlist=["example.com", "*.modal.com"], app=app)
    net = servicer.sandbox_defs[0].network_access
    assert net.network_access_type == api_pb2.NetworkAccess.NetworkAccessType.ALLOWLIST
    assert list(net.allowed_domains) == ["example.com", "*.modal.com"]
    assert len(net.allowed_cidrs) == 0
    sb.terminate()

    # A CIDR allowlist combined with domains becomes the additional IP allowlist.
    sb = Sandbox.create(
        "echo", "test", outbound_domain_allowlist=["example.com"], outbound_cidr_allowlist=["8.8.8.8/32"], app=app
    )
    net = servicer.sandbox_defs[1].network_access
    assert net.network_access_type == api_pb2.NetworkAccess.NetworkAccessType.ALLOWLIST
    assert list(net.allowed_domains) == ["example.com"]
    assert list(net.allowed_cidrs) == ["8.8.8.8/32"]
    sb.terminate()


def test_sandbox__experimental_set_outbound_network_policy(app, servicer):
    sb = Sandbox.create("echo", "test", outbound_domain_allowlist=[], app=app)

    # Allowlist mode
    with servicer.task_command_router.intercept() as tcr_ctx:
        sb._experimental_set_outbound_network_policy(
            outbound_domain_allowlist=["example.com"], outbound_cidr_allowlist=["8.8.8.8/32"]
        )
        (req,) = tcr_ctx.get_requests("TaskSetNetworkAccess")

    net = req.network_access
    assert net.network_access_type == api_pb2.NetworkAccess.NetworkAccessType.ALLOWLIST
    assert list(net.allowed_domains) == ["example.com"]
    assert list(net.allowed_cidrs) == ["8.8.8.8/32"]

    # No arguments sends OPEN type (server may reject this currently).
    with servicer.task_command_router.intercept() as tcr_ctx:
        sb._experimental_set_outbound_network_policy()
        (req,) = tcr_ctx.get_requests("TaskSetNetworkAccess")
    assert req.network_access.network_access_type == api_pb2.NetworkAccess.NetworkAccessType.OPEN

    sb.terminate()


def test_sandbox__experimental_set_outbound_network_policy_v2(app, servicer):
    sb = Sandbox._experimental_create("sleep", "infinity", app=app)

    # Allowlist mode
    with servicer.intercept() as ctx:
        with servicer.task_command_router.intercept() as tcr_ctx:
            sb._experimental_set_outbound_network_policy(
                outbound_domain_allowlist=["example.com"], outbound_cidr_allowlist=["8.8.8.8/32"]
            )

    (access_req,) = ctx.get_requests("SandboxGetCommandRouterAccess")
    assert access_req.sandbox_id == "sb-v2-123"

    (req,) = tcr_ctx.get_requests("TaskSetNetworkAccess")
    assert req.task_id == "ta-v2-123"
    net = req.network_access
    assert net.network_access_type == api_pb2.NetworkAccess.NetworkAccessType.ALLOWLIST
    assert list(net.allowed_domains) == ["example.com"]
    assert list(net.allowed_cidrs) == ["8.8.8.8/32"]

    # No arguments sends OPEN type
    with servicer.task_command_router.intercept() as tcr_ctx:
        sb._experimental_set_outbound_network_policy()
        (req,) = tcr_ctx.get_requests("TaskSetNetworkAccess")
    assert req.network_access.network_access_type == api_pb2.NetworkAccess.NetworkAccessType.OPEN

    sb.terminate()


@skip_non_subprocess
def test_sandbox_inbound_cidr_allowlist(app, servicer):
    # Cannot combine with block_network
    with pytest.raises(InvalidError, match="`inbound_cidr_allowlist` cannot be used when `block_network` is enabled"):
        Sandbox.create("echo", "test", block_network=True, inbound_cidr_allowlist=["10.0.0.0/8"], app=app)

    # Single CIDR is set on the proto
    sb = Sandbox.create("echo", "test", inbound_cidr_allowlist=["10.0.0.0/8"], app=app)
    assert list(servicer.sandbox_defs[0].inbound_cidr_allowlist) == ["10.0.0.0/8"]
    sb.terminate()

    # Multiple CIDRs
    sb = Sandbox.create("echo", "test", inbound_cidr_allowlist=["10.0.0.0/8", "192.168.0.0/16"], app=app)
    assert list(servicer.sandbox_defs[1].inbound_cidr_allowlist) == ["10.0.0.0/8", "192.168.0.0/16"]
    sb.terminate()

    # No restriction when omitted
    sb = Sandbox.create("echo", "test", app=app)
    assert list(servicer.sandbox_defs[2].inbound_cidr_allowlist) == []
    sb.terminate()


def test_sandbox_block_network_with_ports(app, servicer):
    """Test that specifying open ports when block_network is enabled raises an error."""

    # Test with encrypted_ports
    with pytest.raises(InvalidError, match="Cannot specify open ports when `block_network` is enabled"):
        Sandbox.create("echo", "test", block_network=True, encrypted_ports=[8080], app=app)

    # Test with h2_ports
    with pytest.raises(InvalidError, match="Cannot specify open ports when `block_network` is enabled"):
        Sandbox.create("echo", "test", block_network=True, h2_ports=[8080], app=app)

    # Test with unencrypted_ports
    with pytest.raises(InvalidError, match="Cannot specify open ports when `block_network` is enabled"):
        Sandbox.create("echo", "test", block_network=True, unencrypted_ports=[8080], app=app)

    # Test with multiple port types
    with pytest.raises(InvalidError, match="Cannot specify open ports when `block_network` is enabled"):
        Sandbox.create(
            "echo",
            "test",
            block_network=True,
            encrypted_ports=[8080],
            h2_ports=[9090],
            unencrypted_ports=[3000],
            app=app,
        )

    # Test that it works fine when block_network is False
    sb = Sandbox.create(
        "echo", "test", block_network=False, encrypted_ports=[8080], h2_ports=[9090], unencrypted_ports=[3000], app=app
    )
    sb.terminate()


@skip_non_subprocess
def test_sandbox_no_entrypoint(app, servicer):
    sb = Sandbox.create(app=app)

    p = sb.exec("echo", "hi")
    p.wait()
    assert p.returncode == 0
    assert p.stdout.read() == "hi\n"

    sb.terminate()


def test_sandbox_gpu_fallbacks_support(client, servicer):
    with pytest.raises(InvalidError, match="do not support"):
        Sandbox.create(client=client, gpu=["t4", "a100"])  # type: ignore


@skip_non_subprocess
def test_sandbox_exec_with_streamtype_stdout_and_text_true_prints_to_stdout(app, servicer, capsys):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("bash", "-c", "echo hi", stdout=StreamType.STDOUT)
    cp.wait()

    assert capsys.readouterr().out == "hi\n"


@skip_non_subprocess
def test_sandbox_exec_with_streamtype_stderr_and_text_true_prints_to_stdout(app, servicer, capsys):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("bash", "-c", "echo hi >&2", stderr=StreamType.STDOUT)
    cp.wait()

    assert capsys.readouterr().out == "hi\n"


@skip_non_subprocess
def test_sandbox_exec_with_streamtype_stdout_and_text_true_and_bufsize_1_prints_to_stdout(app, servicer, capsys):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("bash", "-c", "echo hi && echo bye", stdout=StreamType.STDOUT, bufsize=1)
    cp.wait()

    assert capsys.readouterr().out == "hi\nbye\n"


@skip_non_subprocess
def test_sandbox_exec_with_streamtype_stdout_and_text_false_prints_to_stdout(app, servicer, capsysbinary):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec(
        "bash",
        "-c",
        "printf '\\x01\\x02\\x03\\n\\x04\\x05\\x06\\n'",
        stdout=StreamType.STDOUT,
        text=False,
    )
    cp.wait()

    assert capsysbinary.readouterr().out == b"\x01\x02\x03\n\x04\x05\x06\n"


@skip_non_subprocess
def test_sandbox_exec_with_streamtype_stdout_read_from_stdout_raises_error(app, servicer, capsys):
    sb = Sandbox.create("sleep", "infinity", app=app)

    cp = sb.exec("bash", "-c", "echo hi", stdout=StreamType.STDOUT)

    with pytest.raises(InvalidError):
        cp.stdout.read()


def test_sandbox_snapshot(app, client, servicer):
    sb = Sandbox.create(app=app, _experimental_enable_snapshot=True)
    sandbox_snapshot = sb._experimental_snapshot()
    snapshot_id = sandbox_snapshot.object_id
    assert snapshot_id == "sn-123"
    sb.terminate()

    with servicer.intercept() as ctx:
        sandbox_snapshot = SandboxSnapshot.from_id(snapshot_id, client=client)
        assert sandbox_snapshot.object_id == snapshot_id  # snapshot id is immediately available
        assert len(ctx.calls) == 0

        ctx.add_response("SandboxSnapshotGet", api_pb2.SandboxSnapshotGetResponse(snapshot_id="sn-123"))
        sandbox_snapshot.hydrate()
        assert sandbox_snapshot.client == client

    sb = Sandbox._experimental_from_snapshot(sandbox_snapshot, client=client)
    sb.terminate()


def test_sandbox_snapshot_v2(app, servicer):
    """V2 sandboxes take memory snapshots through the task command router."""
    sb = Sandbox._experimental_create("sleep", "infinity", app=app)
    with servicer.task_command_router.intercept() as tcr_ctx:
        snapshot = sb._experimental_snapshot()

    assert snapshot.object_id == "sn-01BX5ZZKBKACTAV9WEVGEMMVRY"
    (req,) = tcr_ctx.get_requests("TaskSnapshotMemory")
    assert req.task_id == "ta-v2-123"
    # Validate that the client generates a fresh UUID idempotency key.
    uuid.UUID(req.idempotency_key)

    sb.terminate()


def test_sandbox_from_snapshot_v2(app, client, servicer):
    sb = Sandbox._experimental_create("sleep", "infinity", app=app, encrypted_ports=[8080])
    snapshot = sb._experimental_snapshot()
    assert snapshot.object_id == "sn-01BX5ZZKBKACTAV9WEVGEMMVRY"

    with servicer.intercept() as ctx:
        restored = Sandbox._experimental_from_snapshot(snapshot, client=client)

    assert ctx.get_requests("SandboxRestore") == []
    (v2_req,) = ctx.get_requests("SandboxRestoreV2")
    assert v2_req.snapshot_id == "sn-01BX5ZZKBKACTAV9WEVGEMMVRY"
    assert v2_req.sandbox_name_override_type == api_pb2.SandboxRestoreRequest.SANDBOX_NAME_OVERRIDE_TYPE_UNSPECIFIED

    assert restored.object_id == "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV"
    (wait_req,) = ctx.get_requests("SandboxWaitV2")
    assert wait_req.sandbox_id == "sb-01ARZ3NDEKTSV4RRFFQ69G5FAV"

    with servicer.intercept() as ctx:
        tunnels = restored.tunnels()
    assert 8080 in tunnels
    assert len(ctx.get_requests("SandboxGetTunnelsV2")) == 1


def test_sandbox_from_snapshot_v2_auto_detects_version(app, client, servicer):
    """A snapshot taken from a V2 sandbox remembers its version and restores as V2 without a flag."""
    sb = Sandbox._experimental_create("sleep", "infinity", app=app)
    snapshot = sb._experimental_snapshot()

    with servicer.intercept() as ctx:
        Sandbox._experimental_from_snapshot(snapshot, client=client)

    # The remembered version routes to V2 without a SandboxSnapshotGet round-trip.
    assert ctx.get_requests("SandboxRestore") == []
    assert ctx.get_requests("SandboxSnapshotGet") == []
    (v2_req,) = ctx.get_requests("SandboxRestoreV2")
    assert v2_req.snapshot_id == "sn-01BX5ZZKBKACTAV9WEVGEMMVRY"


def test_sandbox_from_snapshot_v2_auto_detects_from_id(app, client, servicer):
    """A snapshot loaded via from_id learns its version by hydrating, then restores as V2."""
    snapshot = SandboxSnapshot.from_id("sn-01BX5ZZKBKACTAV9WEVGEMMVRY", client=client)

    with servicer.intercept() as ctx:
        ctx.add_response(
            "SandboxSnapshotGet",
            api_pb2.SandboxSnapshotGetResponse(
                snapshot_id="sn-01BX5ZZKBKACTAV9WEVGEMMVRY",
                handle_metadata=api_pb2.SandboxSnapshotHandleMetadata(is_v2=True),
            ),
        )
        Sandbox._experimental_from_snapshot(snapshot, client=client)

    assert ctx.get_requests("SandboxRestore") == []
    assert len(ctx.get_requests("SandboxSnapshotGet")) == 1
    (v2_req,) = ctx.get_requests("SandboxRestoreV2")
    assert v2_req.snapshot_id == "sn-01BX5ZZKBKACTAV9WEVGEMMVRY"


def test_sandbox_snapshot_fs(app, servicer):
    sb = Sandbox.create(app=app)
    with servicer.task_command_router.intercept() as tcr_ctx:
        image = sb.snapshot_filesystem()
        assert image.object_id == "im-snapshot-fs-123"

        # Default ttl is 30 days, sent as a positive ttl_seconds.
        captured_requests = tcr_ctx.get_requests("TaskSnapshotFilesystem")
        assert len(captured_requests) == 1
        assert captured_requests[0].HasField("ttl_seconds")
        assert captured_requests[0].ttl_seconds == 30 * 24 * 3600
        snapshot_id = captured_requests[0].snapshot_id
        assert snapshot_id, "snapshot_id must be non-empty"
        uuid.UUID(snapshot_id)  # Raises ValueError if not a valid UUID

        # Each call generates a unique snapshot_id.
        sb.snapshot_filesystem()
        captured_requests = tcr_ctx.get_requests("TaskSnapshotFilesystem")
        assert captured_requests[-1].snapshot_id != snapshot_id

        # Explicit ttl is forwarded as-is.
        sb.snapshot_filesystem(ttl=3600)
        captured_requests = tcr_ctx.get_requests("TaskSnapshotFilesystem")
        assert captured_requests[-1].ttl_seconds == 3600

        # ttl=None opts out of expiry; the wire carries -1 as the sentinel.
        sb.snapshot_filesystem(ttl=None)
        captured_requests = tcr_ctx.get_requests("TaskSnapshotFilesystem")
        assert captured_requests[-1].ttl_seconds == -1

        # 0 / negative ttl is rejected client-side.
        with pytest.raises(InvalidError, match="must be positive"):
            sb.snapshot_filesystem(ttl=0)
        with pytest.raises(InvalidError, match="must be positive"):
            sb.snapshot_filesystem(ttl=-5)

    sb.terminate()


@pytest.mark.parametrize("legacy_env_var", [False, True])
def test_sandbox_snapshot_fs_v2(app, servicer, monkeypatch, legacy_env_var):
    if legacy_env_var:
        monkeypatch.setenv("MODAL_USE_LEGACY_FILESYSTEM_SNAPSHOT", "1")

    sb = Sandbox._experimental_create("sleep", "infinity", app=app)
    with servicer.intercept() as ctx:
        with servicer.task_command_router.intercept() as tcr_ctx:
            image = sb.snapshot_filesystem()

    assert image.object_id == "im-snapshot-fs-123"
    (access_req,) = ctx.get_requests("SandboxGetCommandRouterAccess")
    assert access_req.sandbox_id == "sb-v2-123"

    (req,) = tcr_ctx.get_requests("TaskSnapshotFilesystem")
    assert req.task_id == "ta-v2-123"
    assert req.HasField("ttl_seconds")
    assert req.ttl_seconds == 30 * 24 * 3600
    # Validate that the client generates a fresh UUID idempotency key.
    uuid.UUID(req.snapshot_id)

    sb.terminate()


def test_sandbox_reload_volumes(app, servicer):
    """V1 sandboxes route reload_volumes() through the command router."""
    sb = Sandbox.create("sleep", "infinity", app=app)
    with servicer.task_command_router.intercept() as tcr_ctx:
        sb.reload_volumes()
        sb.reload_volumes(timeout=17)

    reqs = tcr_ctx.get_requests("TaskReloadVolumes")
    assert len(reqs) == 2
    assert all(req.task_id for req in reqs)

    sb.terminate()


def test_sandbox_reload_volumes_v2(app, servicer):
    """V2 sandboxes route reload_volumes() through the command router."""
    sb = Sandbox._experimental_create("sleep", "infinity", app=app)
    with servicer.task_command_router.intercept() as tcr_ctx:
        sb.reload_volumes()

    (req,) = tcr_ctx.get_requests("TaskReloadVolumes")
    assert req.task_id == "ta-v2-123"

    sb.terminate()


@pytest.mark.parametrize("create", [Sandbox.create, Sandbox._experimental_create], ids=["v1", "v2"])
def test_sandbox_reload_volumes_timeout_validation(app, servicer, create):
    """A non-positive `timeout` is rejected for all sandboxes."""
    sb = create("sleep", "infinity", app=app)
    with pytest.raises(InvalidError, match="must be positive"):
        sb.reload_volumes(timeout=0)
    with pytest.raises(InvalidError, match="must be positive"):
        sb.reload_volumes(timeout=-5)

    sb.terminate()


def test_sandbox_snapshot_fs_legacy_env_var(app, servicer, monkeypatch):
    monkeypatch.setenv("MODAL_USE_LEGACY_FILESYSTEM_SNAPSHOT", "1")

    sb = Sandbox.create(app=app)

    with servicer.task_command_router.intercept() as tcr_ctx:
        with servicer.intercept() as ctx:
            image = sb.snapshot_filesystem(timeout=17)

    assert image.object_id == "im-123"
    (snapshot_req,) = ctx.get_requests("SandboxSnapshotFs")
    assert snapshot_req.sandbox_id == sb.object_id
    assert snapshot_req.timeout == 17
    assert tcr_ctx.get_requests("TaskSnapshotFilesystem") == []

    sb.terminate()


def test_sandbox_experimental_get_exit_snapshot_success(app, servicer):
    sb = Sandbox.create(app=app)

    with servicer.intercept() as ctx:
        ctx.add_response(
            "SandboxGetExitSnapshot",
            api_pb2.SandboxGetExitSnapshotResponse(
                success=api_pb2.SandboxGetExitSnapshotResponse.Success(image_id="im-exit-snapshot-123")
            ),
        )
        image = sb._experimental_get_exit_snapshot(timeout=0)

    assert image.object_id == "im-exit-snapshot-123"
    (req,) = ctx.get_requests("SandboxGetExitSnapshot")
    assert req.sandbox_id == sb.object_id

    sb.terminate()


def test_sandbox_experimental_get_exit_snapshot_polls_until_success_without_timeout(app, servicer, monkeypatch):
    monkeypatch.setattr(modal.sandbox, "_EXIT_SNAPSHOT_POLL_INTERVAL_SECONDS", 0)
    sb = Sandbox.create(app=app)

    with servicer.intercept() as ctx:
        ctx.add_response(
            "SandboxGetExitSnapshot",
            api_pb2.SandboxGetExitSnapshotResponse(pending=api_pb2.SandboxGetExitSnapshotResponse.Pending()),
        )
        ctx.add_response(
            "SandboxGetExitSnapshot",
            api_pb2.SandboxGetExitSnapshotResponse(
                success=api_pb2.SandboxGetExitSnapshotResponse.Success(image_id="im-exit-snapshot-123")
            ),
        )
        image = sb._experimental_get_exit_snapshot()

    assert image.object_id == "im-exit-snapshot-123"
    assert len(ctx.get_requests("SandboxGetExitSnapshot")) == 2

    sb.terminate()


def test_sandbox_experimental_get_exit_snapshot_pending_times_out(app, servicer):
    sb = Sandbox.create(app=app)

    with servicer.intercept() as ctx:
        ctx.add_response(
            "SandboxGetExitSnapshot",
            api_pb2.SandboxGetExitSnapshotResponse(pending=api_pb2.SandboxGetExitSnapshotResponse.Pending()),
        )
        with pytest.raises(TimeoutError, match="timed out"):
            sb._experimental_get_exit_snapshot(timeout=0)

    assert len(ctx.get_requests("SandboxGetExitSnapshot")) == 1

    sb.terminate()


def test_sandbox_experimental_get_exit_snapshot_rejects_negative_timeout(app, servicer):
    sb = Sandbox.create(app=app)

    with servicer.intercept() as ctx:
        with pytest.raises(InvalidError, match="timeout"):
            sb._experimental_get_exit_snapshot(timeout=-1)

    assert ctx.get_requests("SandboxGetExitSnapshot") == []

    sb.terminate()


@pytest.mark.parametrize(
    "error_code",
    [
        api_pb2.SandboxGetExitSnapshotResponse.ERROR_CODE_TIMEOUT,
    ],
)
def test_sandbox_experimental_get_exit_snapshot_not_found_errors(app, servicer, error_code):
    sb = Sandbox.create(app=app)

    with servicer.intercept() as ctx:
        ctx.add_response(
            "SandboxGetExitSnapshot",
            api_pb2.SandboxGetExitSnapshotResponse(
                error=api_pb2.SandboxGetExitSnapshotResponse.Error(
                    error_code=error_code,
                    message="no exit snapshot",
                )
            ),
        )
        with pytest.raises(modal.exception.SnapshotCreationError, match="no exit snapshot"):
            sb._experimental_get_exit_snapshot(timeout=0)

    sb.terminate()


def test_sandbox_experimental_get_exit_snapshot_not_enabled_raises_invalid(app, servicer):
    sb = Sandbox.create(app=app)

    async def responder(servicer, stream):
        await stream.recv_message()
        raise GRPCError(Status.INVALID_ARGUMENT, "Exit snapshot is not enabled for this sandbox")

    with servicer.intercept() as ctx:
        ctx.set_responder("SandboxGetExitSnapshot", responder)
        with pytest.raises(InvalidError, match="not enabled"):
            sb._experimental_get_exit_snapshot(timeout=0)

    sb.terminate()


def test_sandbox_experimental_get_exit_snapshot_internal_error(app, servicer):
    sb = Sandbox.create(app=app)

    with servicer.intercept() as ctx:
        ctx.add_response(
            "SandboxGetExitSnapshot",
            api_pb2.SandboxGetExitSnapshotResponse(
                error=api_pb2.SandboxGetExitSnapshotResponse.Error(
                    error_code=api_pb2.SandboxGetExitSnapshotResponse.ERROR_CODE_INTERNAL,
                    message="malformed snapshot result",
                )
            ),
        )
        with pytest.raises(modal.exception.InternalError, match="malformed snapshot result"):
            sb._experimental_get_exit_snapshot(timeout=0)

    sb.terminate()


def test_sandbox_cpu_request(app, servicer):
    _ = Sandbox.create(cpu=2.0, app=app)

    assert servicer.sandbox_defs[0].resources.milli_cpu == 2000
    assert servicer.sandbox_defs[0].resources.milli_cpu_max == 0


def test_sandbox_cpu_limit(app, servicer):
    _ = Sandbox.create(cpu=(2, 4), app=app)

    assert servicer.sandbox_defs[0].resources.milli_cpu == 2000
    assert servicer.sandbox_defs[0].resources.milli_cpu_max == 4000


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
                        metadata=api_pb2.SandboxHandleMetadata(app_id="ap-list-running"),
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
    assert synchronizer._translate_in(list_result)._app_id == "ap-list-running"


def test_sandbox_list_sets_correct_returncode_for_stopped(client, servicer):
    with servicer.intercept() as ctx:
        # test generic status
        ctx.add_response(
            "SandboxList",
            api_pb2.SandboxListResponse(
                sandboxes=[
                    api_pb2.SandboxInfo(
                        id="sb-123",
                        metadata=api_pb2.SandboxHandleMetadata(app_id="ap-list-stopped"),
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
    assert synchronizer._translate_in(list_result)._app_id == "ap-list-stopped"


@pytest.mark.parametrize("read_only", [True, False])
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


def test_sandbox_create_pty(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox.create("echo", "hi", pty=True, app=app)
        req = ctx.pop_request("SandboxCreate")

        assert req.definition.pty_info is not None
        assert req.definition.pty_info.enabled is True
        assert req.definition.pty_info.pty_type == api_pb2.PTYInfo.PTY_TYPE_SHELL
        assert req.definition.pty_info.no_terminate_on_idle_stdin is True


def test_experimental_sandbox_create_resources_default(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app)
        req = ctx.pop_request("SandboxCreateV2")

        assert req.definition.resources.memory_mb == 0
        assert req.definition.resources.milli_cpu == 0


def test_experimental_sandbox_create_memory_roundtrip(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, cpu=2.0, memory=512)
        req = ctx.pop_request("SandboxCreateV2")

        assert req.definition.resources.memory_mb == 512
        assert req.definition.resources.memory_mb_max == 0
        assert req.definition.resources.milli_cpu == 2000


def test_experimental_sandbox_create_cpu_and_memory_limits(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, cpu=(0.5, 2), memory=(1024, 2048))
        req = ctx.pop_request("SandboxCreateV2")

        assert req.definition.resources.milli_cpu == 500
        assert req.definition.resources.milli_cpu_max == 2000
        assert req.definition.resources.memory_mb == 1024
        assert req.definition.resources.memory_mb_max == 2048


def test_experimental_sandbox_create_custom_domain(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, custom_domain="sandboxes.example.com")
        req = ctx.pop_request("SandboxCreateV2")

        assert req.definition.custom_domain == "sandboxes.example.com"


def test_experimental_sandbox_tunnels_encrypted_only_cached_from_create(app, servicer):
    with servicer.intercept() as ctx:
        sb = Sandbox._experimental_create("echo", "hi", app=app, encrypted_ports=[8080])
        tunnels = sb.tunnels()

        assert len(ctx.get_requests("SandboxGetTunnelsV2")) == 0
        assert tunnels[8080].host == "sb-v2-123-8080.modal.host"
        assert tunnels[8080].port == 443


def test_experimental_sandbox_tunnels_fetches_unencrypted(app, servicer):
    with servicer.intercept() as ctx:
        sb = Sandbox._experimental_create("echo", "hi", app=app, encrypted_ports=[8080], unencrypted_ports=[9000])
        tunnels = sb.tunnels()

        # Unencrypted tunnels are missing from the create response. tunnels() fetches all of them.
        assert len(ctx.get_requests("SandboxGetTunnelsV2")) == 1
        assert len(tunnels) == 2
        assert tunnels[8080].host == "sb-v2-123-8080.modal.host"
        assert tunnels[9000].unencrypted_host == "r1.modal.host"
        assert tunnels[9000].unencrypted_port == 39000


def test_experimental_sandbox_create_experimental_options(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, experimental_options={"enable_docker": True})
        req = ctx.pop_request("SandboxCreateV2")
        assert req.definition.experimental_options_v2 == {"enable_docker": "True"}


def test_experimental_sandbox_create_no_experimental_options_by_default(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app)
        req = ctx.pop_request("SandboxCreateV2")
        assert dict(req.definition.experimental_options_v2) == {}


@pytest.mark.parametrize("read_only", [True, False])
def test_experimental_sandbox_create_volume(app, servicer, read_only):
    volume = Volume.from_name("my-volume", create_if_missing=True)

    if read_only:
        volume = volume.with_mount_options(read_only=True)

    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, volumes={"/mnt": volume})
        req = ctx.pop_request("SandboxCreateV2")
        assert len(req.definition.volume_mounts) == 1
        assert req.definition.volume_mounts[0].mount_path == "/mnt"
        assert req.definition.volume_mounts[0].read_only == read_only


def test_experimental_sandbox_create_cloud_bucket_mount(app, servicer):
    cbm = modal.CloudBucketMount(bucket_name="my-bucket")
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, volumes={"/mnt": cbm})
        req = ctx.pop_request("SandboxCreateV2")
        assert len(req.definition.cloud_bucket_mounts) == 1
        assert req.definition.cloud_bucket_mounts[0].bucket_name == "my-bucket"
        assert req.definition.cloud_bucket_mounts[0].mount_path == "/mnt"
        assert req.definition.cloud_bucket_mounts[0].bucket_type == api_pb2.CloudBucketMount.BucketType.S3


def test_experimental_sandbox_create_cloud_bucket_mount_no_credentials(app, servicer):
    cbm = modal.CloudBucketMount(bucket_name="my-bucket")
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, volumes={"/mnt": cbm})
        req = ctx.pop_request("SandboxCreateV2")
        assert dict(req.cloud_bucket_mount_credentials) == {}
        assert req.definition.cloud_bucket_mounts[0].credentials_secret_id == ""


def test_experimental_sandbox_create_cloud_bucket_mount_credentials(app, servicer):
    secret = modal.Secret.from_dict({"AWS_ACCESS_KEY_ID": "abc", "AWS_SECRET_ACCESS_KEY": "xyz"})
    cbm = modal.CloudBucketMount(bucket_name="my-bucket", secret=secret)
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, volumes={"/mnt": cbm})
        req = ctx.pop_request("SandboxCreateV2")

    # Credentials are passed out-of-band, keyed by mount path, not via a secret id.
    assert len(req.definition.cloud_bucket_mounts) == 1
    mount = req.definition.cloud_bucket_mounts[0]
    assert mount.credentials_secret_id == ""

    assert len(req.cloud_bucket_mount_credentials) == 1
    assert dict(req.cloud_bucket_mount_credentials["/mnt"].contents) == {
        "AWS_ACCESS_KEY_ID": "abc",
        "AWS_SECRET_ACCESS_KEY": "xyz",
    }


def test_experimental_sandbox_create_cloud_bucket_mount_credentials_multiple(app, servicer):
    cbm1 = modal.CloudBucketMount(
        bucket_name="bucket-1",
        secret=modal.Secret.from_dict({"AWS_ACCESS_KEY_ID": "one"}),
    )
    cbm2 = modal.CloudBucketMount(bucket_name="bucket-2")
    cbm3 = modal.CloudBucketMount(
        bucket_name="bucket-3",
        secret=modal.Secret.from_dict({"AWS_ACCESS_KEY_ID": "three"}),
    )
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, volumes={"/mnt1": cbm1, "/mnt2": cbm2, "/mnt3": cbm3})
        req = ctx.pop_request("SandboxCreateV2")

    # Out-of-band credentials are keyed by mount path.
    creds = req.cloud_bucket_mount_credentials
    assert dict(creds["/mnt1"].contents) == {"AWS_ACCESS_KEY_ID": "one"}
    assert "/mnt2" not in creds
    assert dict(creds["/mnt3"].contents) == {"AWS_ACCESS_KEY_ID": "three"}

    assert len(creds) == 2


def test_experimental_sandbox_create_cloud_bucket_mount_credentials_from_name(app, servicer, client):
    Secret.objects.create("my-bucket-creds", {"AWS_ACCESS_KEY_ID": "abc"}, client=client)
    cbm = modal.CloudBucketMount(bucket_name="my-bucket", secret=modal.Secret.from_name("my-bucket-creds"))
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, volumes={"/mnt": cbm})
        req = ctx.pop_request("SandboxCreateV2")

    # `Secret.from_name` credentials are resolved server-side and referenced by id, rather than
    # being inlined out-of-band into `cloud_bucket_mount_credentials`.
    assert len(req.definition.cloud_bucket_mounts) == 1
    assert req.definition.cloud_bucket_mounts[0].credentials_secret_id != ""
    assert dict(req.cloud_bucket_mount_credentials) == {}


def test_experimental_sandbox_create_cloud_bucket_mount_credentials_from_dict_and_from_name(app, servicer, client):
    Secret.objects.create("my-bucket-creds", {"AWS_ACCESS_KEY_ID": "named"}, client=client)
    cbm_dict = modal.CloudBucketMount(
        bucket_name="bucket-dict",
        secret=modal.Secret.from_dict({"AWS_ACCESS_KEY_ID": "inline"}),
    )
    cbm_name = modal.CloudBucketMount(
        bucket_name="bucket-name",
        secret=modal.Secret.from_name("my-bucket-creds"),
    )
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, volumes={"/mnt-dict": cbm_dict, "/mnt-name": cbm_name})
        req = ctx.pop_request("SandboxCreateV2")

    mounts = {m.mount_path: m for m in req.definition.cloud_bucket_mounts}
    assert len(mounts) == 2

    # `from_dict` credentials are inlined out-of-band, keyed by mount path, with no secret id.
    assert mounts["/mnt-dict"].credentials_secret_id == ""
    assert dict(req.cloud_bucket_mount_credentials["/mnt-dict"].contents) == {"AWS_ACCESS_KEY_ID": "inline"}

    # `from_name` credentials are resolved server-side and referenced by id.
    assert mounts["/mnt-name"].credentials_secret_id != ""
    assert "/mnt-name" not in req.cloud_bucket_mount_credentials

    assert len(req.cloud_bucket_mount_credentials) == 1


def test_experimental_sandbox_create_proxy(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, proxy=Proxy.from_name("my-proxy"))
        req = ctx.pop_request("SandboxCreateV2")
        assert req.definition.proxy_id == "pr-123"


def test_experimental_sandbox_create_with_tags(app, servicer):
    tags = {"env": "prod", "team": "infra"}
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, tags=tags)
        req = ctx.pop_request("SandboxCreateV2")
        assert {tag.tag_name: tag.tag_value for tag in req.tags} == tags


def test_experimental_sandbox_create_no_tags_by_default(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app)
        req = ctx.pop_request("SandboxCreateV2")
        assert list(req.tags) == []


def test_experimental_sandbox_create_no_proxy_by_default(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app)
        req = ctx.pop_request("SandboxCreateV2")
        assert req.definition.proxy_id == ""


def test_experimental_sandbox_create_no_oidc_identity_token_by_default(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app)
        req = ctx.pop_request("SandboxCreateV2")
        assert req.definition.include_oidc_identity_token is False


def test_experimental_sandbox_create_include_oidc_identity_token(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, include_oidc_identity_token=True)
        req = ctx.pop_request("SandboxCreateV2")
        assert req.definition.include_oidc_identity_token is True


def test_experimental_sandbox_create_no_i6pn_by_default(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app)
        req = ctx.pop_request("SandboxCreateV2")
        assert req.definition.i6pn_enabled is False


def test_experimental_sandbox_create_i6pn(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, i6pn=True)
        req = ctx.pop_request("SandboxCreateV2")
        assert req.definition.i6pn_enabled is True


def test_experimental_sandbox_create_i6pn_with_block_network_raises(app, servicer):
    with pytest.raises(InvalidError, match="`block_network` disables all networking, including i6pn"):
        Sandbox._experimental_create("echo", "hi", app=app, i6pn=True, block_network=True)


def test_experimental_sandbox_create_cloud_bucket_mount_oidc_auth_role_arn(app, servicer):
    cbm = modal.CloudBucketMount(bucket_name="my-bucket", oidc_auth_role_arn="arn:aws:iam::123456789012:role/r")
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, volumes={"/mnt": cbm})
        req = ctx.pop_request("SandboxCreateV2")
        assert len(req.definition.cloud_bucket_mounts) == 1
        assert req.definition.cloud_bucket_mounts[0].oidc_auth_role_arn == "arn:aws:iam::123456789012:role/r"


def test_experimental_sandbox_create_env_uses_ephemeral_secrets(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, env={"FOO": "bar", "BAZ": "qux"})
        req = ctx.pop_request("SandboxCreateV2")

    assert dict(req.ephemeral_secrets.contents) == {"FOO": "bar", "BAZ": "qux"}
    assert ctx.get_requests("SecretGetOrCreate") == []
    assert list(req.definition.secret_ids) == []


def test_experimental_sandbox_create_env_drops_none_values(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, env={"FOO": "bar", "SKIP": None})
        req = ctx.pop_request("SandboxCreateV2")

    assert dict(req.ephemeral_secrets.contents) == {"FOO": "bar"}


def test_experimental_sandbox_create_secret_from_dict_uses_ephemeral_secrets(app, servicer):
    secret = Secret.from_dict({"DB_PASSWORD": "hunter2"})
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, secrets=[secret])
        req = ctx.pop_request("SandboxCreateV2")

    # `Secret.from_dict` is resolvable locally, so its contents are inlined into the request
    # as ephemeral secrets rather than being created server-side and referenced by id.
    assert dict(req.ephemeral_secrets.contents) == {"DB_PASSWORD": "hunter2"}
    assert ctx.get_requests("SecretGetOrCreate") == []
    assert list(req.definition.secret_ids) == []


def test_experimental_sandbox_create_multiple_secrets_from_dict_merge(app, servicer):
    secrets = [Secret.from_dict({"FOO": "bar"}), Secret.from_dict({"BAZ": "qux"})]
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, secrets=secrets)
        req = ctx.pop_request("SandboxCreateV2")

    assert dict(req.ephemeral_secrets.contents) == {"FOO": "bar", "BAZ": "qux"}
    assert ctx.get_requests("SecretGetOrCreate") == []
    assert list(req.definition.secret_ids) == []


def test_experimental_sandbox_create_env_and_secrets_coexist(app, servicer):
    secret = Secret.from_dict({"DB_PASSWORD": "hunter2"})
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, env={"FOO": "bar"}, secrets=[secret])
        req = ctx.pop_request("SandboxCreateV2")

    # Both the `env` argument and `Secret.from_dict` contents are merged into ephemeral secrets.
    assert dict(req.ephemeral_secrets.contents) == {"FOO": "bar", "DB_PASSWORD": "hunter2"}
    assert ctx.get_requests("SecretGetOrCreate") == []
    assert list(req.definition.secret_ids) == []


def test_experimental_sandbox_create_secret_from_dict_and_from_name(app, servicer, client):
    Secret.objects.create("my-secret", {"DB_PASSWORD": "hunter2"}, client=client)
    secrets = [Secret.from_dict({"FOO": "bar"}), Secret.from_name("my-secret")]
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, secrets=secrets)
        req = ctx.pop_request("SandboxCreateV2")

    # `from_dict` is inlined as an ephemeral secret; `from_name` is resolved server-side
    # and referenced by id.
    assert dict(req.ephemeral_secrets.contents) == {"FOO": "bar"}
    assert len(req.definition.secret_ids) == 1


def test_experimental_sandbox_create_no_env_omits_ephemeral_secrets(app, servicer):
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app)
        req = ctx.pop_request("SandboxCreateV2")

    assert not req.HasField("ephemeral_secrets")


@pytest.mark.parametrize("bad_key", ["MY KEY", "1FOO", "FOO-BAR", "FOO$"])
def test_experimental_sandbox_create_env_rejects_invalid_key(app, servicer, bad_key):
    with servicer.intercept() as ctx:
        with pytest.raises(InvalidError, match="invalid for environment variables"):
            Sandbox._experimental_create("echo", "hi", app=app, env={bad_key: "value"})
        assert ctx.get_requests("SandboxCreateV2") == []


def test_experimental_sandbox_create_env_rejects_empty_key(app, servicer):
    with pytest.raises(InvalidError, match="cannot be empty"):
        Sandbox._experimental_create("echo", "hi", app=app, env={"": "value"})


def test_experimental_sandbox_create_outbound_domain_allowlist(app, servicer):
    # Cannot combine with block_network.
    with pytest.raises(InvalidError, match="`outbound_domain_allowlist` cannot be used when `block_network`"):
        Sandbox._experimental_create(
            "echo", "hi", app=app, block_network=True, outbound_domain_allowlist=["example.com"]
        )

    # Domain allowlist maps to an ALLOWLIST with allowed_domains set.
    with servicer.intercept() as ctx:
        Sandbox._experimental_create("echo", "hi", app=app, outbound_domain_allowlist=["example.com", "*.modal.com"])
        req = ctx.pop_request("SandboxCreateV2")
    net = req.definition.network_access
    assert net.network_access_type == api_pb2.NetworkAccess.NetworkAccessType.ALLOWLIST
    assert list(net.allowed_domains) == ["example.com", "*.modal.com"]
    assert len(net.allowed_cidrs) == 0

    # Domains and CIDRs combine into a single ALLOWLIST.
    with servicer.intercept() as ctx:
        Sandbox._experimental_create(
            "echo", "hi", app=app, outbound_domain_allowlist=["example.com"], outbound_cidr_allowlist=["8.8.8.8/32"]
        )
        req = ctx.pop_request("SandboxCreateV2")
    net = req.definition.network_access
    assert net.network_access_type == api_pb2.NetworkAccess.NetworkAccessType.ALLOWLIST
    assert list(net.allowed_domains) == ["example.com"]
    assert list(net.allowed_cidrs) == ["8.8.8.8/32"]


@skip_non_subprocess
def test_sandbox_exec_pty(app, servicer):
    sb = Sandbox.create("sleep", "infinity", app=app)
    with servicer.task_command_router.intercept() as tcr_ctx:
        cp = sb.exec("echo", "hello", pty=True)
        cp.wait()
    (captured_request,) = tcr_ctx.get_requests("TaskExecStart")
    pty_info = captured_request.pty_info

    assert pty_info is not None
    assert pty_info.enabled is True
    assert pty_info.pty_type == api_pb2.PTYInfo.PTY_TYPE_SHELL
    assert pty_info.no_terminate_on_idle_stdin is True


@skip_non_subprocess
def test_sandbox_exec_env_routing(app, servicer):
    secret = Secret.from_dict({"KEEP": "secret", "SECRET_ONLY": "present"})
    sb = Sandbox.create("sleep", "infinity", app=app)
    with servicer.task_command_router.intercept() as tcr_ctx:
        cp = sb.exec(
            "bash",
            "-c",
            'printf "%s|%s|%s" "${KEEP:-missing}" "${PLAIN:-missing}" "${DROP:-missing}"',
            env={"KEEP": "value", "PLAIN": "plain", "DROP": None},
            secrets=[secret],
        )

        assert cp.stdout.read() == "value|plain|missing"
    (exec_start_request,) = tcr_ctx.get_requests("TaskExecStart")
    assert dict(exec_start_request.env) == {"KEEP": "value", "PLAIN": "plain", "SECRET_ONLY": "present"}
    assert list(exec_start_request.secret_ids) == []


@skip_non_subprocess
def test_sandbox_exec_env_routing_named_secret(app, servicer, client):
    # Named secrets are resolved server-side and passed by ID; their env vars are not inlined.
    Secret.objects.create("my-secret", {"SECRET_ONLY": "present"}, client=client)
    secret = Secret.from_name("my-secret")
    sb = Sandbox.create("sleep", "infinity", app=app)
    with servicer.task_command_router.intercept() as tcr_ctx:
        cp = sb.exec(
            "echo",
            "hello",
            env={"PLAIN": "plain"},
            secrets=[secret],
        )
    (exec_start_request,) = tcr_ctx.get_requests("TaskExecStart")
    assert dict(exec_start_request.env) == {"PLAIN": "plain"}
    # The named secret will be hydreated
    assert list(exec_start_request.secret_ids) == [secret.object_id]


def test_mount_image(servicer, client, app):
    """Test mounting an image at a path in the sandbox."""
    sb = Sandbox.create(app=app)

    with servicer.task_command_router.intercept() as tcr_ctx:
        # Test mounting a prebuilt image
        prebuilt_image = Image.debian_slim().run_commands("echo prebuilt").build(app)
        sb.mount_image("/prebuilt", prebuilt_image)

        captured_requests = tcr_ctx.get_requests("TaskMountDirectory")
        assert len(captured_requests) == 1
        assert captured_requests[0].path == b"/prebuilt"
        assert captured_requests[0].image_id == prebuilt_image.object_id

        # Test mounting an image referenced by id
        image_from_id = Image.from_id(prebuilt_image.object_id, client=client)
        sb.mount_image("/from-id", image_from_id)

        captured_requests = tcr_ctx.get_requests("TaskMountDirectory")
        assert len(captured_requests) == 2
        assert captured_requests[1].path == b"/from-id"
        assert captured_requests[1].image_id == prebuilt_image.object_id

        # Test mounting a snapshot image from the current session
        snapshot_source = Sandbox.create(app=app)
        snapshot_image = snapshot_source.snapshot_filesystem()
        snapshot_source.terminate()

        sb.mount_image("/snapshot", snapshot_image)

        captured_requests = tcr_ctx.get_requests("TaskMountDirectory")
        assert len(captured_requests) == 3
        assert captured_requests[2].path == b"/snapshot"
        assert captured_requests[2].image_id == snapshot_image.object_id

        # Test validation: image created with non-copy add_local_file should raise
        image_with_mount_layer = prebuilt_image.add_local_file(Path(__file__), "/tmp/sandbox_test.py")
        sb_with_mount_layer = Sandbox.create(image=image_with_mount_layer, app=app)
        sb_with_mount_layer.terminate()

        with pytest.raises(InvalidError) as exc_info:
            sb.mount_image("/mount-layer", image_with_mount_layer)

        assert "add_local*" in str(exc_info.value)
        assert "copy=True" in str(exc_info.value)
        assert ".build()" in str(exc_info.value)
        assert len(tcr_ctx.get_requests("TaskMountDirectory")) == 3

        # Test validation: unbuilt images should raise with guidance
        unbuilt_image = Image.debian_slim()
        with pytest.raises(InvalidError) as exc_info:
            sb.mount_image("/unbuilt", unbuilt_image)
        assert "currently only supports Images that are either" in str(exc_info.value)
        assert len(tcr_ctx.get_requests("TaskMountDirectory")) == 3

        # Test validation: argument must be an Image instance
        with pytest.raises(TypeError, match="expects an Image"):
            sb.mount_image("/none", None)
        with pytest.raises(TypeError, match="expects an Image"):
            sb.mount_image("/bad-type", "not-an-image")  # type: ignore[arg-type]
        assert len(tcr_ctx.get_requests("TaskMountDirectory")) == 3

        # Test validation: non-absolute path should raise
        with pytest.raises(InvalidError, match="must be absolute"):
            sb.mount_image("relative/path", prebuilt_image)

    sb.terminate()


def test_mount_image_from_scratch_uses_empty_image_id(servicer, client, app):
    """Test mounting an explicit empty image via Image.from_scratch()."""
    sb = Sandbox.create(app=app)
    with servicer.task_command_router.intercept() as tcr_ctx:
        sb.mount_image("/empty", Image.from_scratch())

    captured_requests = tcr_ctx.get_requests("TaskMountDirectory")
    assert len(captured_requests) == 1
    assert captured_requests[0].path == b"/empty"
    assert captured_requests[0].image_id == ""

    sb.terminate()


def test_mount_image_customer_supplied_encryption_key(servicer, client, app):
    sb = Sandbox.create(app=app)
    csek = b"customer-supplied encryption key"
    image = Image.from_id("im-csek123", client=client)

    with servicer.task_command_router.intercept() as tcr_ctx:
        sb.mount_image("/csek", image, _experimental_encryption_key=csek)
        captured_requests = tcr_ctx.get_requests("TaskMountDirectory")
        assert len(captured_requests) == 1
        assert captured_requests[0].customer_supplied_encryption_key == csek

        with pytest.raises(TypeError, match="positional"):
            sb.mount_image("/bad-csek", image, csek)  # type: ignore[misc]
        with pytest.raises(TypeError, match="_experimental_encryption_key must be bytes"):
            sb.mount_image("/bad-csek", image, _experimental_encryption_key="not-bytes")  # type: ignore[arg-type]
        with pytest.raises(InvalidError, match="_experimental_encryption_key must not be empty"):
            sb.mount_image("/bad-csek", image, _experimental_encryption_key=b"")
        with pytest.raises(InvalidError, match="_experimental_encryption_key must be at least 16 bytes"):
            sb.mount_image("/bad-csek", image, _experimental_encryption_key=b"1" * 15)
        with pytest.raises(InvalidError, match="_experimental_encryption_key must be at most 512 bytes"):
            sb.mount_image("/bad-csek", image, _experimental_encryption_key=b"1" * 513)
        assert len(tcr_ctx.get_requests("TaskMountDirectory")) == 1

    sb.terminate()


def test_snapshot_directory(servicer, client, app):
    """Test snapshotting a directory to create a new image."""
    sb = Sandbox.create(app=app)

    with servicer.task_command_router.intercept() as tcr_ctx:
        # Create and snapshot a directory
        image = sb.snapshot_directory("/tmp")

        assert image.object_id == "im-snapshot-123"  # From mock

        # Verify snapshot_id is set
        captured_requests = tcr_ctx.get_requests("TaskSnapshotDirectory")
        assert len(captured_requests) == 1
        snapshot_id = captured_requests[0].snapshot_id
        assert snapshot_id, "snapshot_id must be non-empty"

        # Verify each call generates a unique snapshot_id
        sb.snapshot_directory("/var")
        captured_requests = tcr_ctx.get_requests("TaskSnapshotDirectory")
        assert len(captured_requests) == 2
        snapshot_id2 = captured_requests[1].snapshot_id
        assert snapshot_id2, "snapshot_id must be non-empty"
        assert snapshot_id != snapshot_id2, "Each snapshot call should generate a unique snapshot_id"

        # Test validation: non-absolute path should raise
        with pytest.raises(InvalidError, match="must be absolute"):
            sb.snapshot_directory("relative/path")

        # Default ttl is 30 days, sent as a positive ttl_seconds.
        assert captured_requests[0].HasField("ttl_seconds")
        assert captured_requests[0].ttl_seconds == 30 * 24 * 3600

        # Explicit ttl is forwarded as-is.
        sb.snapshot_directory("/tmp", ttl=3600)
        captured_requests = tcr_ctx.get_requests("TaskSnapshotDirectory")
        assert captured_requests[-1].HasField("ttl_seconds")
        assert captured_requests[-1].ttl_seconds == 3600

        # ttl=None opts out of expiry; the wire carries -1 as the sentinel.
        sb.snapshot_directory("/tmp", ttl=None)
        captured_requests = tcr_ctx.get_requests("TaskSnapshotDirectory")
        assert captured_requests[-1].HasField("ttl_seconds")
        assert captured_requests[-1].ttl_seconds == -1

        # 0 / negative ttl is rejected client-side.
        with pytest.raises(InvalidError, match="must be positive"):
            sb.snapshot_directory("/tmp", ttl=0)
        with pytest.raises(InvalidError, match="must be positive"):
            sb.snapshot_directory("/tmp", ttl=-5)

    sb.terminate()


def test_snapshot_directory_customer_supplied_encryption_key(servicer, client, app):
    sb = Sandbox.create(app=app)
    csek = b"customer-supplied encryption key"

    with servicer.task_command_router.intercept() as tcr_ctx:
        image = sb.snapshot_directory("/tmp", _experimental_encryption_key=csek)
        captured_requests = tcr_ctx.get_requests("TaskSnapshotDirectory")
        assert len(captured_requests) == 1
        assert captured_requests[0].customer_supplied_encryption_key == csek

        with pytest.raises(TypeError, match="positional"):
            getattr(sb, "snapshot_directory")("/tmp", csek)
        assert len(tcr_ctx.get_requests("TaskSnapshotDirectory")) == 1

        sb.mount_image("/csek-snapshot", image, _experimental_encryption_key=csek)
        mount_requests = tcr_ctx.get_requests("TaskMountDirectory")
        assert len(mount_requests) == 1
        assert mount_requests[0].customer_supplied_encryption_key == csek

        sb.mount_image("/csek-snapshot-without-key", image)
        mount_requests = tcr_ctx.get_requests("TaskMountDirectory")
        assert len(mount_requests) == 2
        assert not mount_requests[1].HasField("customer_supplied_encryption_key")

        with pytest.raises(TypeError, match="_experimental_encryption_key must be bytes"):
            sb.snapshot_directory("/tmp", _experimental_encryption_key="not-bytes")  # type: ignore[arg-type]
        with pytest.raises(InvalidError, match="_experimental_encryption_key must not be empty"):
            sb.snapshot_directory("/tmp", _experimental_encryption_key=b"")
        with pytest.raises(InvalidError, match="_experimental_encryption_key must be at least 16 bytes"):
            sb.snapshot_directory("/tmp", _experimental_encryption_key=b"1" * 15)
        with pytest.raises(InvalidError, match="_experimental_encryption_key must be at most 512 bytes"):
            sb.snapshot_directory("/tmp", _experimental_encryption_key=b"1" * 513)
        assert len(tcr_ctx.get_requests("TaskSnapshotDirectory")) == 1

    sb.terminate()


def test_sandbox_create_with_snapshot_directory_image(servicer, client, app):
    sb = Sandbox.create(app=app)
    image = sb.snapshot_directory("/tmp")
    sb.terminate()

    assert image.is_hydrated
    assert image.object_id == "im-snapshot-123"

    sb2 = Sandbox.create(image=image, app=app)
    sb2.terminate()

    assert servicer.sandbox_defs[-1].image_id == "im-snapshot-123"


def test_publish_sandbox_snapshot_images(servicer, client, app):
    sb = Sandbox.create(app=app)
    with servicer.task_command_router.intercept():
        filesystem_image = sb.snapshot_filesystem()
        directory_image = sb.snapshot_directory("/tmp")
    sb.terminate()

    snapshot_images = {
        "sandbox-filesystem-snapshot": filesystem_image,
        "sandbox-directory-snapshot": directory_image,
    }
    for name, image in snapshot_images.items():
        assert image.is_hydrated
        servicer.images[image.object_id] = api_pb2.Image()
        image.publish(name, client=client)
        assert servicer.image_tags[f"{name}:latest"] == image.object_id


def test_unmount_image(servicer, client, app):
    """Test unmounting an image from a path in the sandbox."""
    sb = Sandbox.create(app=app)

    with servicer.task_command_router.intercept() as tcr_ctx:
        sb.unmount_image("/mounted")

        captured_requests = tcr_ctx.get_requests("TaskUnmountDirectory")
        assert len(captured_requests) == 1
        assert captured_requests[0].path == b"/mounted"

        with pytest.raises(InvalidError, match="must be absolute"):
            sb.unmount_image("relative/path")

    sb.terminate()


def test_exec_on_terminate_sandbox_raises(servicer, client, app):
    sb = Sandbox.create(app=app)
    sb.terminate()

    with servicer.intercept() as ctx:
        ctx.add_response(
            "SandboxGetTaskId",
            api_pb2.SandboxGetTaskIdResponse(
                task_result=api_pb2.GenericResult(
                    status=api_pb2.GenericResult.GENERIC_STATUS_TERMINATED,
                    exception="Sandbox was cancelled by user",
                ),
            ),
        )
        with pytest.raises(ConflictError, match="Sandbox was cancelled by user"):
            sb.exec("echo", "hello")


# Values are None for deprecated methods
# Since we also programmatically assert that we cover all public API
detach_error_funcs = {
    "get_tags": lambda sb: sb.get_tags(),
    "set_tags": lambda sb: sb.set_tags({"hello": "world"}),
    "_experimental_set_outbound_network_policy": lambda sb: sb._experimental_set_outbound_network_policy(
        outbound_domain_allowlist=[], outbound_cidr_allowlist=[]
    ),
    "_experimental_set_name": lambda sb: sb._experimental_set_name("my-name"),
    "snapshot_filesystem": lambda sb: sb.snapshot_filesystem(),
    "snapshot_directory": lambda sb: sb.snapshot_directory("/tmp"),
    "tunnels": lambda sb: sb.tunnels(),
    "create_connect_token": lambda sb: sb.create_connect_token(),
    "reload_volumes": lambda sb: sb.reload_volumes(),
    "terminate": lambda sb: sb.terminate(),
    "wait_until_ready": lambda sb: sb.wait_until_ready(),
    "poll": lambda sb: sb.poll(),
    "_experimental_get_exit_snapshot": lambda sb: sb._experimental_get_exit_snapshot(timeout=0),
    "exec": lambda sb: sb.exec("echo", "hello"),
    "mount_image": lambda sb: sb.mount_image("/mnt", modal._image._Image.from_scratch()),
    "unmount_image": lambda sb: sb.unmount_image("/mnt"),
    "_experimental_snapshot": lambda sb: sb._experimental_snapshot(),
    "open": None,
    "ls": None,
    "mkdir": None,
    "rm": None,
    "watch": None,
    "_experimental_sidecars": lambda sb: sb._experimental_sidecars,
    "stdout": lambda sb: sb.stdout,
    "stderr": lambda sb: sb.stderr,
    "stdin": lambda sb: sb.stdin,
    "filesystem": lambda sb: sb.filesystem,
}
ALLOW_AFTER_DETACH = {"detach", "returncode", "wait"}


def test_func_map_covers_all_public_methods_and_properties():
    attributes_to_raise_on_detached = {
        attr.name
        for attr in inspect.classify_class_attrs(modal.sandbox._Sandbox)
        if attr.defining_class == modal.sandbox._Sandbox
        and (
            not (attr.name.startswith("_") or attr.name in ALLOW_AFTER_DETACH) or attr.name.startswith("_experimental")
        )
        and attr.kind in ("method", "property")
    }
    assert set(detach_error_funcs) == attributes_to_raise_on_detached


@pytest.mark.parametrize("name", detach_error_funcs)
def test_detach_errors(servicer, client, app, name):
    """Check that detached sandbox actually raise."""
    sb = Sandbox.create(app=app)
    sb.detach()

    func = detach_error_funcs[name]

    if func is not None:
        with pytest.raises(modal.exception.ClientClosed):
            func(sb)


def test_detach_twice(servicer, client, app):
    """Calling detach twice does **not** raise errors."""
    sb = Sandbox.create(app=app)
    sb.detach()
    sb.detach()


def test_sandbox_terminate_wait(app, servicer):
    sb = Sandbox.create("bash", "-c", "sleep 100", app=app)

    with servicer.intercept() as ctx:
        exit_code = sb.terminate(wait=True)
        req = ctx.pop_request("SandboxWait")

    assert req.sandbox_id == sb.object_id
    assert exit_code != 0


@skip_non_subprocess
def test_sandbox_container_terminate_wait(app, servicer):
    image = mock.Mock()
    image.object_id = "im-test-1"
    image._mount_layers = []

    sb = Sandbox.create("bash", "-c", "sleep 100", app=app)

    container = sb._experimental_sidecars.create("bash", "-c", "sleep 100", name="worker", image=image)

    assert container.poll() is None

    exit_code = container.terminate(wait=True)
    assert exit_code == 137
    assert container.poll() == 137

    with pytest.raises(modal.exception.SandboxTerminatedError):
        container.wait()

    container.wait(raise_on_termination=False)
    assert container.poll() == 137
    terminated = sb._experimental_sidecars.list(include_terminated=True)
    assert all(isinstance(listed_container, SidecarContainer) for listed_container in terminated)
    assert any(
        listed_container.name == "worker"
        and listed_container.object_id == container.object_id
        and listed_container.poll() == 137
        for listed_container in terminated
    )


@skip_non_subprocess
def test_sandbox_container_wait_after_natural_exit(app, servicer):
    image = mock.Mock()
    image.object_id = "im-test-1"
    image._mount_layers = []

    sb = Sandbox.create("bash", "-c", "sleep 100", app=app)

    container = sb._experimental_sidecars.create("bash", "-c", "exit 42", name="oneshot", image=image)
    container.wait()
    container.wait()
    assert container.poll() == 42

    with pytest.raises((modal.exception.NotFoundError, GRPCError)):
        sb._experimental_sidecars.get(name="oneshot")
    assert sb._experimental_sidecars.get(name="oneshot", include_terminated=True).object_id == container.object_id

    replacement = sb._experimental_sidecars.create("bash", "-c", "sleep 100", name="oneshot", image=image)
    assert replacement.object_id != container.object_id
    terminated = sb._experimental_sidecars.list(include_terminated=True)
    assert all(isinstance(listed_container, SidecarContainer) for listed_container in terminated)
    assert any(
        listed_container.name == "oneshot"
        and listed_container.object_id == container.object_id
        and listed_container.poll() == 42
        for listed_container in terminated
    )
    assert any(
        listed_container.name == "oneshot"
        and listed_container.object_id == replacement.object_id
        and listed_container.poll() is None
        for listed_container in terminated
    )

    container.terminate(wait=True)
    assert container.poll() == 42
    assert sb._experimental_sidecars.get(name="oneshot").object_id == replacement.object_id
    assert sb._experimental_sidecars.get(name="oneshot", include_terminated=True).object_id == replacement.object_id


@skip_non_subprocess
def test_sandbox_container_get_and_list_forward_include_terminated(app, servicer):
    image = mock.Mock()
    image.object_id = "im-test-1"
    image._mount_layers = []

    sb = Sandbox.create("bash", "-c", "sleep 100", app=app)
    with servicer.task_command_router.intercept() as tcr_ctx:
        container = sb._experimental_sidecars.create("bash", "-c", "exit 0", name="oneshot", image=image)
        container.wait()

        sb._experimental_sidecars.get(name="oneshot", include_terminated=True)
        containers = sb._experimental_sidecars.list(include_terminated=True)

    (get_request,) = tcr_ctx.get_requests("TaskContainerGet")
    assert get_request.include_terminated is True
    (list_request,) = tcr_ctx.get_requests("TaskContainerList")
    assert list_request.include_terminated is True
    assert all(isinstance(container, SidecarContainer) for container in containers)


@skip_non_subprocess
def test_sandbox_container_create_accepts_prebuilt_image(app, servicer):
    image = mock.Mock()
    image.object_id = "im-test-1"
    image._mount_layers = []

    sb = Sandbox.create("bash", "-c", "sleep 100", app=app)

    with servicer.task_command_router.intercept() as tcr_ctx:
        sb._experimental_sidecars.create("bash", "-c", "sleep 100", name="worker", image=image)

    (container_create_request,) = tcr_ctx.get_requests("TaskContainerCreate")
    assert container_create_request.image_id == image.object_id
    assert list(container_create_request.secret_ids) == []


@skip_non_subprocess
def test_sandbox_container_create_forwards_secret_ids_and_env(app, servicer):
    image = mock.Mock()
    image.object_id = "im-test-1"
    image._mount_layers = []
    secret = Secret.from_dict({"API_KEY": "secret-value"})

    sb = Sandbox.create("bash", "-c", "sleep 100", app=app)

    with servicer.task_command_router.intercept() as tcr_ctx:
        sb._experimental_sidecars.create(
            "bash",
            "-c",
            "sleep 100",
            name="worker",
            image=image,
            env={"API_KEY": "override", "PLAIN_ENV": "plain"},
            secrets=[secret],
        )

    (container_create_request,) = tcr_ctx.get_requests("TaskContainerCreate")
    assert container_create_request.image_id == image.object_id
    assert dict(container_create_request.env) == {"API_KEY": "override", "PLAIN_ENV": "plain"}
    assert list(container_create_request.secret_ids) == [secret.object_id]


def test_sandbox_container_create_rejects_image_with_mount_layers(app):
    image = mock.Mock()
    image._mount_layers = [mock.Mock()]
    image._object_id = None
    sb = Sandbox.create("bash", "-c", "sleep 100", app=app)

    with pytest.raises(InvalidError, match=r"only supports pre-built images"):
        sb._experimental_sidecars.create("bash", "-c", "sleep 100", name="worker", image=image)


def test_sandbox_container_create_rejects_unhydrated_image(app):
    image = Image.debian_slim()
    sb = Sandbox.create("bash", "-c", "sleep 100", app=app)

    with pytest.raises(InvalidError, match=r"currently only supports Images that are either"):
        sb._experimental_sidecars.create("bash", "-c", "sleep 100", name="worker", image=image)


def test_sandbox_container_create_rejects_mounts_kwarg(app):
    image = mock.Mock()
    image.object_id = "im-test-1"
    image._mount_layers = []

    sb = Sandbox.create("bash", "-c", "sleep 100", app=app)
    create = typing.cast(typing.Any, sb._experimental_sidecars.create)

    with pytest.raises(TypeError):
        create(name="worker", image=image, mounts=())


@skip_non_subprocess
def test_sandbox_container_volume_mounts(app, servicer):
    image = mock.Mock()
    image.object_id = "im-test-1"
    image._mount_layers = []

    sb = Sandbox.create("bash", "-c", "sleep 100", app=app)

    read_only_volume = Volume.from_name("sidecar-ro-volume", create_if_missing=True).with_mount_options(read_only=True)
    writable_volume = Volume.from_name("sidecar-rw-volume", create_if_missing=True)

    with servicer.task_command_router.intercept() as tcr_ctx:
        sb._experimental_sidecars.create(
            "bash",
            "-c",
            "sleep 100",
            name="worker",
            image=image,
            volumes={"/mnt/ro": read_only_volume, "/mnt/rw": writable_volume},
        )

    (req,) = tcr_ctx.get_requests("TaskContainerCreate")
    mounts_by_id = {mount.volume_id: mount for mount in req.volume_mounts}
    assert len(mounts_by_id) == 2

    ro_mount = mounts_by_id[read_only_volume.object_id]
    assert ro_mount.mount_path == "/mnt/ro"
    assert ro_mount.read_only is True

    # Writable volumes are now supported in sidecars.
    rw_mount = mounts_by_id[writable_volume.object_id]
    assert rw_mount.mount_path == "/mnt/rw"
    assert rw_mount.read_only is False


def test_sandbox_wait_allowed_after_detached(app, servicer):
    sb = Sandbox.create("bash", "-c", "sleep 100", app=app)
    sb.terminate()
    sb.detach()

    sb.wait(raise_on_termination=False)
    assert sb.returncode != 0


def _create_wait_until_ready_sandbox(app, sandbox_version: SandboxVersion, *, with_readiness_probe: bool = True):
    if sandbox_version == SandboxVersion.V2:
        return Sandbox._experimental_create("bash", "-c", "sleep 100", app=app)
    readiness_probe = modal.Probe.with_tcp(8080) if with_readiness_probe else None
    return Sandbox.create("bash", "-c", "sleep 100", app=app, readiness_probe=readiness_probe)


@pytest.mark.parametrize("sandbox_version", [SandboxVersion.V1, SandboxVersion.V2], ids=["v1", "v2"])
def test_sandbox_wait_until_ready(app, servicer, sandbox_version):
    sb = _create_wait_until_ready_sandbox(app, sandbox_version)

    with servicer.task_command_router.intercept() as tcr_ctx:
        tcr_ctx.add_response("SandboxWaitUntilReady", sr_pb2.SandboxWaitUntilReadyTcrResponse(ready_at=123.456))
        sb.wait_until_ready(timeout=5)
        req = tcr_ctx.pop_request("SandboxWaitUntilReady")

    assert req.task_id
    assert 0 < req.timeout <= 5

    sb.terminate()


@pytest.mark.parametrize("sandbox_version", [SandboxVersion.V1, SandboxVersion.V2], ids=["v1", "v2"])
def test_sandbox_wait_until_ready_times_out(app, servicer, sandbox_version):
    sb = _create_wait_until_ready_sandbox(app, sandbox_version)
    requests: list[sr_pb2.SandboxWaitUntilReadyTcrRequest] = []

    async def handle_wait_until_ready_request(servicer, stream):
        req: sr_pb2.SandboxWaitUntilReadyTcrRequest = await stream.recv_message()
        requests.append(req)
        # The worker signals that the sandbox did not become ready in time.
        raise GRPCError(Status.DEADLINE_EXCEEDED, "Timed out waiting for sandbox to become ready")

    try:
        with servicer.task_command_router.intercept() as tcr_ctx:
            tcr_ctx.set_responder("SandboxWaitUntilReady", handle_wait_until_ready_request)
            with pytest.raises(TimeoutError):
                sb.wait_until_ready(timeout=5)
            assert requests[0].task_id
            assert 0 < requests[0].timeout <= 5
    finally:
        sb.terminate()


@pytest.mark.parametrize("sandbox_version", [SandboxVersion.V1, SandboxVersion.V2], ids=["v1", "v2"])
def test_sandbox_wait_until_ready_no_probe_raises_invalid_error(app, servicer, sandbox_version):
    sb = _create_wait_until_ready_sandbox(app, sandbox_version, with_readiness_probe=False)

    async def handle_wait_until_ready_request(servicer, stream):
        await stream.recv_message()
        # The worker rejects waits on sandboxes created without a readiness probe.
        raise GRPCError(Status.FAILED_PRECONDITION, "Sandbox does not have a readiness probe configured")

    with servicer.task_command_router.intercept() as tcr_ctx:
        tcr_ctx.set_responder("SandboxWaitUntilReady", handle_wait_until_ready_request)
        with pytest.raises(InvalidError, match="readiness probe"):
            sb.wait_until_ready(timeout=5)

    sb.terminate()


def test_sandbox_create_reuses_hydrated_image(app, servicer):
    async def sandbox_create_no_subprocess(servicer_self, stream):
        request = await stream.recv_message()
        servicer_self.sandbox_defs.append(request.definition)
        await stream.send_message(
            api_pb2.SandboxCreateResponse(
                sandbox_id="sb-123",
                metadata=api_pb2.SandboxHandleMetadata(app_id=request.app_id),
            )
        )

    image = Image.debian_slim()

    with servicer.intercept() as ctx:
        ctx.set_responder("SandboxCreate", sandbox_create_no_subprocess)
        Sandbox.create("echo", "first", image=image, app=app)
        Sandbox.create("echo", "second", image=image, app=app)

        image_gets = ctx.get_requests("ImageGetOrCreate")
        assert len(image_gets) == 1, f"Expected 1 ImageGetOrCreate request, got {len(image_gets)}"


def test_sandbox_create_timing_log_caps_dependency_list():
    """The formatter caps the per-dep list at 10 entries (slowest first) and
    appends a `+N more` suffix for the remainder."""
    from modal.sandbox import _format_sandbox_create_timing_log

    deps = [(f"im-{i:03d}", float(i)) for i in range(15)]
    line = _format_sandbox_create_timing_log("sb-abc", 12.34, 0.5, deps)

    assert "+5 more" in line
    assert "im-014: 14.00s" in line
    assert "im-005: 5.00s" in line
    assert "im-004" not in line


def test_sandbox_create_logs_per_dependency_timing(app, servicer, caplog):
    """V1 Sandbox.create emits a debug log with the sandbox id, total + RPC
    elapsed, and per-dependency object_id timings.
    """
    import logging

    image = Image.debian_slim().add_local_python_source("modal", copy=True)
    secret = Secret.from_dict({"FOO": "bar"})

    with caplog.at_level(logging.DEBUG, logger="modal-client"):
        sb = Sandbox.create("echo", "hi", app=app, image=image, secrets=[secret])

    matching = [r for r in caplog.records if "created in" in r.message and "dependencies:" in r.message]
    assert matching, f"expected a sandbox-create debug log, got: {[r.message for r in caplog.records]}"
    msg = matching[-1].message
    assert sb.object_id in msg
    assert "create rpc:" in msg
    assert image.object_id in msg
    assert secret.object_id in msg

    sb.terminate()


def test_experimental_sandbox_create_logs_per_dependency_timing(app, servicer, client, caplog):
    """V2 Sandbox._experimental_create emits the same per-dependency debug log."""
    import logging

    image = Image.debian_slim().add_local_python_source("modal", copy=True)
    # Use a named secret: `Secret.from_dict` is inlined as an ephemeral secret and never
    # resolved as a dependency, so it would not appear in the per-dependency timing log.
    Secret.objects.create("my-secret", {"FOO": "bar"}, client=client)
    secret = Secret.from_name("my-secret")

    with caplog.at_level(logging.DEBUG, logger="modal-client"):
        sb = Sandbox._experimental_create("echo", "hi", app=app, image=image, secrets=[secret])

    matching = [r for r in caplog.records if "created in" in r.message and "dependencies:" in r.message]
    assert matching, f"expected a sandbox-create debug log, got: {[r.message for r in caplog.records]}"
    msg = matching[-1].message
    assert sb.object_id in msg
    assert "create rpc:" in msg
    assert image.object_id in msg
    assert secret.object_id in msg

    sb.terminate()
