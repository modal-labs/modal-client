# Copyright Modal Labs 2026
import asyncio
import io
import pickle
import pytest
import sys
from unittest import mock

from grpclib import GRPCError, Status

import modal
from modal import App, Image, Sandbox
from modal._serialization import deserialize, serialize
from modal._utils.async_utils import synchronize_api, synchronizer
from modal.exception import (
    ClientClosed,
    DeprecationError,
    InvalidError,
    NotFoundError,
    ResourceExhaustedError,
    SandboxPaused,
    SandboxTerminatedError,
    ServiceError,
)
from modal.experimental import OutboundPolicy
from modal.sandbox._impl_persistent import (
    PERSISTENT_SANDBOX_DRAIN_TIMEOUT,
    PERSISTENT_SANDBOX_SESSION_IDLE_TIMEOUT,
    PERSISTENT_SANDBOX_SESSION_TIMEOUT,
)
from modal_proto import api_pb2, task_command_router_pb2 as sr_pb2

from .supports.skip import skip_windows

# Covers the outbound network APIs that `outbound_policy` and `Sandbox.set_outbound_policy` replace.
ignore_deprecated_network_apis = pytest.mark.filterwarnings(
    "ignore:.*(`outbound_(cidr|domain)_allowlist`|`_experimental_outbound_policy`"
    "|`Sandbox._experimental_(set_outbound_network_policy|update_outbound_policy)`) is deprecated"
    ":modal.exception.DeprecationError"
)


@pytest.fixture
def app(client):
    return App.lookup("persistent-app", create_if_missing=True, client=client)


def _impl(sb):
    return synchronizer._translate_in(sb)._impl


def _create(app, client, **kwargs):
    kwargs.setdefault("timeout", None)
    return Sandbox.create(app=app, _experimental_persistent=True, client=client, **kwargs)


@pytest.mark.parametrize("running_loop", [False, True], ids=["sync-thread", "asyncio-run"])
def test_persistent_sandbox_from_id_stdio_constructed_off_loop(app, client, servicer, running_loop):
    sandbox_id = _create(app, client).object_id
    assert sandbox_id.startswith("sx-")

    def construct():
        sandbox = Sandbox.from_id(sandbox_id, client=client)
        return sandbox.stdout, sandbox.stdin

    async def construct_in_loop():
        assert asyncio.get_running_loop() != synchronizer._get_loop(start=False)
        return construct()

    with servicer.intercept() as ctx, servicer.task_command_router.intercept() as tcr_ctx:
        reader, writer = asyncio.run(construct_in_loop()) if running_loop else construct()
        assert ctx.calls == []
        assert tcr_ctx.calls == []
        tcr_ctx.add_response(
            "SandboxStdioReadV2", sr_pb2.SandboxStdioReadV2Response(data=b"output\n", starting_offset=0)
        )
        tcr_ctx.add_response("SandboxStdinWriteV2", sr_pb2.SandboxStdinWriteV2Response())
        assert reader.read() == "output\n"
        writer.write(b"input")
        writer.write_eof()
        writer.drain()
        (request,) = tcr_ctx.get_requests("SandboxStdinWriteV2")
        assert request.data == b"input"
        assert request.eof


@synchronize_api
async def _deserialize(data, client):
    return deserialize(data, client)


@pytest.mark.parametrize("terminated", [False, True])
@pytest.mark.parametrize("from_id", [False, True])
def test_persistent_sandbox_id_roundtrip(app, client, servicer, terminated, from_id):
    sb = _create(app, client)
    if terminated:
        sb.terminate()
    if from_id:
        sb = Sandbox.from_id(sb.object_id, client=client)

    class MetadataUnpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            return pid

    data = serialize(sb)
    assert MetadataUnpickler(io.BytesIO(data)).load() == (sb.object_id, "o", None)
    with servicer.intercept() as ctx:
        restored = _deserialize(data, client)
        assert restored.object_id == sb.object_id
        assert not ctx.get_requests("SxSandboxGet")
        assert restored.poll() == sb.poll()
        assert len(ctx.get_requests("SxSandboxGet")) == (1 if terminated and not from_id else 2)
        assert not ctx.get_requests("SxSandboxResume")
        if terminated:
            with pytest.raises(InvalidError, match="has been terminated"):
                restored._experimental_resume()


@pytest.mark.parametrize("has_result", [False, True])
def test_persistent_sandbox_ignores_serialized_metadata(app, client, servicer, has_result):
    sandbox = _create(app, client)

    class LegacyPickler(pickle.Pickler):
        def persistent_id(self, obj):
            if obj is sandbox:
                return (
                    sandbox.object_id,
                    "o",
                    api_pb2.SandboxHandleMetadata(
                        app_id="ap-stale",
                        result=api_pb2.GenericResult(
                            status=(
                                api_pb2.GenericResult.GENERIC_STATUS_TERMINATED
                                if has_result
                                else api_pb2.GenericResult.GENERIC_STATUS_UNSPECIFIED
                            )
                        ),
                    ),
                )
            return None

    buf = io.BytesIO()
    LegacyPickler(buf).dump(sandbox)
    with servicer.intercept() as ctx:
        restored = _deserialize(buf.getvalue(), client)
        assert restored.returncode is None
        assert not ctx.get_requests("SxSandboxGet")
        assert restored.poll() is None
        assert len(ctx.get_requests("SxSandboxGet")) == 1
        assert not ctx.get_requests("SxSandboxResume")


@pytest.mark.parametrize("terminated", [False, True])
@skip_windows("Needs subprocess support")
def test_deserialized_persistent_sandbox_adopts_lookup_session(app, client, servicer, terminated):
    sb = _create(app, client)
    if terminated:
        sb.terminate()
    with servicer.intercept() as ctx, servicer.task_command_router.intercept() as tcr_ctx:
        restored = _deserialize(serialize(sb), client)
        assert not ctx.get_requests("SxSandboxGet")
        if terminated:
            with pytest.raises(InvalidError, match="has been terminated"):
                restored.exec("echo", "hello")
        else:
            process = restored.exec("echo", "hello")
            assert process.stdout.read() == "hello\n"
            assert process.wait() == 0
            assert len(tcr_ctx.get_requests("TaskExecStart")) == 1
        assert len(ctx.get_requests("SxSandboxGet")) == 1
        assert not ctx.get_requests("SxSandboxResume")


def test_deserialized_persistent_sandbox_unknown_id(app, client, servicer):
    sb = _create(app, client)
    restored = _deserialize(serialize(sb), client)
    servicer.persistent_sandbox_create_requests.clear()
    with pytest.raises(NotFoundError):
        restored.poll()


def test_deserialized_persistent_sandbox_wait_uses_lookup(app, client, servicer):
    sb = _create(app, client)
    sb.terminate()
    with servicer.intercept() as ctx:
        restored = _deserialize(serialize(sb), client)
        restored.wait(raise_on_termination=False)
        assert restored.returncode is not None
        assert len(ctx.get_requests("SxSandboxGet")) == 1
        assert not ctx.get_requests("SxSandboxResume")


def test_persistent_sandbox_create(app, client, servicer):
    sb = _create(app, client)

    assert sb.object_id == "sx-0000000000000000000000001"
    assert _impl(sb).app_id == app.app_id
    [req] = servicer.persistent_sandbox_create_requests
    assert not req.definition.name
    assert req.definition.timeout_secs == PERSISTENT_SANDBOX_SESSION_TIMEOUT
    assert req.definition.idle_timeout_secs == PERSISTENT_SANDBOX_SESSION_IDLE_TIMEOUT
    assert req.definition.drain_timeout_secs == PERSISTENT_SANDBOX_DRAIN_TIMEOUT
    assert not servicer.sandbox_create_v2_requests
    assert sb.poll() is None


@pytest.mark.parametrize("explicit_client", [False, True])
def test_persistent_sandbox_hydrates_client(app, client, container_client, servicer, monkeypatch, explicit_client):
    sb = _create(app, client)
    monkeypatch.setattr(synchronizer._translate_in(client), "_snapshotted", True)
    from_env = mock.AsyncMock(return_value=synchronizer._translate_in(container_client))
    monkeypatch.setattr("modal.client._Client.from_env", from_env)
    with servicer.intercept() as ctx:
        sb.hydrate(container_client if explicit_client else None)
        assert sb.client is container_client
        assert sb._is_hydrated
        assert ctx.calls == []
    assert from_env.await_count == (0 if explicit_client else 1)


def test_persistent_sandbox_drain_timeout(app, client, servicer):
    _create(app, client, experimental_options={"drain_timeout_secs": 300, "enable_exit_snapshot": True})

    [req] = servicer.persistent_sandbox_create_requests
    assert req.definition.drain_timeout_secs == 300
    assert dict(req.definition.experimental_options_v2) == {"enable_exit_snapshot": "True"}


def test_persistent_sandbox_rejects_drain_timeout_of_wrong_type(app, client, servicer):
    with pytest.raises(InvalidError, match="must be an int"):
        _create(app, client, experimental_options={"drain_timeout_secs": "60"})
    assert not servicer.persistent_sandbox_create_requests


@pytest.mark.parametrize("drain_timeout", [0, 601])
def test_persistent_sandbox_rejects_drain_timeout_out_of_range(app, client, servicer, drain_timeout):
    with pytest.raises(InvalidError, match="must be between 1 and 600 seconds"):
        _create(app, client, experimental_options={"drain_timeout_secs": drain_timeout})
    assert not servicer.persistent_sandbox_create_requests


def test_regular_sandbox_rejects_drain_timeout(app, client, servicer):
    with pytest.raises(InvalidError, match="only supported for experimental persistent Sandboxes"):
        Sandbox.create(app=app, experimental_options={"drain_timeout_secs": 60}, client=client)
    assert not servicer.sandbox_create_v2_requests


def test_persistent_sandbox_rejects_name(app, client, servicer):
    with pytest.raises(InvalidError, match="Named persistent Sandboxes are not supported"):
        _create(app, client, name="my-persistent-sandbox")
    assert not servicer.persistent_sandbox_create_requests


@pytest.mark.parametrize(
    "kwargs",
    [{"timeout": 300}, {"timeout": 600}, {"timeout": None, "idle_timeout": 60}],
)
def test_persistent_sandbox_rejects_timeouts(app, client, servicer, kwargs):
    with pytest.raises(InvalidError, match="require `timeout=None` and don't support `idle_timeout`"):
        _create(app, client, **kwargs)
    assert not servicer.persistent_sandbox_create_requests


def test_persistent_sandbox_rejects_default_timeout(app, client):
    with pytest.raises(InvalidError, match="require `timeout=None`"):
        Sandbox.create(app=app, _experimental_persistent=True, client=client)


def test_regular_sandbox_rejects_timeout_none(app, client):
    with pytest.raises(InvalidError, match="only supported for experimental persistent Sandboxes"):
        Sandbox.create(app=app, timeout=None, client=client)


@pytest.mark.parametrize("kwargs", [{"gpu": "any"}, {"env": {"A": "1"}}, {"tags": {"a": "b"}}])
def test_persistent_sandbox_rejects_unsupported_options(app, client, servicer, kwargs):
    with pytest.raises(InvalidError, match="not supported by persistent Sandboxes"):
        _create(app, client, **kwargs)
    assert not servicer.persistent_sandbox_create_requests


def test_persistent_sandbox_pause_resume(app, client, servicer):
    sb = _create(app, client)

    sb._experimental_pause()
    assert servicer._sandbox_terminated
    assert sb.poll() is None

    sb._experimental_resume()
    assert not servicer._sandbox_terminated
    assert servicer.persistent_sandbox_resumes == 1

    # Resuming a running persistent Sandbox keeps its session.
    sb._experimental_resume()
    assert servicer.persistent_sandbox_resumes == 1


@skip_windows("Needs subprocess support")
def test_persistent_sandbox_requires_explicit_resume_on_failed_operation(app, client, servicer):
    sb = _create(app, client)
    sb._experimental_pause()

    with servicer.intercept() as ctx, servicer.task_command_router.intercept() as tcr_ctx:
        tcr_ctx.add_response("TaskExecStart", GRPCError(Status.NOT_FOUND, "sandbox has stopped"))
        with pytest.raises(SandboxPaused):
            sb.exec("echo", "hello")
        assert not ctx.get_requests("SxSandboxResume")
        assert len(tcr_ctx.get_requests("TaskExecStart")) == 1

        sb._experimental_resume()
        p = sb.exec("echo", "hello")
        assert p.stdout.read() == "hello\n"
        assert p.wait() == 0

        assert len(ctx.get_requests("SxSandboxResume")) == 1
        assert len(tcr_ctx.get_requests("TaskExecStart")) == 2
    assert servicer.persistent_sandbox_resumes == 1


@skip_windows("Needs subprocess support")
def test_persistent_sandbox_from_id_requires_resume_to_switch_sessions(app, client, servicer):
    sb = _create(app, client)
    sb2 = Sandbox.from_id(sb.object_id, client=client)
    sb2.poll()
    sb._experimental_pause()
    servicer.sandbox_task_ids["sb-v2-resumed"] = "ta-v2-resumed"
    replacement = api_pb2.SxSandboxGetResponse(sx_id=sb.object_id, current_sandbox_id="sb-v2-resumed")

    with servicer.intercept() as ctx, servicer.task_command_router.intercept() as tcr_ctx:
        ctx.add_response("SxSandboxGet", replacement)
        tcr_ctx.add_response("TaskExecStart", GRPCError(Status.NOT_FOUND, "task no longer exists"))
        assert sb2.poll() is None
        ctx.add_response("SxSandboxGet", replacement)
        with pytest.raises(SandboxPaused):
            sb2.exec("echo", "hello")
        assert not ctx.get_requests("SxSandboxResume")

        ctx.add_response("SxSandboxResume", api_pb2.SxSandboxResumeResponse(sandbox_id="sb-v2-resumed"))
        sb2._experimental_resume()
        p = sb2.exec("echo", "hello")
        assert p.stdout.read() == "hello\n"
        assert p.wait() == 0

    [resume] = ctx.get_requests("SxSandboxResume")
    assert resume.sx_id == sb.object_id
    first, retried = tcr_ctx.get_requests("TaskExecStart")
    assert first.task_id == "ta-v2-123"
    assert retried.task_id == "ta-v2-resumed"


@skip_windows("Needs subprocess support")
def test_persistent_sandbox_from_id_running(app, client, servicer):
    sb = _create(app, client)

    with servicer.intercept() as ctx:
        sb2 = Sandbox.from_id(sb.object_id, client=client)
        assert ctx.calls == []
        assert sb2.object_id == sb.object_id
        assert type(_impl(sb2)) is type(_impl(sb))
        assert sb2.poll() is None
        p = sb2.exec("echo", "hello")
        assert p.stdout.read() == "hello\n"
        assert p.wait() == 0
    # The running session is adopted directly, without resuming.
    assert len(ctx.get_requests("SxSandboxGet")) == 1
    assert not ctx.get_requests("SxSandboxResume")


@skip_windows("Needs subprocess support")
def test_persistent_sandbox_from_id_paused(app, client, servicer):
    sb = _create(app, client)
    sb._experimental_pause()

    sb2 = Sandbox.from_id(sb.object_id, client=client)
    assert sb2.poll() is None

    # Pausing an already paused Sandbox is a no-op and doesn't resume it.
    sb2._experimental_pause()
    assert servicer.persistent_sandbox_resumes == 0

    with servicer.intercept() as ctx, servicer.task_command_router.intercept() as tcr_ctx:
        tcr_ctx.add_response("TaskExecStart", GRPCError(Status.NOT_FOUND, "sandbox has stopped"))
        with pytest.raises(SandboxPaused):
            sb2.exec("echo", "hello")
        assert not ctx.get_requests("SxSandboxResume")
        assert len(tcr_ctx.get_requests("TaskExecStart")) == 1


def test_persistent_sandbox_from_id_terminated(app, client, servicer):
    sb = _create(app, client)
    sb.terminate()

    with servicer.intercept() as ctx:
        sb2 = Sandbox.from_id(sb.object_id, client=client)
        assert sb2.returncode is None
        assert ctx.calls == []
        assert sb2.poll() is not None
        with pytest.raises(InvalidError, match="has been terminated"):
            sb2.exec("echo", "hello")
        with pytest.raises(InvalidError, match="has been terminated"):
            sb2._experimental_resume()
        with pytest.raises(InvalidError, match="has been terminated"):
            sb2._experimental_pause()
    # The termination timestamp comes from the lookup; nothing is resumed.
    assert not ctx.get_requests("SxSandboxResume")


@pytest.mark.parametrize("state", ["running", "paused", "terminated"])
def test_persistent_sandbox_from_id_poll_one_rpc(app, client, servicer, state):
    original = _create(app, client)
    if state == "paused":
        original._experimental_pause()
    elif state == "terminated":
        original.terminate()

    with servicer.intercept() as ctx:
        sb = Sandbox.from_id(original.object_id, client=client)
        assert sb.returncode is None
        assert ctx.calls == []
        code = sb.poll()
        assert code == (137 if state == "terminated" else None)
        assert sb.returncode == code
    assert [rpc for rpc, _ in ctx.calls] == ["SxSandboxGet"]


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_client", [False, True])
async def test_persistent_sandbox_from_id_is_not_async(app, client, servicer, monkeypatch, explicit_client):
    original = await Sandbox.create.aio(app=app, client=client, timeout=None, _experimental_persistent=True)
    from_env = mock.AsyncMock(return_value=synchronizer._translate_in(client))
    monkeypatch.setattr("modal.client._Client.from_env", from_env)
    with servicer.intercept() as ctx:
        sandbox = Sandbox.from_id(original.object_id, client=client if explicit_client else None)
        with pytest.warns(DeprecationError, match=r"Please use Sandbox\.from_id\("):
            deprecated = await Sandbox.from_id.aio(original.object_id, client=client if explicit_client else None)
        assert isinstance(deprecated, Sandbox)
        assert sandbox.object_id == original.object_id
        assert sandbox.returncode is None
        from_env.assert_not_called()
        assert ctx.calls == []
        assert await sandbox.poll.aio() is None
    assert [rpc for rpc, _ in ctx.calls] == ["SxSandboxGet"]


def test_persistent_sandbox_from_id_not_found(client, servicer):
    sb = Sandbox.from_id("sx-000000000000000000000000z", client=client)
    with pytest.raises(NotFoundError):
        sb.poll()
    assert not servicer.persistent_sandbox_create_requests


@skip_windows("Needs subprocess support")
def test_persistent_sandbox_exec_reports_a_shutting_down_session(app, client, servicer):
    sb = _create(app, client)

    with servicer.intercept() as ctx, servicer.task_command_router.intercept() as tcr_ctx:
        tcr_ctx.add_response(
            "TaskExecStart",
            GRPCError(Status.FAILED_PRECONDITION, "Modal Sandbox is shutting down.", [sr_pb2.TaskShuttingDown()]),
        )
        with pytest.raises(SandboxPaused):
            sb.exec("echo", "hello")
        assert not ctx.get_requests("SxSandboxResume")
        assert len(tcr_ctx.get_requests("TaskExecStart")) == 1


@pytest.mark.parametrize(
    "rejection",
    [
        (
            "SandboxGetTaskIdV2",
            api_pb2.SandboxGetTaskIdResponse(
                task_result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_TERMINATED)
            ),
        ),
        ("SandboxGetCommandRouterAccess", GRPCError(Status.FAILED_PRECONDITION, "Sandbox has finished")),
    ],
    ids=["task-finished", "router-access-refused"],
)
@skip_windows("Needs subprocess support")
def test_persistent_sandbox_exec_reports_a_stopped_session_it_cannot_reach(app, client, servicer, rejection):
    sb = _create(app, client)
    servicer._sandbox_terminated = True

    with servicer.intercept() as ctx, servicer.task_command_router.intercept() as tcr_ctx:
        ctx.add_response(*rejection)

        with pytest.raises(SandboxPaused):
            sb.exec("echo", "hello")
        assert not ctx.get_requests("SxSandboxResume")
        assert not tcr_ctx.get_requests("TaskExecStart")


def test_persistent_sandbox_exec_not_repeated_after_it_may_have_run(app, client, servicer):
    sb = _create(app, client)

    with servicer.intercept() as ctx, servicer.task_command_router.intercept() as tcr_ctx:
        servicer._sandbox_terminated = True
        # Nothing in this error says the command didn't run before the session stopped.
        tcr_ctx.add_response("TaskExecStart", GRPCError(Status.RESOURCE_EXHAUSTED, "exec failed"))
        with pytest.raises(ResourceExhaustedError, match="exec failed"):
            sb.exec("echo", "hello")

        assert ctx.get_requests("SxSandboxResume") == []
        assert len(tcr_ctx.get_requests("TaskExecStart")) == 1


@pytest.mark.parametrize("terminal_status", [Status.NOT_FOUND, Status.FAILED_PRECONDITION])
@skip_windows("Needs subprocess support")
def test_persistent_exec_keeps_uncertainty_across_retries(app, client, servicer, tmp_path, terminal_status):
    sb = _create(app, client)
    marker = tmp_path / "executions"
    command = (
        sys.executable,
        "-c",
        "import sys\nwith open(sys.argv[1], 'a') as f:\n    f.write('ran\\n')\n",
        str(marker),
    )
    attempts = []

    async def exec_start(_router, stream):
        request = await stream.recv_message()
        attempts.append(request)
        if len(attempts) == 2:
            details = [sr_pb2.TaskShuttingDown()] if terminal_status == Status.FAILED_PRECONDITION else None
            raise GRPCError(terminal_status, "session stopped", details)

        # A start succeeds before its acknowledgement is lost. Replaying the
        # command in another session repeats its externally visible effect.
        proc = await asyncio.create_subprocess_exec(*request.command_args)
        assert await proc.wait() == 0
        if len(attempts) == 1:
            servicer._sandbox_terminated = True
            raise GRPCError(Status.UNAVAILABLE, "start acknowledgement lost")
        await stream.send_message(sr_pb2.TaskExecStartResponse())

    with servicer.intercept() as ctx, servicer.task_command_router.intercept() as tcr_ctx:
        tcr_ctx.set_responder("TaskExecStart", exec_start)
        error = None
        try:
            sb.exec(*command)
        except modal.Error as exc:
            error = exc

        assert marker.read_text().splitlines() == ["ran"]
        assert ctx.get_requests("SxSandboxResume") == []
        assert isinstance(error, ServiceError)
        assert error._grpc_status == Status.UNAVAILABLE
        assert attempts[0].exec_id == attempts[1].exec_id


@pytest.mark.timeout(5)
def test_persistent_sandbox_detached_does_not_resume(app, client, servicer):
    sb = _create(app, client)
    stdout = sb.stdout
    sb._experimental_pause()
    # Forget the session's task ID, so using the stream would have to look the task up again.
    session = _impl(sb)._session
    assert session is not None
    session._task_id = None
    sb.detach()

    with servicer.intercept() as ctx:
        with pytest.raises(ClientClosed):
            stdout.read()
    assert ctx.get_requests("SandboxGetTaskIdV2") == []
    assert ctx.get_requests("SxSandboxGet") == []
    assert ctx.get_requests("SxSandboxResume") == []
    assert servicer.persistent_sandbox_resumes == 0


def test_persistent_sandbox_terminate(app, client, servicer):
    sb = _create(app, client)
    sb.terminate()

    assert servicer.persistent_sandbox_terminated == {"sx-0000000000000000000000001"}
    assert sb.poll() is not None
    with pytest.raises(InvalidError, match="has been terminated"):
        sb._experimental_pause()
    with pytest.raises(InvalidError, match="has been terminated"):
        sb._experimental_resume()


@ignore_deprecated_network_apis
def test_persistent_sandbox_unsupported_operations(app, client):
    sb = _create(app, client)
    with pytest.raises(NotImplementedError):
        sb._experimental_update_outbound_policy(OutboundPolicy())
    with pytest.raises(NotImplementedError, match="Setting the outbound policy"):
        sb.set_outbound_policy(modal.OutboundPolicy(routes=[modal.OutboundRoute.any()]))
    with pytest.raises(NotImplementedError):
        sb.mount_image("/mnt", Image.from_id("im-123", client=client))
    with pytest.raises(NotImplementedError):
        sb.unmount_image("/mnt")
    with pytest.raises(NotImplementedError, match="Tags"):
        sb.get_tags()
    with pytest.raises(NotImplementedError, match="Statistics"):
        sb._experimental_stats()
    with pytest.raises(NotImplementedError, match="Tags"):
        sb.set_tags({"a": "b"})
    with pytest.raises(NotImplementedError, match="Memory snapshots"):
        sb._experimental_snapshot()
    with pytest.raises(NotImplementedError, match="network access"):
        sb._experimental_set_outbound_network_policy(outbound_cidr_allowlist=["10.0.0.0/8"])
    with pytest.raises(InvalidError, match="can't be changed"):
        sb._experimental_set_name("other")


@pytest.mark.parametrize("sandbox_v2", [False, True])
def test_regular_sandbox_pause_resume_unsupported(app, client, sandbox_v2, monkeypatch):
    monkeypatch.setenv("MODAL_SANDBOX_V2", "1" if sandbox_v2 else "0")
    sb = Sandbox.create(app=app, client=client)
    with pytest.raises(NotImplementedError, match="requires a persistent Sandbox"):
        sb._experimental_pause()
    with pytest.raises(NotImplementedError, match="requires a persistent Sandbox"):
        sb._experimental_resume()


def test_persistent_sandbox_terminated_result(app, client):
    sb = _create(app, client)
    sb.terminate()
    sb.wait(raise_on_termination=False)
    assert _impl(sb).result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED


@pytest.mark.parametrize("paused", [False, True])
def test_persistent_sandbox_poll_observes_external_termination(app, client, servicer, paused):
    sandbox_a = _create(app, client)
    if paused:
        sandbox_a._experimental_pause()
    sandbox_b = Sandbox.from_id(sandbox_a.object_id, client=client)
    assert sandbox_b.poll() is None
    sandbox_a.terminate()
    with servicer.intercept() as ctx:
        assert sandbox_b.poll() == 137
        assert sandbox_b.poll() == 137
        assert len(ctx.get_requests("SxSandboxGet")) == 1
        assert not ctx.get_requests("SxSandboxResume")


@pytest.mark.asyncio
@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("raise_on_termination", [False, True])
async def test_persistent_sandbox_wait_observes_external_termination(
    app, client, servicer, paused, raise_on_termination
):
    sandbox_a = await Sandbox.create.aio(app=app, client=client, timeout=None, _experimental_persistent=True)
    if paused:
        await sandbox_a._experimental_pause.aio()
    sandbox_b = Sandbox.from_id(sandbox_a.object_id, client=client)
    first_get = True

    async def get(self, stream):
        nonlocal first_get
        await stream.recv_message()
        response = api_pb2.SxSandboxGetResponse()
        if first_get:
            first_get = False
            await sandbox_a.terminate.aio()
        else:
            response.terminated_at.SetInParent()
        await stream.send_message(response)

    with servicer.intercept() as ctx:
        ctx.set_responder("SxSandboxGet", get)
        waited = sandbox_b.wait.aio(raise_on_termination=raise_on_termination)
        if raise_on_termination:
            with pytest.raises(SandboxTerminatedError):
                await asyncio.wait_for(waited, 3)
        else:
            await asyncio.wait_for(waited, 3)
        assert sandbox_b.returncode == 137
        assert len(ctx.get_requests("SxSandboxGet")) == 2
        assert not ctx.get_requests("SxSandboxResume")


@pytest.mark.parametrize("operation", ["poll", "wait"])
def test_persistent_sandbox_refresh_error(app, client, servicer, operation):
    sandbox = _create(app, client)
    servicer.persistent_sandbox_create_requests.clear()
    with pytest.raises(NotFoundError):
        getattr(sandbox, operation)()


@pytest.mark.parametrize("stream_name", ["stdin", "stdout", "stderr"])
@pytest.mark.parametrize("serialized", [False, True])
def test_detached_sessionless_persistent_stream_does_not_resume(app, client, servicer, stream_name, serialized):
    sandbox = _create(app, client)
    sandbox._experimental_pause()
    if serialized:
        sandbox = _deserialize(serialize(sandbox), client)
    else:
        sandbox = Sandbox.from_id(sandbox.object_id, client=client)
    stream = getattr(sandbox, stream_name)
    sandbox.detach()
    with servicer.intercept() as ctx:
        with pytest.raises(ClientClosed, match="detached sandbox"):
            if stream_name == "stdin":
                stream.write(b"hello")
                stream.drain()
            else:
                stream.read()
        assert not ctx.get_requests("SxSandboxGet")
        assert not ctx.get_requests("SxSandboxResume")


def test_persistent_sandbox_tunnels(app, client, servicer):
    sb = _create(app, client, encrypted_ports=[8080, 8443])

    tunnels = sb.tunnels()
    assert {port: tunnel.host for port, tunnel in tunnels.items()} == {
        8080: "0000000000000000000000001-8080.tunnel.modal.host",
        8443: "0000000000000000000000001-8443.tunnel.modal.host",
    }


def test_persistent_sandbox_pause_stops_session_started_elsewhere(app, client, servicer):
    sb = _create(app, client)

    with servicer.intercept() as ctx:
        # Another handle paused and resumed the Sandbox, so a different session is running.
        ctx.add_response("SxSandboxPause", api_pb2.SxSandboxPauseResponse(sandbox_id="sb-v2-other"))
        ctx.add_response(
            "SandboxWaitV2",
            api_pb2.SandboxWaitResponse(
                result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_TERMINATED)
            ),
        )
        sb._experimental_pause()
    # Pause waits for the session it stopped, not the one this handle started.
    assert [req.sandbox_id for req in ctx.get_requests("SandboxWaitV2")] == ["sb-v2-other"]
    assert ctx.get_requests("SxSandboxGet") == []
    assert ctx.get_requests("SandboxTerminateV2") == []
    assert ctx.get_requests("SxSandboxResume") == []


def test_persistent_sandbox_pause_does_not_resume(app, client, servicer):
    sb = _create(app, client)
    sb._experimental_pause()

    with servicer.intercept() as ctx:
        sb._experimental_pause()
    # Pausing a paused Sandbox stops nothing, so there is nothing to wait for.
    assert [req.sx_id for req in ctx.get_requests("SxSandboxPause")] == [sb.object_id]
    assert ctx.get_requests("SandboxWaitV2") == []
    assert ctx.get_requests("SandboxTerminateV2") == []
    assert ctx.get_requests("SxSandboxResume") == []


def test_persistent_sandbox_pause_retries_while_starting(app, client, servicer):
    sb = _create(app, client)

    with servicer.intercept() as ctx:
        # A session is still starting, so the first attempt is turned away.
        ctx.add_response("SxSandboxPause", GRPCError(Status.UNAVAILABLE, "starting"))
        sb._experimental_pause()
    assert len(ctx.get_requests("SxSandboxPause")) == 2
    assert servicer._sandbox_terminated
    assert servicer.persistent_sandbox_resumes == 0


@pytest.mark.parametrize("from_id", [False, True])
def test_persistent_sandbox_exit_snapshot_requires_termination(app, client, servicer, from_id):
    sb = _create(app, client)
    sb._experimental_pause()
    if from_id:
        sb = Sandbox.from_id(sb.object_id, client=client)

    # A paused Sandbox has not exited: it is neither resumed nor asked for a snapshot.
    with servicer.intercept() as ctx:
        with pytest.raises(modal.exception.TimeoutError):
            sb._experimental_get_exit_snapshot(timeout=0)
    assert ctx.get_requests("SxSandboxResume") == []
    assert ctx.get_requests("SandboxGetExitSnapshot") == []


@pytest.mark.parametrize("terminated_by", ["this_handle", "other_handle"])
def test_persistent_sandbox_from_id_exit_snapshot_after_terminate(app, client, servicer, terminated_by):
    created = _create(app, client)
    sb = Sandbox.from_id(created.object_id, client=client)

    with servicer.intercept() as ctx:
        ctx.add_response(
            "SandboxGetExitSnapshotV2",
            api_pb2.SandboxGetExitSnapshotResponse(
                success=api_pb2.SandboxGetExitSnapshotResponse.Success(image_id="im-exit-snapshot-123")
            ),
        )
        if terminated_by == "this_handle":
            sb.terminate()
        else:
            created.terminate()
        image = sb._experimental_get_exit_snapshot(timeout=0)

    assert image.object_id == "im-exit-snapshot-123"
    (req,) = ctx.get_requests("SandboxGetExitSnapshotV2")
    assert req.sandbox_id == "sb-v2-123"
    # Terminating reports the last session, so only the other handle's termination needs a lookup.
    assert len(ctx.get_requests("SxSandboxGet")) == (0 if terminated_by == "this_handle" else 1)
    assert ctx.get_requests("SxSandboxResume") == []


def test_persistent_sandbox_from_id_logs_do_not_resume(app, client, servicer):
    sb = _create(app, client)
    sb._experimental_pause()

    sb2 = Sandbox.from_id(sb.object_id, client=client)
    with servicer.intercept() as ctx:

        async def fetch_logs(servicer, stream):
            await stream.recv_message()
            await stream.send_message(api_pb2.AppFetchLogsResponse())

        ctx.set_responder("AppFetchLogs", fetch_logs)
        assert list(sb2.logs.tail()) == []
    # Logs are read from the paused session.
    assert {req.task_id for req in ctx.get_requests("AppFetchLogs")} == {"ta-v2-123"}
    assert ctx.get_requests("SxSandboxResume") == []


@pytest.mark.parametrize("detached", [False, True])
def test_persistent_sandbox_logs_use_last_session(app, client, servicer, detached):
    sb = _create(app, client)
    sb._experimental_pause()
    if detached:
        sb.detach()

    with servicer.intercept() as ctx:

        async def fetch_logs(servicer, stream):
            await stream.recv_message()
            await stream.send_message(api_pb2.AppFetchLogsResponse())

        ctx.set_responder("AppFetchLogs", fetch_logs)
        assert list(sb.logs.tail()) == []
    assert all(request.task_id for request in ctx.get_requests("AppFetchLogs"))
    assert ctx.get_requests("SxSandboxResume") == []


def test_persistent_sandbox_create_connect_token(app, client, servicer):
    sb = _create(app, client)
    sb._experimental_pause()

    with servicer.intercept() as ctx:
        creds = sb.create_connect_token(user_metadata={"user": "alice"}, port=3000)
    assert creds.url == f"https://a-{sb.object_id}-3000.tunnel.modal.host"
    assert creds.token == f'token-{sb.object_id}-{{"user": "alice"}}'
    (req,) = ctx.get_requests("SxSandboxCreateConnectToken")
    assert req.sx_id == sb.object_id
    assert req.port == 3000
    # The token is addressed by the persistent Sandbox, so creating it doesn't resume a paused one.
    assert ctx.get_requests("SxSandboxResume") == []
    assert ctx.get_requests("SandboxCreateConnectTokenV2") == []
