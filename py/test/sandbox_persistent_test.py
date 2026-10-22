# Copyright Modal Labs 2026
import pytest

from grpclib import GRPCError, Status

from modal import App, Image, Sandbox
from modal._utils.async_utils import synchronizer
from modal.exception import InvalidError
from modal.experimental import OutboundPolicy
from modal.sandbox._impl_persistent import (
    PERSISTENT_SANDBOX_SESSION_IDLE_TIMEOUT,
    PERSISTENT_SANDBOX_SESSION_TIMEOUT,
)
from modal_proto import api_pb2

from .supports.skip import skip_windows


@pytest.fixture
def app(client):
    return App.lookup("persistent-app", create_if_missing=True, client=client)


def _impl(sb):
    return synchronizer._translate_in(sb)._impl


def _create(app, client, **kwargs):
    kwargs.setdefault("timeout", None)
    return Sandbox.create(app=app, _experimental_persistent=True, client=client, **kwargs)


def test_persistent_sandbox_create(app, client, servicer):
    sb = _create(app, client)

    assert _impl(sb)._persistent_sandbox_id == "sp-1"
    [req] = servicer.persistent_sandbox_create_requests
    assert not req.definition.name
    assert req.definition.timeout_secs == PERSISTENT_SANDBOX_SESSION_TIMEOUT
    assert req.definition.idle_timeout_secs == PERSISTENT_SANDBOX_SESSION_IDLE_TIMEOUT
    assert not servicer.sandbox_create_v2_requests
    assert sb.poll() is None


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
def test_persistent_sandbox_resumes_on_failed_operation(app, client, servicer):
    sb = _create(app, client)
    sb._experimental_pause()

    with servicer.intercept() as ctx, servicer.task_command_router.intercept() as tcr_ctx:
        # The paused session no longer accepts commands, so the exec is retried on a resumed session.
        tcr_ctx.add_response("TaskExecStart", GRPCError(Status.NOT_FOUND, "sandbox has stopped"))
        p = sb.exec("echo", "hello")
        assert p.stdout.read() == "hello\n"
        assert p.wait() == 0

        assert len(ctx.get_requests("PersistentSandboxResume")) == 1
        assert len(tcr_ctx.get_requests("TaskExecStart")) == 2
    assert servicer.persistent_sandbox_resumes == 1


def test_persistent_sandbox_terminate(app, client, servicer):
    sb = _create(app, client)
    sb.terminate()

    assert servicer.persistent_sandbox_terminated == {"sp-1"}
    assert sb.poll() is not None
    with pytest.raises(InvalidError, match="has been terminated"):
        sb._experimental_pause()
    with pytest.raises(InvalidError, match="has been terminated"):
        sb._experimental_resume()


def test_persistent_sandbox_unsupported_operations(app, client):
    sb = _create(app, client)
    with pytest.raises(NotImplementedError):
        sb._experimental_update_outbound_policy(OutboundPolicy())
    with pytest.raises(NotImplementedError):
        sb.mount_image("/mnt", Image.from_id("im-123", client=client))
    with pytest.raises(NotImplementedError):
        sb.unmount_image("/mnt")
    with pytest.raises(NotImplementedError, match="Tags"):
        sb.get_tags()
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
