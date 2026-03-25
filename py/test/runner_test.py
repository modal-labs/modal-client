# Copyright Modal Labs 2023
import contextlib
import pytest
import time
import typing
from unittest import mock

from grpclib import GRPCError, Status

import modal
from modal._resolver import Resolver
from modal.client import Client
from modal.exception import AuthError
from modal.runner import deploy_app, run_app
from modal_proto import api_pb2

T = typing.TypeVar("T")


def test_run_app(servicer, client):
    dummy_app = modal.App()
    with servicer.intercept() as ctx:
        with run_app(dummy_app, client=client):
            pass

    ctx.pop_request("AppCreate")
    ctx.pop_request("AppPublish")
    ctx.pop_request("AppClientDisconnect")


def test_run_app_shutdown_cleanliness(servicer, client, caplog):
    dummy_app = modal.App()

    heartbeat_interval_secs = 1.0

    # Introduce jittery response delay to catch race conditions between
    # concurrently executing RPCs.
    servicer.set_resp_jitter(heartbeat_interval_secs)

    with mock.patch("modal.runner.HEARTBEAT_INTERVAL", heartbeat_interval_secs):
        with modal.enable_output(), run_app(dummy_app, client=client):
            time.sleep(heartbeat_interval_secs)

    # Verify no ERROR logs were emitted, during shutdown or otherwise.
    error_logs = [record for record in caplog.records if record.levelname == "ERROR"]
    assert len(error_logs) == 0, f"Found unexpected error logs: {error_logs}"


def test_run_app_unauthenticated(servicer):
    dummy_app = modal.App()
    with Client.anonymous(servicer.client_addr) as client:
        with pytest.raises(AuthError):
            with run_app(dummy_app, client=client):
                pass


def dummy(): ...


def test_run_app_profile_env_with_refs(servicer, client, monkeypatch):
    monkeypatch.setenv("MODAL_ENVIRONMENT", "profile_env")
    with servicer.intercept() as ctx:
        dummy_app = modal.App(include_source=False)
        ref = modal.Secret.from_name("some_secret")
        dummy_app.function(secrets=[ref])(dummy)

    assert ctx.calls == []  # all calls should be deferred

    with servicer.intercept() as ctx:
        ctx.add_response("SecretGetOrCreate", api_pb2.SecretGetOrCreateResponse(secret_id="st-123"))
        with run_app(dummy_app, client=client):
            pass

    with pytest.raises(Exception):
        ctx.pop_request("SecretCreate")  # should not create a new secret...

    app_create = ctx.pop_request("AppCreate")
    assert app_create.environment_name == "profile_env"

    secret_get_or_create = ctx.pop_request("SecretGetOrCreate")
    assert secret_get_or_create.environment_name == "profile_env"


def test_run_app_custom_env_with_refs(servicer, client, monkeypatch):
    monkeypatch.setenv("MODAL_ENVIRONMENT", "profile_env")
    dummy_app = modal.App(include_source=False)
    own_env_secret = modal.Secret.from_name("own_env_secret")
    other_env_secret = modal.Secret.from_name("other_env_secret", environment_name="third")  # explicit lookup

    dummy_app.function(secrets=[own_env_secret, other_env_secret])(dummy)

    with servicer.intercept() as ctx:
        ctx.add_response("SecretGetOrCreate", api_pb2.SecretGetOrCreateResponse(secret_id="st-123"))
        ctx.add_response("SecretGetOrCreate", api_pb2.SecretGetOrCreateResponse(secret_id="st-456"))
        with run_app(dummy_app, client=client, environment_name="custom"):
            pass

    with pytest.raises(Exception):
        ctx.pop_request("SecretCreate")

    app_create = ctx.pop_request("AppCreate")
    assert app_create.environment_name == "custom"

    secret_get_or_create = ctx.pop_request("SecretGetOrCreate")
    assert secret_get_or_create.environment_name == "custom"

    secret_get_or_create_2 = ctx.pop_request("SecretGetOrCreate")
    assert secret_get_or_create_2.environment_name == "third"


def test_deploy_without_rich(servicer, client, no_rich):
    app = modal.App("dummy-app", include_source=False)
    app.function()(dummy)
    deploy_app(app, client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize("build_validation", ["error", "warn", "ignore"])
async def test_mid_build_modifications(servicer, client, tmp_path, monkeypatch, build_validation):
    monkeypatch.setenv("MODAL_BUILD_VALIDATION", build_validation)

    # Patch resolver to give it a build start time that is old, which triggers validation.
    class PatchedResolver(Resolver):
        @property
        def build_start(self) -> float:
            return 0.0

    monkeypatch.setattr("modal.runner.Resolver", PatchedResolver)

    (large_dir := tmp_path / "large_files").mkdir()
    (large_dir / "1.txt").write_bytes("large 1".encode())

    image = modal.Image.debian_slim().add_local_dir(large_dir, "/root/large_files")

    app = modal.App(image=image, include_source=False)
    app.function()(dummy)

    handler_assertion: contextlib.AbstractContextManager
    if build_validation == "error":
        handler_assertion = pytest.raises(modal.exception.ExecutionError, match="modified during build")
    elif build_validation == "warn":
        handler_assertion = pytest.warns(UserWarning, match="modified during build")
    else:
        handler_assertion = contextlib.nullcontext()

    with handler_assertion:
        async with app.run.aio(client=client):
            pass


def test_run_app_interactive_no_spinner(servicer, client):
    """Don't show status spinner in interactive mode to avoid interfering with breakpoints."""
    app = modal.App()

    with mock.patch("modal._output.rich.RichOutputManager.show_status_spinner") as mock_spinner:
        with modal.enable_output():
            with run_app(app, client=client, interactive=True):
                pass
        mock_spinner.return_value.__enter__.assert_not_called()

    with mock.patch("modal._output.rich.RichOutputManager.show_status_spinner") as mock_spinner:
        with modal.enable_output():
            with run_app(app, client=client, interactive=False):
                pass
        mock_spinner.return_value.__enter__.assert_called_once()


@pytest.mark.parametrize("mode", ["deploy", "run"])
def test_run_app_recreate_deployment(servicer, client, monkeypatch, mode):
    monkeypatch.setattr(modal.runner, "WAIT_FOR_CONTAINER_STOP_SLEEP_INTERVAL", 0.01)
    task_list_calls = 0

    async def task_list(servicer, stream):
        nonlocal task_list_calls
        await stream.recv_message()

        if task_list_calls <= 1:
            resp = api_pb2.TaskListResponse(
                tasks=[
                    api_pb2.TaskStats(task_id="ta-123", enqueued_at=123),
                    api_pb2.TaskStats(task_id="ta-321", enqueued_at=321),
                ]
            )
        elif task_list_calls == 2:
            resp = api_pb2.TaskListResponse(tasks=[api_pb2.TaskStats(task_id="ta-123", enqueued_at=123)])
        else:
            resp = api_pb2.TaskListResponse(tasks=[])

        task_list_calls += 1
        await stream.send_message(resp)

    dummy_app = modal.App("my-app")
    with servicer.intercept() as ctx:
        ctx.set_responder("TaskList", task_list)
        if mode == "deploy":
            dummy_app.deploy(client=client, strategy="recreate")
        else:
            # Test the `modal serve` case since it uses run_app
            with run_app(dummy_app, client=client, deployment_strategy="recreate"):
                pass

    assert task_list_calls == 4

    task_ids = set(servicer.container_stop_ids)
    assert task_ids == set(["ta-123", "ta-321"])


def test_run_app_recreate_deployment_no_op_deployment(servicer, client, monkeypatch):
    monkeypatch.setattr(modal.runner, "WAIT_FOR_CONTAINER_STOP_SLEEP_INTERVAL", 0.01)

    dummy_app = modal.App("my-app")
    servicer.app_publish_is_noop = True
    with servicer.intercept():
        dummy_app.deploy(client=client, strategy="recreate")

    assert servicer.task_list_calls == 0


def test_run_app_recreate_deployment_timeout(servicer, client, monkeypatch):
    monkeypatch.setattr(modal.runner, "WAIT_FOR_CONTAINER_STOP_TIMEOUT", 0.1)

    async def task_list(servicer, stream):
        await stream.recv_message()
        resp = api_pb2.TaskListResponse(
            tasks=[
                api_pb2.TaskStats(task_id="ta-123", enqueued_at=123),
                api_pb2.TaskStats(task_id="ta-321", enqueued_at=321),
            ]
        )
        await stream.send_message(resp)

    dummy_app = modal.App("my-app")

    msg = "App updated successfully, but containers did not all terminate."
    with pytest.warns(UserWarning, match=msg):
        with servicer.intercept() as ctx:
            ctx.set_responder("TaskList", task_list)
            dummy_app.deploy(client=client, strategy="recreate")

    task_ids = set(servicer.container_stop_ids)
    assert task_ids == set(["ta-123", "ta-321"])


def test_run_app_recreate_deployment_stop_fails(servicer, client, monkeypatch):
    monkeypatch.setattr(modal.runner, "WAIT_FOR_CONTAINER_STOP_TIMEOUT", 1.0)

    async def task_list(servicer, stream):
        await stream.recv_message()
        resp = api_pb2.TaskListResponse(
            tasks=[
                api_pb2.TaskStats(task_id="ta-123", enqueued_at=123),
                api_pb2.TaskStats(task_id="ta-321", enqueued_at=321),
            ]
        )
        await stream.send_message(resp)

    async def container_stop(servicer, stream):
        await stream.recv_message()
        raise GRPCError(Status.NOT_FOUND, "Unable to stop containers")

    dummy_app = modal.App("my-app")

    msg = "App updated successfully, but containers did not all terminate."
    with pytest.warns(UserWarning, match=msg):
        with servicer.intercept() as ctx:
            ctx.set_responder("TaskList", task_list)
            ctx.set_responder("ContainerStop", container_stop)
            dummy_app.deploy(client=client, strategy="recreate")
