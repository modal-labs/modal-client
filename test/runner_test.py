# Copyright Modal Labs 2023
import asyncio
import contextlib
import pytest
import time
import typing
from unittest import mock

import modal
from modal.client import Client
from modal.exception import AuthError, DeprecationError
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
        dummy_app = modal.App()
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
    dummy_app = modal.App()
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
    app = modal.App("dummy-app")
    app.function()(dummy)
    deploy_app(app, client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize("build_validation", ["error", "warn", "ignore"])
async def test_mid_build_modifications(servicer, client, tmp_path, monkeypatch, build_validation):
    monkeypatch.setenv("MODAL_BUILD_VALIDATION", build_validation)

    (large_dir := tmp_path / "large_files").mkdir()
    for i in range(512 + 1):  # Equivalent to file upload concurrency
        (large_dir / f"{i:02d}.txt").write_bytes(f"large {i:02d}".encode())

    image = modal.Image.debian_slim().add_local_dir(large_dir, "/root/large_files")

    app = modal.App(image=image, include_source=False)
    app.function()(dummy)

    async def change_file_after_delay():
        await asyncio.sleep(0.2)  # "Uploading" large should take 2 seconds; see mock MountPutFile
        for f in large_dir.iterdir():
            f.touch()

    handler_assertion: contextlib.AbstractContextManager
    if build_validation == "error":
        handler_assertion = pytest.raises(modal.exception.ExecutionError, match="modified during build")
    elif build_validation == "warn":
        handler_assertion = pytest.warns(UserWarning, match="modified during build")
    else:
        handler_assertion = contextlib.nullcontext()

    at = asyncio.create_task(change_file_after_delay())
    try:
        with handler_assertion:
            async with app.run.aio(client=client):
                ...
    finally:
        await at


def test_deploy_app_namespace_deprecated(servicer, client):
    # Test deploy_app with namespace parameter warns
    app = modal.App("test-app")

    with pytest.warns(
        DeprecationError,
        match="The `namespace` parameter for `modal.runner.deploy_app` is deprecated",
    ):
        deploy_app(app, name="test-deploy", namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE, client=client)

    # Test that deploy_app without namespace parameter doesn't warn about namespace
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        deploy_app(app, name="test-deploy", client=client)
    # Filter out any unrelated warnings
    namespace_warnings = [w for w in record if "namespace" in str(w.message).lower()]
    assert len(namespace_warnings) == 0


def test_run_app_interactive_no_spinner(servicer, client):
    """Don't show status spinner in interactive mode to avoid interfering with breakpoints."""
    app = modal.App()

    with mock.patch("modal._output.OutputManager.show_status_spinner") as mock_spinner:
        with modal.enable_output():
            with run_app(app, client=client, interactive=True):
                pass
        mock_spinner.return_value.__enter__.assert_not_called()

    with mock.patch("modal._output.OutputManager.show_status_spinner") as mock_spinner:
        with modal.enable_output():
            with run_app(app, client=client, interactive=False):
                pass
        mock_spinner.return_value.__enter__.assert_called_once()
