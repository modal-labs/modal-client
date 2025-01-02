# Copyright Modal Labs 2022
import json
import os
import pytest
import time
from contextlib import contextmanager
from unittest import mock

from google.protobuf.empty_pb2 import Empty
from google.protobuf.message import Message

from modal import App, interact
from modal._runtime.container_io_manager import ContainerIOManager
from modal._utils.async_utils import synchronize_api
from modal._utils.grpc_utils import retry_transient_errors
from modal.exception import InvalidError
from modal.running_app import RunningApp
from modal_proto import api_pb2


def my_f_1(x):
    pass


def temp_restore_path(tmpdir):
    # Write out a restore file so that snapshot+restore will complete
    restore_path = tmpdir.join("fake-restore-state.json")
    restore_path.write_text(
        json.dumps(
            {
                "task_id": "ta-i-am-restored",
                "task_secret": "ts-i-am-restored",
                "function_id": "fu-i-am-restored",
            }
        ),
        encoding="utf-8",
    )
    return restore_path


@pytest.mark.asyncio
async def test_container_function_lazily_imported(container_client):
    function_ids: dict[str, str] = {
        "my_f_1": "fu-123",
    }
    object_handle_metadata: dict[str, Message] = {
        "fu-123": api_pb2.FunctionHandleMetadata(),
    }
    container_app = RunningApp("ap-123", function_ids=function_ids, object_handle_metadata=object_handle_metadata)
    app = App()

    # This is normally done in _container_entrypoint
    app._init_container(container_client, container_app)

    # Now, let's create my_f after the app started running and make sure it works
    my_f_container = app.function()(my_f_1)
    assert await my_f_container.remote.aio(42) == 1764  # type: ignore


def square(x):
    pass


@synchronize_api
async def stop_app(client, app_id):
    # helper to ensur we run the rpc from the synchronicity loop - otherwise we can run into weird deadlocks
    return await retry_transient_errors(client.stub.AppStop, api_pb2.AppStopRequest(app_id=app_id))


@contextmanager
def set_env_vars(restore_path, container_addr):
    with mock.patch.dict(
        os.environ,
        {
            "MODAL_RESTORE_STATE_PATH": str(restore_path),
            "MODAL_SERVER_URL": container_addr,
            "MODAL_TASK_ID": "ta-123",
            "MODAL_IS_REMOTE": "1",
        },
    ):
        yield


@pytest.mark.asyncio
async def test_container_snapshot_reference_capture(container_client, tmpdir, servicer, client):
    app = App()
    from modal import Function
    from modal.runner import deploy_app

    app.function()(square)
    app_name = "my-app"
    app_id = deploy_app(app, app_name, client=container_client).app_id
    f = Function.lookup(app_name, "square", client=container_client)
    assert f.object_id == "fu-1"
    await f.remote.aio()
    assert f.object_id == "fu-1"
    io_manager = ContainerIOManager(api_pb2.ContainerArguments(checkpoint_id="ch-123"), container_client)
    restore_path = temp_restore_path(tmpdir)
    with set_env_vars(restore_path, servicer.container_addr):
        io_manager.memory_snapshot()

    # Stop the App, invalidating the fu- ID stored in `f`.
    stop_app(client, app_id)
    # After snapshot-restore the previously looked-up Function should get refreshed and have the
    # new fu- ID. ie. the ID should not be stale and invalid.
    new_app_id = deploy_app(app, app_name, client=client).app_id
    assert new_app_id != app_id
    await f.remote.aio()
    assert f.object_id == "fu-2"
    # Purposefully break FunctionGet to check the hydration is cached.
    del servicer.app_objects[new_app_id]
    await f.remote.aio()  # remote call succeeds because it didn't re-hydrate Function
    assert f.object_id == "fu-2"


def test_container_snapshot_restore_heartbeats(tmpdir, servicer, container_client):
    io_manager = ContainerIOManager(api_pb2.ContainerArguments(checkpoint_id="ch-123"), container_client)
    restore_path = temp_restore_path(tmpdir)

    # Ensure that heartbeats only run after the snapshot
    heartbeat_interval_secs = 0.01
    with io_manager.heartbeats(True):
        with set_env_vars(restore_path, servicer.container_addr):
            with mock.patch("modal.runner.HEARTBEAT_INTERVAL", heartbeat_interval_secs):
                time.sleep(heartbeat_interval_secs * 2)
                assert not list(
                    filter(lambda req: isinstance(req, api_pb2.ContainerHeartbeatRequest), servicer.requests)
                )
                io_manager.memory_snapshot()
                time.sleep(heartbeat_interval_secs * 2)
                assert list(filter(lambda req: isinstance(req, api_pb2.ContainerHeartbeatRequest), servicer.requests))


@pytest.mark.asyncio
async def test_container_debug_snapshot(container_client, tmpdir, servicer):
    # Get an IO manager, where restore takes place
    io_manager = ContainerIOManager(api_pb2.ContainerArguments(checkpoint_id="ch-123"), container_client)
    restore_path = tmpdir.join("fake-restore-state.json")
    # Write the restore file to start a debugger
    restore_path.write_text(
        json.dumps({"snapshot_debug": "1"}),
        encoding="utf-8",
    )

    # Test that the breakpoint was called
    test_breakpoint = mock.Mock()
    with mock.patch("sys.breakpointhook", test_breakpoint):
        with set_env_vars(restore_path, servicer.container_addr):
            io_manager.memory_snapshot()
            test_breakpoint.assert_called_once()


@pytest.mark.asyncio
async def test_rpc_wrapping_restores(container_client, servicer, tmpdir):
    import modal

    io_manager = ContainerIOManager(api_pb2.ContainerArguments(checkpoint_id="ch-123"), container_client)
    restore_path = temp_restore_path(tmpdir)

    d = modal.Dict.lookup("my-amazing-dict", {"xyz": 123}, create_if_missing=True, client=container_client)
    d["abc"] = 42

    with set_env_vars(restore_path, servicer.container_addr):
        io_manager.memory_snapshot()

    # TODO(Jonathon): These RPC wrappers are tested directly because I could not
    # find a way to construct in this test a UnaryStreamWrapper with a stale snapshotted client.
    @synchronize_api
    async def exercise_rpcs():
        n = 0
        # Test UnaryStreamWrapper
        async for _ in container_client.stub.DictContents.unary_stream(
            api_pb2.DictContentsRequest(dict_id=d.object_id, keys=True)
        ):
            n += 1
        assert n == 2
        # Test UnaryUnaryWrapper
        await container_client.stub.DictClear(api_pb2.DictClearRequest(dict_id=d.object_id))

    await exercise_rpcs.aio()


def test_interact(container_client, servicer):
    # Initialize container singleton
    function_def = api_pb2.Function(pty_info=api_pb2.PTYInfo(pty_type=api_pb2.PTYInfo.PTY_TYPE_SHELL))
    ContainerIOManager(api_pb2.ContainerArguments(function_def=function_def), container_client)
    with servicer.intercept() as ctx:
        ctx.add_response("FunctionStartPtyShell", Empty())
        interact()


def test_interact_no_pty_error(container_client, servicer):
    # Initialize container singleton
    ContainerIOManager(api_pb2.ContainerArguments(), container_client)
    with pytest.raises(InvalidError, match=r"modal.interact\(\) without running Modal in interactive mode"):
        interact()
