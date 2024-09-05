# Copyright Modal Labs 2022
import asyncio
import json
import os
import pytest
from typing import Dict
from unittest import mock

from google.protobuf.empty_pb2 import Empty
from google.protobuf.message import Message

from modal import App, interact
from modal._container_io_manager import ContainerIOManager, _ContainerIOManager
from modal._utils.grpc_utils import create_channel, retry_transient_errors
from modal.client import _Client
from modal.exception import InvalidError
from modal.running_app import RunningApp
from modal_proto import api_grpc, api_pb2, modal_api_grpc


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
    tag_to_object_id: Dict[str, str] = {
        "my_f_1": "fu-123",
        "my_d": "di-123",
    }
    object_handle_metadata: Dict[str, Message] = {
        "fu-123": api_pb2.FunctionHandleMetadata(),
    }
    container_app = RunningApp(
        app_id="ap-123", tag_to_object_id=tag_to_object_id, object_handle_metadata=object_handle_metadata
    )
    app = App()

    # This is normally done in _container_entrypoint
    app._init_container(container_client, container_app)

    # Now, let's create my_f after the app started running and make sure it works
    my_f_container = app.function()(my_f_1)
    assert await my_f_container.remote.aio(42) == 1764  # type: ignore


@pytest.mark.asyncio
async def test_container_snapshot_restore(container_client, tmpdir, servicer):
    # Get a reference to a Client instance in memory
    old_client = container_client
    io_manager = ContainerIOManager(api_pb2.ContainerArguments(), container_client)
    restore_path = temp_restore_path(tmpdir)
    with mock.patch.dict(
        os.environ, {"MODAL_RESTORE_STATE_PATH": str(restore_path), "MODAL_SERVER_URL": servicer.container_addr}
    ):
        io_manager.memory_snapshot()
        # In-memory Client instance should have updated credentials, not old credentials
        assert old_client.credentials == ("ta-i-am-restored", "ts-i-am-restored")


def square(x):
    pass


@pytest.mark.asyncio
async def test_container_snapshot_reference_capture(container_client, tmpdir, servicer):
    app = App()
    from modal import Function
    from modal.runner import deploy_app

    channel = create_channel(servicer.client_addr)
    client_stub = modal_api_grpc.ModalClientModal(api_grpc.ModalClientStub(channel))
    app.function()(square)
    app_name = "my-app"
    app_id = deploy_app(app, app_name, client=container_client).app_id

    f = Function.lookup(app_name, "square", client=container_client)
    assert f.object_id == "fu-1"
    await f.remote.aio()
    assert f.object_id == "fu-1"

    io_manager = ContainerIOManager(api_pb2.ContainerArguments(), container_client)
    restore_path = temp_restore_path(tmpdir)
    with mock.patch.dict(
        os.environ, {"MODAL_RESTORE_STATE_PATH": str(restore_path), "MODAL_SERVER_URL": servicer.container_addr}
    ):
        io_manager.memory_snapshot()

    # Stop the App, invalidating the fu- ID stored in `f`.
    assert await retry_transient_errors(client_stub.AppStop, api_pb2.AppStopRequest(app_id=app_id))
    # After snapshot-restore the previously looked-up Function should get refreshed and have the
    # new fu- ID. ie. the ID should not be stale and invalid.
    new_app_id = deploy_app(app, app_name, client=container_client).app_id
    assert new_app_id != app_id
    await f.remote.aio()
    assert f.object_id == "fu-2"
    # Purposefully break FunctionGet to check the hydration is cached.
    del servicer.app_objects[new_app_id]
    await f.remote.aio()  # remote call succeeds because it didn't re-hydrate Function
    assert f.object_id == "fu-2"
    channel.close()


@pytest.mark.asyncio
async def test_container_snapshot_restore_heartbeats(tmpdir, servicer):
    client = _Client(servicer.container_addr, api_pb2.CLIENT_TYPE_CONTAINER, ("ta-123", "task-secret"))
    async with client as async_client:
        io_manager = _ContainerIOManager(api_pb2.ContainerArguments(), async_client)
        restore_path = temp_restore_path(tmpdir)

        # Ensure that heartbeats only run after the snapshot
        heartbeat_interval_secs = 0.01
        async with io_manager.heartbeats(True):
            with mock.patch.dict(
                os.environ,
                {"MODAL_RESTORE_STATE_PATH": str(restore_path), "MODAL_SERVER_URL": servicer.container_addr},
            ):
                with mock.patch("modal.runner.HEARTBEAT_INTERVAL", heartbeat_interval_secs):
                    await asyncio.sleep(heartbeat_interval_secs * 2)
                    assert not list(
                        filter(lambda req: isinstance(req, api_pb2.ContainerHeartbeatRequest), servicer.requests)
                    )
                    await io_manager.memory_snapshot()
                    await asyncio.sleep(heartbeat_interval_secs * 2)
                    assert list(
                        filter(lambda req: isinstance(req, api_pb2.ContainerHeartbeatRequest), servicer.requests)
                    )


@pytest.mark.asyncio
async def test_container_debug_snapshot(container_client, tmpdir, servicer):
    # Get an IO manager, where restore takes place
    io_manager = ContainerIOManager(api_pb2.ContainerArguments(), container_client)
    restore_path = tmpdir.join("fake-restore-state.json")
    # Write the restore file to start a debugger
    restore_path.write_text(
        json.dumps({"snapshot_debug": "1"}),
        encoding="utf-8",
    )

    # Test that the breakpoint was called
    test_breakpoint = mock.Mock()
    with mock.patch("sys.breakpointhook", test_breakpoint):
        with mock.patch.dict(
            os.environ, {"MODAL_RESTORE_STATE_PATH": str(restore_path), "MODAL_SERVER_URL": servicer.container_addr}
        ):
            io_manager.memory_snapshot()
            test_breakpoint.assert_called_once()


@pytest.fixture(scope="function")
def fake_torch_module():
    module_path = os.path.join(os.getcwd(), "torch.py")
    with open(module_path, "w") as f:
        f.write(
            """
import dataclasses
@dataclasses.dataclass
class CUDA:
    device_count = lambda self: 0
    _device_count_nvml = lambda self: 2

cuda = CUDA()
"""
        )

    yield module_path
    # Teardown: remove the torch.py file
    os.remove(module_path)


@pytest.fixture(scope="function")
def weird_torch_module():
    module_path = os.path.join(os.getcwd(), "torch.py")
    with open(module_path, "w") as f:
        f.write("IM_WEIRD = 42\n")

    yield module_path

    os.remove(module_path)  # Teardown: remove the torch.py file


@pytest.mark.asyncio
async def test_container_snapshot_patching(fake_torch_module, container_client, tmpdir, servicer):
    io_manager = ContainerIOManager(api_pb2.ContainerArguments(), container_client)

    # bring fake torch into scope and call the utility fn
    import torch

    assert torch.cuda.device_count() == 0

    # Write out a restore file so that snapshot+restore will complete
    restore_path = temp_restore_path(tmpdir)
    with mock.patch.dict(
        os.environ, {"MODAL_RESTORE_STATE_PATH": str(restore_path), "MODAL_SERVER_URL": servicer.container_addr}
    ):
        io_manager.memory_snapshot()
        import torch

        assert torch.cuda.device_count() == 2


@pytest.mark.asyncio
async def test_container_snapshot_patching_err(weird_torch_module, container_client, tmpdir, servicer):
    io_manager = ContainerIOManager(api_pb2.ContainerArguments(), container_client)
    restore_path = temp_restore_path(tmpdir)

    # bring weird torch into scope and call the utility fn
    import torch as trch

    assert trch.IM_WEIRD == 42

    with mock.patch.dict(
        os.environ, {"MODAL_RESTORE_STATE_PATH": str(restore_path), "MODAL_SERVER_URL": servicer.container_addr}
    ):
        io_manager.memory_snapshot()  # should not crash


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
