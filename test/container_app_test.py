# Copyright Modal Labs 2022
import json
import os
import pytest
from typing import Dict
from unittest import mock

from google.protobuf.empty_pb2 import Empty
from google.protobuf.message import Message

from modal import App, interact
from modal._container_io_manager import ContainerIOManager
from modal.running_app import RunningApp
from modal_proto import api_pb2

from .supports.skip import skip_windows_unix_socket


def my_f_1(x):
    pass


@skip_windows_unix_socket
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


@skip_windows_unix_socket
@pytest.mark.asyncio
async def test_container_snapshot_restore(container_client, tmpdir, servicer):
    # Get a reference to a Client instance in memory
    old_client = container_client
    io_manager = ContainerIOManager(api_pb2.ContainerArguments(), container_client)
    restore_path = tmpdir.join("fake-restore-state.json")
    # Write out a restore file so that snapshot+restore will complete
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
    with mock.patch.dict(
        os.environ, {"MODAL_RESTORE_STATE_PATH": str(restore_path), "MODAL_SERVER_URL": servicer.remote_addr}
    ):
        io_manager.memory_snapshot()
        # In-memory Client instance should have update credentials, not old credentials
        assert old_client.credentials == ("ta-i-am-restored", "ts-i-am-restored")


@skip_windows_unix_socket
def test_interact(container_client, unix_servicer):
    # Initialize container singleton
    ContainerIOManager(api_pb2.ContainerArguments(), container_client)
    with unix_servicer.intercept() as ctx:
        ctx.add_response("FunctionStartPtyShell", Empty())
        interact()
