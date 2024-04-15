# Copyright Modal Labs 2022
import pytest

from google.protobuf.empty_pb2 import Empty

from modal import Stub, interact
from modal._container_io_manager import ContainerIOManager
from modal.app import _init_container_app
from modal_proto import api_pb2

from .supports.skip import skip_windows_unix_socket


def my_f_1(x):
    pass


@skip_windows_unix_socket
@pytest.mark.asyncio
async def test_container_function_lazily_imported(container_client):
    items = [
        api_pb2.AppGetObjectsItem(
            tag="my_f_1",
            object=api_pb2.Object(
                object_id="fu-123",
                function_handle_metadata=api_pb2.FunctionHandleMetadata(),
            ),
        ),
        api_pb2.AppGetObjectsItem(
            tag="my_d",
            object=api_pb2.Object(object_id="di-123"),
        ),
    ]
    container_app = _init_container_app(items, "ap-123")
    stub = Stub()

    # This is normally done in _container_entrypoint
    stub._init_container(container_client, container_app)

    # Now, let's create my_f after the app started running and make sure it works
    my_f_container = stub.function()(my_f_1)
    assert await my_f_container.remote.aio(42) == 1764  # type: ignore


@skip_windows_unix_socket
def test_interact(container_client, unix_servicer):
    # Initialize container singleton
    ContainerIOManager(api_pb2.ContainerArguments(), container_client)

    with unix_servicer.intercept() as ctx:
        ctx.add_response("FunctionStartPtyShell", Empty())
        interact()
