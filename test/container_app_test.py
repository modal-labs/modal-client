# Copyright Modal Labs 2022
import pytest

from modal import Stub
from modal.app import _container_app, init_container_app
from modal_proto import api_pb2

from .supports.skip import skip_windows_unix_socket


def my_f_1(x):
    pass


@skip_windows_unix_socket
@pytest.mark.asyncio
async def test_container_function_lazily_imported(unix_servicer, container_client):
    unix_servicer.app_objects["ap-123"] = {
        "my_f_1": "fu-123",
        "my_d": "di-123",
    }
    unix_servicer.app_functions["fu-123"] = api_pb2.Function()

    await init_container_app.aio(container_client, "ap-123")
    stub = Stub()

    # This is normally done in _container_entrypoint
    stub._init_container(container_client, _container_app)

    # Now, let's create my_f after the app started running and make sure it works
    my_f_container = stub.function()(my_f_1)
    assert await my_f_container.remote.aio(42) == 1764  # type: ignore
