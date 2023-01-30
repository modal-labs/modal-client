# Copyright Modal Labs 2022
import pytest

from modal.aio import AioApp, AioFunctionHandle, AioStub, aio_container_app

from .supports.skip import skip_windows


def my_f_1(x):
    pass


def my_f_2(x):
    pass


@skip_windows
@pytest.mark.asyncio
async def test_container_function_initialization(unix_servicer, aio_container_client):
    stub = AioStub()
    # my_f_1_container = stub.function(my_f_1)

    unix_servicer.app_objects["ap-123"] = {
        "my_f_1": "fu-123",
        "my_f_2": "fu-456",
    }
    await AioApp.init_container(aio_container_client, "ap-123")

    # Make sure these functions exist and have the right type
    my_f_1_app = aio_container_app["my_f_1"]
    my_f_2_app = aio_container_app["my_f_1"]
    assert isinstance(my_f_1_app, AioFunctionHandle)
    assert isinstance(my_f_2_app, AioFunctionHandle)

    # Make sure we can call my_f_1 inside the container
    # assert await my_f_1_container.call(42) == 1764
    # TODO(erikbern): it's actually impossible for a stub function
    # to be created before the app inside a container, so let's
    # ignore this issue for now. It's just theoretical.

    # Now, let's create my_f_2 after the app started running
    # This might happen if some local module is imported lazily
    my_f_2_container = stub.function(my_f_2)
    assert await my_f_2_container.call(42) == 1764
