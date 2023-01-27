# Copyright Modal Labs 2022
import pytest

from modal.aio import AioApp, AioFunctionHandle, AioStub, aio_container_app

from .supports.skip import skip_windows


def my_f_1(x):
    # Body doesn't matter, the fixture overrides this anyway
    return x**3


@skip_windows
@pytest.mark.asyncio
async def test_container_function_initialization(unix_servicer, aio_container_client):
    unix_servicer.app_objects["ap-123"] = {
        "my_f_1": "fu-123",
        "my_f_2": "fu-456",
    }
    await AioApp._init_container(aio_container_client, "ap-123")

    # Make sure the app has a handle for this function
    f = aio_container_app["my_f_1"]
    assert isinstance(f, AioFunctionHandle)

    # Now, let's create a function with this name
    stub = AioStub()
    f = stub.function(my_f_1)
    assert isinstance(f, AioFunctionHandle)

    # We should be able to call this function inside the container
    assert await f.call(42) == 1764
