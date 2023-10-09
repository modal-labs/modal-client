# Copyright Modal Labs 2022
import importlib
import os
import pytest
from unittest import mock

import modal.secret
from modal import Dict, Image, Stub
from modal.app import ContainerApp
from modal.exception import InvalidError
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

    await ContainerApp.init_container.aio(container_client, "ap-123")
    stub = Stub()

    # Now, let's create my_f after the app started running and make sure it works
    my_f_container = stub.function()(my_f_1)
    assert await my_f_container.remote.aio(42) == 1764  # type: ignore

    # Also make sure dicts work
    my_d_container = Dict.new()
    stub.my_d = my_d_container  # should trigger id assignment
    assert my_d_container.object_id == "di-123"


@skip_windows_unix_socket
@pytest.mark.asyncio
async def test_is_inside(servicer, unix_servicer, client, container_client):
    def get_stub():
        image_1 = Image.debian_slim().pip_install(["abc"])
        image_2 = Image.debian_slim().pip_install(["def"])
        return Stub(image=image_1, image_2=image_2)

    stub = get_stub()

    # Run container
    async with stub.run(client=client):
        # We're not inside the container (yet)
        assert not stub.is_inside()
        assert not stub.is_inside(stub.image)
        assert not stub.is_inside(stub.image_2)

        app_id = stub.app_id
        image_1_id = stub["image"].object_id
        image_2_id = stub["image_2"].object_id

        # Copy the app objects to the container servicer
        unix_servicer.app_objects[app_id] = servicer.app_objects[app_id]

        # Pretend that we're inside the container
        await ContainerApp.init_container.aio(container_client, app_id)

        # Create a new stub (TODO: tie it to the previous stub through name or similar)
        stub = get_stub()

        # Pretend that we're inside image 1
        with mock.patch.dict(os.environ, {"MODAL_IMAGE_ID": image_1_id}):
            assert stub.is_inside()
            assert stub.is_inside(stub.image)
            assert not stub.is_inside(stub.image_2)

        # Pretend that we're inside image 2
        with mock.patch.dict(os.environ, {"MODAL_IMAGE_ID": image_2_id}):
            assert stub.is_inside()
            assert not stub.is_inside(stub.image)
            assert stub.is_inside(stub.image_2)


def f():
    pass


@skip_windows_unix_socket
@pytest.mark.asyncio
async def test_is_inside_default_image(servicer, unix_servicer, client, container_client):
    stub = Stub()
    stub.function()(f)

    assert not stub.is_inside()

    from modal.stub import _default_image

    async with stub.run(client=client):
        app_id = stub.app_id
        default_image_id = _default_image.object_id

    # Copy the app objects to the container servicer
    unix_servicer.app_objects = servicer.app_objects
    unix_servicer.app_functions = servicer.app_functions

    await ContainerApp.init_container.aio(container_client, app_id)

    # Create a new stub (TODO: tie it to the previous stub through name or similar)
    stub = Stub()

    with mock.patch.dict(os.environ, {"MODAL_IMAGE_ID": default_image_id}):
        assert stub.is_inside()


@pytest.mark.skip("runtime type checking has been temporarily disabled")
def test_typechecking_not_enforced_in_container():
    def incorrect_usage():
        class InvalidType:
            pass

        modal.secret.Secret(env_dict={"foo": InvalidType()})  # type: ignore

    with pytest.raises(InvalidError):
        incorrect_usage()  # should throw when running locally

    with mock.patch.dict(os.environ, {"MODAL_IMAGE_ID": "im-123"}):
        importlib.reload(modal.secret)
        incorrect_usage()  # should not throw in container, since typechecks add a lot of overhead on import
