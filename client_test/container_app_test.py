# Copyright Modal Labs 2022
import importlib
import os
import pytest
from unittest import mock

import modal.secret
from modal import App, FunctionHandle, Image, Stub
from modal.exception import InvalidError

from .supports.skip import skip_windows_unix_socket


def my_f_1(x):
    pass


def my_f_2(x):
    pass


@skip_windows_unix_socket
@pytest.mark.asyncio
async def test_container_function_initialization(unix_servicer, container_client):
    unix_servicer.app_objects["ap-123"] = {
        "my_f_1": "fu-123",
        "my_f_2": "fu-456",
    }

    container_app = await App.init_container.aio(container_client, "ap-123")

    stub = Stub()
    # my_f_1_container = stub.function()(my_f_1)

    # Make sure these functions exist and have the right type
    my_f_1_app = container_app["my_f_1"]
    my_f_2_app = container_app["my_f_2"]
    assert isinstance(my_f_1_app, FunctionHandle)
    assert isinstance(my_f_2_app, FunctionHandle)

    # Make sure we can call my_f_1 inside the container
    # assert await my_f_1_container.call(42) == 1764
    # TODO(erikbern): it's actually impossible for a stub function
    # to be created before the app inside a container, so let's
    # ignore this issue for now. It's just theoretical.

    # Now, let's create my_f_2 after the app started running
    # This might happen if some local module is imported lazily
    my_f_2_container = stub.function()(my_f_2)
    assert await my_f_2_container.call.aio(42) == 1764  # type: ignore


@skip_windows_unix_socket
@pytest.mark.asyncio
async def test_is_inside(servicer, unix_servicer, client, container_client):
    image_1 = Image.debian_slim().pip_install(["abc"])
    image_2 = Image.debian_slim().pip_install(["def"])

    def get_stub():
        return Stub(image=image_1, image_2=image_2)

    stub = get_stub()

    # Run container
    async with stub.run(client=client) as app:
        # We're not inside the container (yet)
        assert not stub.is_inside()
        assert not stub.is_inside(image_1)
        assert not stub.is_inside(image_2)

        app_id = app.app_id
        image_1_id = app["image"].object_id
        image_2_id = app["image_2"].object_id

        # Copy the app objects to the container servicer
        unix_servicer.app_objects[app_id] = servicer.app_objects[app_id]

        # Pretend that we're inside the container
        await App.init_container.aio(container_client, app_id)

        # Create a new stub (TODO: tie it to the previous stub through name or similar)
        stub = get_stub()

        # Pretend that we're inside image 1
        with mock.patch.dict(os.environ, {"MODAL_IMAGE_ID": image_1_id}):
            assert stub.is_inside()
            assert stub.is_inside(image_1)
            assert not stub.is_inside(image_2)

        # Pretend that we're inside image 2
        with mock.patch.dict(os.environ, {"MODAL_IMAGE_ID": image_2_id}):
            assert stub.is_inside()
            assert not stub.is_inside(image_1)
            assert stub.is_inside(image_2)


def f():
    pass


@skip_windows_unix_socket
@pytest.mark.asyncio
async def test_is_inside_default_image(servicer, unix_servicer, client, container_client):
    stub = Stub()
    stub.function()(f)

    assert not stub.is_inside()

    from modal.stub import _default_image

    app = await App._init_new.aio(client)
    app_id = app.app_id
    default_image_handle = await app.create_one_object.aio(_default_image, "")
    default_image_id = default_image_handle.object_id

    # Copy the app objects to the container servicer
    unix_servicer.app_objects[app_id] = servicer.app_objects[app_id]

    await App.init_container.aio(container_client, app_id)

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
