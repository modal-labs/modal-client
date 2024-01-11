# Copyright Modal Labs 2022
import importlib
import os
import pytest
from unittest import mock

import modal.secret
from modal import Dict, Stub
from modal.app import container_app
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

    await container_app.init.aio(container_client, "ap-123")
    stub = Stub()

    # Now, let's create my_f after the app started running and make sure it works
    my_f_container = stub.function()(my_f_1)
    assert await my_f_container.remote.aio(42) == 1764  # type: ignore

    # Also make sure dicts work
    my_d_container = Dict.new()
    stub.my_d = my_d_container  # should trigger id assignment
    assert my_d_container.object_id == "di-123"


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
