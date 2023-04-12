# Copyright Modal Labs 2023
import pytest

from modal import Stub
from modal.app import App
from modal._serialization import deserialize


@pytest.mark.asyncio
async def test_serialize_deserialize_function(servicer, client):
    stub = Stub()

    @stub.function(serialized=True, name="foo")
    def foo():
        2 * bar.call()

    @stub.function(serialized=True, name="bar")
    def bar():
        return 5

    assert foo.object_id is None
    assert bar.object_id is None

    with stub.run(client=client) as app:
        pass

    assert foo.object_id is not None
    assert bar.object_id is not None
    assert {foo.object_id, bar.object_id} == servicer.reserved_functions

    foo_def = servicer.app_functions[foo.object_id]

    assert len(servicer.client_calls) == 0
    from modal_utils.async_utils import synchronizer

    container_app = App.init_container(client, app.app_id)
    _container_app = synchronizer._translate_in(container_app)
    deserialized_function = deserialize(foo_def.function_serialized, client, _container_app)
    deserialized_function()  # call locally as if in container, this should trigger a "remote" bar() call
    assert len(servicer.client_calls) == 1
    function_call_id = list(servicer.client_calls.keys())[0]
    assert servicer.function_id_for_function_call[function_call_id] == bar.object_id
