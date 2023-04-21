# Copyright Modal Labs 2023
import pytest

from modal import Stub
from modal._serialization import deserialize


@pytest.mark.asyncio
async def test_serialize_deserialize_function(servicer, client):
    stub = Stub()

    @stub.function(serialized=True, name="foo")
    def foo():
        2 * foo.call()

    assert foo.object_id is None

    with stub.run(client=client):
        pass

    assert foo.object_id is not None
    assert {foo.object_id} == servicer.precreated_functions

    foo_def = servicer.app_functions[foo.object_id]

    assert len(servicer.client_calls) == 0

    deserialized_function_body = deserialize(foo_def.function_serialized, client)
    deserialized_function_body()  # call locally as if in container, this should trigger a "remote" foo() call
    assert len(servicer.client_calls) == 1
    function_call_id = list(servicer.client_calls.keys())[0]
    assert servicer.function_id_for_function_call[function_call_id] == foo.object_id
