# Copyright Modal Labs 2023
import pytest

from modal import App
from modal._serialization import deserialize


@pytest.mark.asyncio
async def test_serialize_deserialize_function(servicer, client):
    app = App()

    @app.function(serialized=True, name="foo")
    def foo():
        2 * foo.remote()

    assert not foo.is_hydrated
    with pytest.raises(Exception):
        foo.object_id  # noqa

    with app.run(client=client):
        object_id = foo.object_id

    assert object_id is not None
    assert {object_id} == servicer.precreated_functions

    foo_def = servicer.app_functions[object_id]

    assert len(servicer.function_call_inputs) == 0

    deserialized_function_body = deserialize(foo_def.function_serialized, client)
    deserialized_function_body()  # call locally as if in container, this should trigger a "remote" foo() call
    assert len(servicer.function_call_inputs) == 1
    function_call_id = list(servicer.function_call_inputs.keys())[0]
    assert servicer.function_id_for_function_call[function_call_id] == object_id
