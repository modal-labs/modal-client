import pytest

from modal import Stub, App


@pytest.mark.asyncio
def test_serialize_deserialize_function(servicer, client):
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
    # kind of janky assertion - relies on cloudpickle not obfuscating the object id of the contained reference
    assert bar.object_id.encode("ascii") in foo_def.function_serialized

    app = App.init_container(client, app.app_id)
    app.foo()  # intentional "local" call
    assert len(servicer.client_calls) == 1
