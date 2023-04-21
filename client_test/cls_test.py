# Copyright Modal Labs 2022
import pytest

from modal import Stub, method
from modal.aio import AioStub, aio_method
from modal.functions import FunctionHandle
from modal_proto import api_pb2
from modal._serialization import deserialize

stub = Stub()


@stub.cls(cpu=42)
class Foo:
    @method()
    def bar(self, x):
        return x**3


def test_run_class(client, servicer):
    assert servicer.n_functions == 0
    with stub.run(client=client) as app:
        pass

    assert servicer.n_functions == 1
    (function_id,) = servicer.app_functions.keys()
    function = servicer.app_functions[function_id]
    assert function.function_name == "Foo.bar"
    objects = servicer.app_objects[app.app_id]
    assert objects == {"Foo.bar": function_id}


def test_call_class_sync(client, servicer):
    with stub.run(client=client):
        foo = Foo()
        assert foo.bar.call(42) == 1764


aio_stub = AioStub()


@aio_stub.cls(cpu=42)
class Bar:
    @aio_method()
    def baz(self, x):
        return x**3


@pytest.mark.asyncio
async def test_call_class_async(aio_client, servicer):
    async with aio_stub.run(client=aio_client):
        bar = Bar()
        assert await bar.baz.call(42) == 1764


def test_run_class_serialized(client, servicer):
    stub_ser = Stub()

    @stub_ser.cls(cpu=42, serialized=True)
    class FooSer:
        @method()
        def bar(self, x):
            return x**3

    assert servicer.n_functions == 0
    with stub_ser.run(client=client):
        pass

    assert servicer.n_functions == 1
    (function_id,) = servicer.app_functions.keys()
    function = servicer.app_functions[function_id]
    assert function.function_name.endswith("FooSer.bar")  # because it's defined in a local scope
    assert function.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED
    cls = deserialize(function.class_serialized, client)
    fun = deserialize(function.function_serialized, client)

    # Create bound method
    obj = cls()
    meth = fun.__get__(obj, cls)

    assert isinstance(obj.bar, FunctionHandle)
    # Make sure it's callable
    assert meth(100) == 1000000


stub_local = Stub()


@stub_local.cls(cpu=42)
class FooLocal:
    @method()
    def bar(self, x):
        return x**3

    @method()
    def baz(self, y):
        return self.bar(y + 1)


def test_can_call_locally():
    foo = FooLocal()
    assert foo.bar(4) == 64
    assert foo.baz(4) == 125


def test_can_call_remotely_from_local(client):
    with stub_local.run(client=client):
        foo = FooLocal()
        # remote calls use the mockservicer func impl
        # which just squares the arguments
        assert foo.bar.call(8) == 64
        assert foo.baz.call(9) == 81
