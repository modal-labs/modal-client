# Copyright Modal Labs 2022
import cloudpickle
import pytest

import modal
import modal.aio
from modal_proto import api_pb2

stub = modal.Stub()


@stub.service(cpu=42)
class Foo:
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
        assert foo.bar.call(42) == 1764  # type: ignore


aio_stub = modal.aio.AioStub()


@aio_stub.service(cpu=42)
class Bar:
    def baz(self, x):
        return x**3


@pytest.mark.asyncio
async def test_call_class_async(aio_client, servicer):
    async with aio_stub.run(client=aio_client):
        bar = Bar()
        assert await bar.baz.call(42) == 1764  # type: ignore


def test_run_class_serialized(client, servicer):
    class FooSer:
        def bar(self, x):
            return x**3

    stub_ser = modal.Stub()
    stub_ser.service(cpu=42, serialized=True)(FooSer)

    assert servicer.n_functions == 0
    with stub_ser.run(client=client):
        pass

    assert servicer.n_functions == 1
    (function_id,) = servicer.app_functions.keys()
    function = servicer.app_functions[function_id]
    assert function.function_name.endswith("FooSer.bar")  # because it's defined in a local scope
    assert function.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED
    cls = cloudpickle.loads(function.class_serialized)
    fun = cloudpickle.loads(function.function_serialized)

    # Create bound method
    obj = cls()
    meth = fun.__get__(obj, cls)

    # Make sure it's callable
    assert meth(100) == 1000000


stub_local = modal.Stub()


@stub_local.service(cpu=42)
class FooLocal:
    def bar(self, x):
        return x**3

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
        assert foo.bar.call(8) == 64  # type: ignore
        assert foo.baz.call(9) == 81  # type: ignore
