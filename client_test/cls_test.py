# Copyright Modal Labs 2022
import inspect
import pytest

from modal import Stub, method
from modal._serialization import deserialize
from modal.cls import ClsMixin
from modal.functions import FunctionHandle
from modal_proto import api_pb2

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


# Reusing the stub runs into an issue with stale function handles.
# TODO (akshat): have all the client tests use separate stubs, and throw
# an exception if the user tries to reuse a stub.
stub_remote = Stub()


@stub_remote.cls(cpu=42)
class FooRemote(ClsMixin):
    def __init__(self, x: int, y: str) -> None:
        self.x = x
        self.y = y

    @method()
    def bar(self, z: int):
        return z**3


def test_call_cls_remote_sync(client):
    with stub_remote.run(client=client):
        foo_remote = FooRemote.remote(3, "hello")
        # Mock servicer just squares the argument
        # This means remote function call is taking place.
        assert foo_remote.bar.call(8) == 64
        assert foo_remote.bar(8) == 64


def test_call_cls_remote_invalid_type(client):
    with stub_remote.run(client=client):

        def my_function():
            print("Hello, world!")

        with pytest.raises(ValueError) as excinfo:
            FooRemote.remote(42, my_function)

        exc = excinfo.value
        assert "y=" in str(exc)


stub_2 = Stub()


@stub_2.cls(cpu=42)
class Bar:
    @method()
    def baz(self, x):
        return x**3


@pytest.mark.asyncio
async def test_call_class_async(client, servicer):
    async with stub_2.run(client=client):
        bar = Bar()
        assert await bar.baz.call.aio(42) == 1764


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


stub_remote_2 = Stub()


@stub_remote_2.cls(cpu=42)
class BarRemote(ClsMixin):
    def __init__(self, x: int, y: str) -> None:
        self.x = x
        self.y = y

    @method()
    def baz(self, z: int):
        return z**3


@pytest.mark.asyncio
async def test_call_cls_remote_async(client):
    async with stub_remote_2.run(client=client):
        coro = BarRemote.remote.aio(3, "hello")  # type: ignore
        assert inspect.iscoroutine(coro)
        bar_remote = await coro
        # Mock servicer just squares the argument
        # This means remote function call is taking place.
        assert await bar_remote.baz.call.aio(8) == 64
        assert bar_remote.baz(8) == 64


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
