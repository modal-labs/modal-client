# Copyright Modal Labs 2022
import inspect
import pytest
from typing import TYPE_CHECKING

from typing_extensions import assert_type

from modal import Function, Stub, method
from modal._serialization import deserialize
from modal.cls import ClsMixin
from modal.exception import DeprecationError
from modal_proto import api_pb2

stub = Stub()


@stub.cls(cpu=42)
class Foo:
    @method()
    def bar(self, x: int) -> float:
        return x**3.5


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
        foo: Foo = Foo()
        ret: float = foo.bar.remote(42)
        assert ret == 1764


# Reusing the stub runs into an issue with stale function handles.
# TODO (akshat): have all the client tests use separate stubs, and throw
# an exception if the user tries to reuse a stub.
stub_remote = Stub()


@stub_remote.cls(cpu=42)
class FooRemote:
    def __init__(self, x: int, y: str) -> None:
        self.x = x
        self.y = y

    @method()
    def bar(self, z: int):
        return z**3


def test_call_cls_remote_sync(client):
    with stub_remote.run(client=client):
        foo_remote: FooRemote
        with pytest.warns(DeprecationError):
            foo_remote = FooRemote.remote(3, "hello")  # type: ignore
        # Mock servicer just squares the argument
        # This means remote function call is taking place.
        assert foo_remote.bar.remote(8) == 64
        with pytest.warns(DeprecationError):
            assert foo_remote.bar(8) == 64

        # Check new syntax
        foo_remote_2: FooRemote = FooRemote(3, "hello")
        ret: float = foo_remote_2.bar.remote(8)
        assert ret == 64


def test_call_cls_remote_invalid_type(client):
    with stub_remote.run(client=client):

        def my_function():
            print("Hello, world!")

        with pytest.raises(ValueError) as excinfo:
            FooRemote(42, my_function)  # type: ignore

        exc = excinfo.value
        assert "function" in str(exc)


stub_2 = Stub()


@stub_2.cls(cpu=42)
class Bar:
    @method()
    def baz(self, x):
        return x**3


@pytest.mark.asyncio
async def test_call_class_async(client, servicer):
    async with stub_2.run(client=client):
        with pytest.warns(DeprecationError):
            bar = await Bar.remote.aio()  # type: ignore
        bar = Bar()
        assert await bar.baz.remote.aio(42) == 1764


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

    # Make sure it's callable
    assert meth(100) == 1000000


stub_remote_2 = Stub()


@stub_remote_2.cls(cpu=42)
class BarRemote:
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
        with pytest.warns(DeprecationError):
            bar_remote = await coro
        # Mock servicer just squares the argument
        # This means remote function call is taking place.
        assert await bar_remote.baz.remote.aio(8) == 64
        with pytest.warns(DeprecationError):
            assert bar_remote.baz(8) == 64


stub_local = Stub()


@stub_local.cls(cpu=42)
class FooLocal:
    @method()
    def bar(self, x):
        return x**3

    @method()
    def baz(self, y):
        return self.bar.local(y + 1)


def test_can_call_locally():
    foo = FooLocal()
    assert foo.bar.local(4) == 64
    assert foo.baz.local(4) == 125


def test_can_call_remotely_from_local(client):
    with stub_local.run(client=client):
        foo = FooLocal()
        # remote calls use the mockservicer func impl
        # which just squares the arguments
        assert foo.bar.remote(8) == 64
        assert foo.baz.remote(9) == 81


stub_remote_3 = Stub()


@stub_remote_3.cls(cpu=42)
class NoArgRemote:
    def __init__(self) -> None:
        pass

    @method()
    def baz(self, z: int):
        return z**3


def test_call_cls_remote_no_args(client):
    with stub_remote_3.run(client=client):
        with pytest.warns(DeprecationError):
            foo_remote = NoArgRemote.remote()  # type: ignore
        # Mock servicer just squares the argument
        # This means remote function call is taking place.
        with pytest.warns(DeprecationError):
            assert foo_remote.baz(8) == 64

        foo_remote = NoArgRemote()
        assert foo_remote.baz.remote(8) == 64


def test_deprecated_mixin():
    with pytest.warns(DeprecationError):

        class FooRemote(ClsMixin):
            pass


if TYPE_CHECKING:
    # Check that type annotations carry through to the decorated classes
    assert_type(Foo(), Foo)
    assert_type(Foo().bar, Function)
