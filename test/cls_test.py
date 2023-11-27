# Copyright Modal Labs 2022
import inspect
import pytest
import threading
from typing import TYPE_CHECKING

from typing_extensions import assert_type

from modal import Cls, Function, Stub, method
from modal._serialization import deserialize
from modal.app import ContainerApp
from modal.cls import ClsMixin
from modal.exception import DeprecationError, ExecutionError
from modal.runner import deploy_stub
from modal_proto import api_pb2
from modal_test_support.base_class import BaseCls2

stub = Stub("stub")


@stub.cls(cpu=42)
class Foo:
    @method()
    def bar(self, x: int) -> float:
        return x**3


def test_run_class(client, servicer):
    assert servicer.n_functions == 0
    with stub.run(client=client):
        function_id = Foo.bar.object_id
        assert isinstance(Foo, Cls)
        class_id = Foo.object_id
        app_id = stub.app_id

    objects = servicer.app_objects[app_id]
    assert len(objects) == 2  # classes and functions
    assert objects["Foo.bar"] == function_id
    assert objects["Foo"] == class_id


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
        # Check old cls syntax
        with pytest.raises(DeprecationError):
            FooRemote.remote(3, "hello")  # type: ignore

        # Check new syntax
        foo_remote: FooRemote = FooRemote(3, "hello")
        ret: float = foo_remote.bar.remote(8)
        assert ret == 64  # Mock servicer just squares the argument

        # Check old method syntax
        assert foo_remote.bar.remote(8) == 64
        with pytest.raises(DeprecationError):
            foo_remote.bar(8)


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
        bar = Bar()
        assert await bar.baz.remote.aio(42) == 1764

        with pytest.raises(DeprecationError):
            await Bar.remote.aio()  # type: ignore


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
        bar_remote = BarRemote(3, "hello")
        assert await bar_remote.baz.remote.aio(8) == 64  # Mock servicer just squares the argument

        # Check deprecated method syntax
        with pytest.raises(DeprecationError):
            bar_remote.baz(8)

        # Check deprecated cls syntax
        coro = BarRemote.remote.aio(3, "hello")  # type: ignore
        assert inspect.iscoroutine(coro)
        with pytest.raises(DeprecationError):
            await coro


stub_local = Stub()


@stub_local.cls(cpu=42)
class FooLocal:
    @method()
    def bar(self, x):
        return x**3

    @method()
    def baz(self, y):
        return self.bar.local(y + 1)


def test_can_call_locally(client):
    foo = FooLocal()
    assert foo.bar.local(4) == 64
    assert foo.baz.local(4) == 125
    with stub_local.run(client=client):
        assert foo.baz.local(2) == 27


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
        # Check new cls syntax
        foo_remote = NoArgRemote()
        assert foo_remote.baz.remote(8) == 64  # Mock servicer just squares the argument

        # Check old cls syntax
        with pytest.raises(DeprecationError):
            NoArgRemote.remote()  # type: ignore

        # Check old method syntax
        with pytest.raises(DeprecationError):
            foo_remote.baz(8)


def test_deprecated_mixin():
    with pytest.raises(DeprecationError):

        class FooRemote(ClsMixin):
            pass


if TYPE_CHECKING:
    # Check that type annotations carry through to the decorated classes
    assert_type(Foo(), Foo)
    assert_type(Foo().bar, Function)


def test_lookup(client, servicer):
    deploy_stub(stub, "my-cls-app", client=client)

    cls: Cls = Cls.lookup("my-cls-app", "Foo", client=client)

    assert cls.object_id.startswith("cs-")
    assert cls.bar.object_id.startswith("fu-")

    # Check that function properties are preserved
    assert cls.bar.is_generator is False

    # Make sure we can instantiate the class
    obj = cls("foo", 234)

    # Make sure we can methods
    # (mock servicer just returns the sum of the squares of the args)
    assert obj.bar.remote(42, 77) == 7693

    # Make sure local calls fail
    with pytest.raises(ExecutionError):
        assert obj.bar.local(1, 2)


def test_lookup_lazy_remote(client, servicer):
    # See #972 (PR) and #985 (revert PR): adding unit test to catch regression
    deploy_stub(stub, "my-cls-app", client=client)
    cls: Cls = Cls.lookup("my-cls-app", "Foo", client=client)
    obj = cls("foo", 234)
    assert obj.bar.remote(42, 77) == 7693


def test_lookup_lazy_spawn(client, servicer):
    # See #1071
    deploy_stub(stub, "my-cls-app", client=client)
    cls: Cls = Cls.lookup("my-cls-app", "Foo", client=client)
    obj = cls("foo", 234)
    function_call = obj.bar.spawn(42, 77)
    assert function_call.get() == 7693


baz_stub = Stub()


@baz_stub.cls()
class Baz:
    def __init__(self, x):
        self.x = x

    def not_modal_method(self, y: int) -> int:
        return self.x * y


def test_call_not_modal_method():
    baz: Baz = Baz(5)
    assert baz.x == 5
    assert baz.not_modal_method(7) == 35


cls_with_enter_stub = Stub()


def get_thread_id():
    return threading.current_thread().name


@cls_with_enter_stub.cls()
class ClsWithEnter:
    def __init__(self, thread_id):
        self.inited = True
        self.entered = False
        self.thread_id = thread_id
        assert get_thread_id() == self.thread_id

    def __enter__(self):
        self.entered = True
        assert get_thread_id() == self.thread_id

    def not_modal_method(self, y: int) -> int:
        return y**2

    @method()
    def modal_method(self, y: int) -> int:
        return y**2


def test_dont_enter_on_local_access():
    obj = ClsWithEnter(get_thread_id())
    with pytest.raises(AttributeError):
        obj.doesnt_exist  # type: ignore
    assert obj.inited
    assert not obj.entered


def test_dont_enter_on_local_non_modal_call():
    obj = ClsWithEnter(get_thread_id())
    assert obj.not_modal_method(7) == 49
    assert obj.inited
    assert not obj.entered


def test_enter_on_local_modal_call():
    obj = ClsWithEnter(get_thread_id())
    assert obj.modal_method.local(7) == 49
    assert obj.inited
    assert obj.entered


@cls_with_enter_stub.cls()
class ClsWithAenter:
    def __init__(self):
        self.inited = True
        self.entered = False

    async def __aenter__(self):
        self.entered = True

    @method()
    async def modal_method(self, y: int) -> int:
        return y**2


@pytest.mark.asyncio
async def test_aenter_on_local_modal_call():
    obj = ClsWithAenter()
    assert await obj.modal_method.local(7) == 49
    assert obj.inited
    assert obj.entered


inheritance_stub = Stub()


class BaseCls:
    def __enter__(self):
        self.x = 2

    @method()
    def run(self, y):
        return self.x * y


@inheritance_stub.cls()
class DerivedCls(BaseCls):
    pass


def test_derived_cls(client, servicer):
    with inheritance_stub.run(client=client):
        # default servicer fn just squares the number
        assert DerivedCls().run.remote(3) == 9


inheritance_stub_2 = Stub()


@inheritance_stub_2.cls()
class DerivedCls2(BaseCls2):
    pass


def test_derived_cls_external_file(client, servicer):
    with inheritance_stub_2.run(client=client):
        # default servicer fn just squares the number
        assert DerivedCls2().run.remote(3) == 9


def test_rehydrate(client, servicer):
    # Issue introduced in #922 - brief description in #931

    # Sanity check that local calls work
    obj = Foo()
    assert obj.bar.local(7) == 343

    # Deploy stub to get an app id
    app_id = deploy_stub(stub, "my-cls-app", client=client).app_id

    # Initialize a container
    app = ContainerApp()
    app.init(client, app_id, "stub")

    # Associate app with stub
    app._associate_stub_container(stub)

    # Hydration shouldn't overwrite local function definition
    obj = Foo()
    assert obj.bar.local(7) == 343


stub_unhydrated = Stub()


@stub_unhydrated.cls()
class FooUnhydrated:
    @method()
    def bar(self):
        ...


def test_unhydrated():
    foo = FooUnhydrated()
    with pytest.raises(ExecutionError, match="hydrated"):
        foo.bar.remote(42)


stub_method_args = Stub()


@stub_method_args.cls()
class XYZ:
    @method(keep_warm=3)
    def foo(self):
        ...

    @method(keep_warm=7)
    def bar(self):
        ...


def test_method_args(servicer, client):
    with stub_method_args.run(client=client):
        funcs = servicer.app_functions.values()
        assert [f.function_name for f in funcs] == ["XYZ.foo", "XYZ.bar"]
        assert [f.warm_pool_size for f in funcs] == [3, 7]


class ClsWith1Method:
    @method()
    def foo(self):
        ...


class ClsWith2Methods:
    @method()
    def foo(self):
        ...

    @method()
    def bar(self):
        ...


def test_keep_warm_depr():
    stub = Stub()

    # This should be fine
    stub.cls(keep_warm=2)(ClsWith1Method)

    with pytest.warns(DeprecationError, match="@method"):
        stub.cls(keep_warm=2)(ClsWith2Methods)
