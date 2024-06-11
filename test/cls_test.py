# Copyright Modal Labs 2022
import inspect
import pytest
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Dict

from typing_extensions import assert_type

import modal.partial_function
from modal import App, Cls, Function, Image, Queue, build, enter, exit, method
from modal._serialization import deserialize, serialize
from modal._utils.async_utils import synchronizer
from modal.exception import DeprecationError, ExecutionError, InvalidError
from modal.partial_function import (
    PartialFunction,
    _find_callables_for_obj,
    _find_partial_methods_for_cls,
    _PartialFunction,
    _PartialFunctionFlags,
)
from modal.runner import deploy_app
from modal.running_app import RunningApp
from modal_proto import api_pb2

from .supports.base_class import BaseCls2

app = App("app")


@pytest.fixture(autouse=True)
def auto_use_set_env_client(set_env_client):
    # TODO(elias): remove set_env_client fixture here if/when possible - this is required only since
    #  Client.from_env happens to inject an unused client when loading the
    #  parameterized function
    return


@app.cls(cpu=42)
class Foo:
    @method()
    def bar(self, x: int) -> float:
        return x**3


def test_run_class(client, servicer):
    assert servicer.n_functions == 0
    with app.run(client=client):
        function_id = Foo.bar.object_id
        assert isinstance(Foo, Cls)
        class_id = Foo.object_id
        app_id = app.app_id

    objects = servicer.app_objects[app_id]
    assert len(objects) == 2  # classes and functions
    assert objects["Foo.bar"] == function_id
    assert objects["Foo"] == class_id


def test_call_class_sync(client, servicer):
    with app.run(client=client):
        foo: Foo = Foo()
        ret: float = foo.bar.remote(42)
        assert ret == 1764


# Reusing the app runs into an issue with stale function handles.
# TODO (akshat): have all the client tests use separate apps, and throw
# an exception if the user tries to reuse an app.
app_remote = App()


@app_remote.cls(cpu=42)
class FooRemote:
    def __init__(self, x: int, y: str) -> None:
        self.x = x
        self.y = y

    @method()
    def bar(self, z: int):
        return z**3


def test_call_cls_remote_sync(client):
    with app_remote.run(client=client):
        foo_remote: FooRemote = FooRemote(3, "hello")
        ret: float = foo_remote.bar.remote(8)
        assert ret == 64  # Mock servicer just squares the argument


def test_call_cls_remote_invalid_type(client):
    with app_remote.run(client=client):

        def my_function():
            print("Hello, world!")

        with pytest.raises(ValueError) as excinfo:
            FooRemote(42, my_function)  # type: ignore

        exc = excinfo.value
        assert "function" in str(exc)


def test_call_cls_remote_modal_type(client):
    with app_remote.run(client=client):
        with Queue.ephemeral(client) as q:
            FooRemote(42, q)  # type: ignore


app_2 = App()


@app_2.cls(cpu=42)
class Bar:
    @method()
    def baz(self, x):
        return x**3


@pytest.mark.asyncio
async def test_call_class_async(client, servicer):
    async with app_2.run(client=client):
        bar = Bar()
        assert await bar.baz.remote.aio(42) == 1764


def test_run_class_serialized(client, servicer):
    app_ser = App()

    @app_ser.cls(cpu=42, serialized=True)
    class FooSer:
        @method()
        def bar(self, x):
            return x**3

    assert servicer.n_functions == 0
    with app_ser.run(client=client):
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


app_remote_2 = App()


@app_remote_2.cls(cpu=42)
class BarRemote:
    def __init__(self, x: int, y: str) -> None:
        self.x = x
        self.y = y

    @method()
    def baz(self, z: int):
        return z**3


@pytest.mark.asyncio
async def test_call_cls_remote_async(client):
    async with app_remote_2.run(client=client):
        bar_remote = BarRemote(3, "hello")
        assert await bar_remote.baz.remote.aio(8) == 64  # Mock servicer just squares the argument


app_local = App()


@app_local.cls(cpu=42, enable_memory_snapshot=True)
class FooLocal:
    def __init__(self):
        self.side_effects = ["__init__"]

    @enter(snap=True)
    def presnap(self):
        self.side_effects.append("presnap")

    @enter()
    def postsnap(self):
        self.side_effects.append("postsnap")

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
    with app_local.run(client=client):
        assert foo.baz.local(2) == 27
        assert foo.side_effects == ["__init__", "presnap", "postsnap"]


def test_can_call_remotely_from_local(client):
    with app_local.run(client=client):
        foo = FooLocal()
        # remote calls use the mockservicer func impl
        # which just squares the arguments
        assert foo.bar.remote(8) == 64
        assert foo.baz.remote(9) == 81


app_remote_3 = App()


@app_remote_3.cls(cpu=42)
class NoArgRemote:
    def __init__(self) -> None:
        pass

    @method()
    def baz(self, z: int):
        return z**3


def test_call_cls_remote_no_args(client):
    with app_remote_3.run(client=client):
        foo_remote = NoArgRemote()
        assert foo_remote.baz.remote(8) == 64  # Mock servicer just squares the argument


if TYPE_CHECKING:
    # Check that type annotations carry through to the decorated classes
    assert_type(Foo(), Foo)
    assert_type(Foo().bar, Function)


def test_lookup(client, servicer):
    deploy_app(app, "my-cls-app", client=client)

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
    deploy_app(app, "my-cls-app", client=client)
    cls: Cls = Cls.lookup("my-cls-app", "Foo", client=client)
    obj = cls("foo", 234)
    assert obj.bar.remote(42, 77) == 7693


def test_lookup_lazy_spawn(client, servicer):
    # See #1071
    deploy_app(app, "my-cls-app", client=client)
    cls: Cls = Cls.lookup("my-cls-app", "Foo", client=client)
    obj = cls("foo", 234)
    function_call = obj.bar.spawn(42, 77)
    assert function_call.get() == 7693


baz_app = App()


@baz_app.cls()
class Baz:
    def __init__(self, x):
        self.x = x

    def not_modal_method(self, y: int) -> int:
        return self.x * y


def test_call_not_modal_method():
    baz: Baz = Baz(5)
    assert baz.x == 5
    assert baz.not_modal_method(7) == 35


cls_with_enter_app = App()


def get_thread_id():
    return threading.current_thread().name


@cls_with_enter_app.cls()
class ClsWithEnter:
    def __init__(self, thread_id):
        self.inited = True
        self.entered = False
        self.thread_id = thread_id
        assert get_thread_id() == self.thread_id

    @enter()
    def enter(self):
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


@cls_with_enter_app.cls()
class ClsWithAsyncEnter:
    def __init__(self):
        self.inited = True
        self.entered = False

    @enter()
    async def enter(self):
        self.entered = True

    @method()
    async def modal_method(self, y: int) -> int:
        return y**2


@pytest.mark.asyncio
async def test_async_enter_on_local_modal_call():
    obj = ClsWithAsyncEnter()
    assert await obj.modal_method.local(7) == 49
    assert obj.inited
    assert obj.entered


inheritance_app = App()


class BaseCls:
    @enter()
    def enter(self):
        self.x = 2

    @method()
    def run(self, y):
        return self.x * y


@inheritance_app.cls()
class DerivedCls(BaseCls):
    pass


def test_derived_cls(client, servicer):
    with inheritance_app.run(client=client):
        # default servicer fn just squares the number
        assert DerivedCls().run.remote(3) == 9


inheritance_app_2 = App()


@inheritance_app_2.cls()
class DerivedCls2(BaseCls2):
    pass


def test_derived_cls_external_file(client, servicer):
    with inheritance_app_2.run(client=client):
        # default servicer fn just squares the number
        assert DerivedCls2().run.remote(3) == 9


def test_rehydrate(client, servicer, reset_container_app):
    # Issue introduced in #922 - brief description in #931

    # Sanity check that local calls work
    obj = Foo()
    assert obj.bar.local(7) == 343

    # Deploy app to get an app id
    app_id = deploy_app(app, "my-cls-app", client=client).app_id

    # Initialize a container
    container_app = RunningApp(app_id=app_id)

    # Associate app with app
    app._init_container(client, container_app)

    # Hydration shouldn't overwrite local function definition
    obj = Foo()
    assert obj.bar.local(7) == 343


app_unhydrated = App()


@app_unhydrated.cls()
class FooUnhydrated:
    @method()
    def bar(self):
        ...


def test_unhydrated():
    foo = FooUnhydrated()
    with pytest.raises(ExecutionError, match="hydrated"):
        foo.bar.remote(42)


app_method_args = App()


@app_method_args.cls()
class XYZ:
    @method(keep_warm=3)
    def foo(self):
        ...

    @method(keep_warm=7)
    def bar(self):
        ...


def test_method_args(servicer, client):
    with app_method_args.run(client=client):
        funcs = servicer.app_functions.values()
        assert {f.function_name for f in funcs} == {"XYZ.foo", "XYZ.bar"}
        assert {f.warm_pool_size for f in funcs} == {3, 7}


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
    app = App()

    # This should be fine
    app.cls(keep_warm=2)(ClsWith1Method)

    with pytest.warns(DeprecationError, match="@method"):
        app.cls(keep_warm=2)(ClsWith2Methods)


class ClsWithHandlers:
    @build()
    def my_build(self):
        pass

    @enter(snap=True)
    def my_memory_snapshot(self):
        pass

    @enter()
    def my_enter(self):
        pass

    @build()
    @enter()
    def my_build_and_enter(self):
        pass

    @exit()
    def my_exit(self):
        pass


def test_handlers():
    pfs: Dict[str, _PartialFunction]

    pfs = _find_partial_methods_for_cls(ClsWithHandlers, _PartialFunctionFlags.BUILD)
    assert list(pfs.keys()) == ["my_build", "my_build_and_enter"]

    pfs = _find_partial_methods_for_cls(ClsWithHandlers, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT)
    assert list(pfs.keys()) == ["my_memory_snapshot"]

    pfs = _find_partial_methods_for_cls(ClsWithHandlers, _PartialFunctionFlags.ENTER_POST_SNAPSHOT)
    assert list(pfs.keys()) == ["my_enter", "my_build_and_enter"]

    pfs = _find_partial_methods_for_cls(ClsWithHandlers, _PartialFunctionFlags.EXIT)
    assert list(pfs.keys()) == ["my_exit"]


handler_app = App("handler-app")


image = Image.debian_slim().pip_install("xyz")


@handler_app.cls(image=image)
class ClsWithBuild:
    @build()
    def build(self):
        pass

    @method()
    def method(self):
        pass


def test_build_image(client, servicer):
    with handler_app.run(client=client):
        f_def = servicer.app_functions[ClsWithBuild.method.object_id]
        # The function image should have added a new layer with original image as the parent
        f_image = servicer.images[f_def.image_id]
        assert f_image.base_images[0].image_id == image.object_id


@pytest.mark.parametrize("decorator", [build, enter, exit])
def test_disallow_lifecycle_decorators_with_method(decorator):
    name = decorator.__name__.split("_")[-1]  # remove synchronicity prefix
    with pytest.raises(InvalidError, match=f"Cannot use `@{name}` decorator with `@method`."):

        class ClsDecoratorMethodStack:
            @decorator()
            @method()
            def f(self):
                pass


def test_deprecated_sync_methods():
    with pytest.warns(DeprecationError, match="Support for decorating parameterized methods with `@exit`"):

        class ClsWithDeprecatedSyncMethods:
            def __enter__(self):
                return 42

            @enter()
            def my_enter(self):
                return 43

            def __exit__(self, exc_type, exc, tb):
                return 44

            @exit()
            def my_exit(self, exc_type, exc, tb):
                return 45

    obj = ClsWithDeprecatedSyncMethods()

    with pytest.raises(DeprecationError, match="Using `__enter__`.+`modal.enter` decorator"):
        _find_callables_for_obj(obj, _PartialFunctionFlags.ENTER_POST_SNAPSHOT)

    with pytest.raises(DeprecationError, match="Using `__exit__`.+`modal.exit` decorator"):
        _find_callables_for_obj(obj, _PartialFunctionFlags.EXIT)


@pytest.mark.asyncio
async def test_deprecated_async_methods():
    with pytest.warns(DeprecationError, match="Support for decorating parameterized methods with `@exit`"):

        class ClsWithDeprecatedAsyncMethods:
            async def __aenter__(self):
                return 42

            @enter()
            async def my_enter(self):
                return 43

            async def __aexit__(self, exc_type, exc, tb):
                return 44

            @exit()
            async def my_exit(self, exc_type, exc, tb):
                return 45

    obj = ClsWithDeprecatedAsyncMethods()

    with pytest.raises(DeprecationError, match=r"Using `__aenter__`.+`modal.enter` decorator \(on an async method\)"):
        _find_callables_for_obj(obj, _PartialFunctionFlags.ENTER_POST_SNAPSHOT)

    with pytest.raises(DeprecationError, match=r"Using `__aexit__`.+`modal.exit` decorator \(on an async method\)"):
        _find_callables_for_obj(obj, _PartialFunctionFlags.EXIT)


class HasSnapMethod:
    @enter(snap=True)
    def enter(self):
        pass

    @method()
    def f(self):
        pass


def test_snap_method_without_snapshot_enabled():
    with pytest.raises(InvalidError, match="A class must have `enable_memory_snapshot=True`"):
        app.cls(enable_memory_snapshot=False)(HasSnapMethod)


def test_partial_function_descriptors(client):
    class Foo:
        def __init__(self):
            pass

        @modal.enter()
        def enter_method(self):
            pass

        @modal.method()
        def bar(self):
            return "a"

        @modal.web_endpoint()
        def web(self):
            pass

    assert isinstance(Foo.bar, PartialFunction)

    assert Foo().bar() == "a"
    assert inspect.ismethod(Foo().bar)
    app = modal.App()

    modal_foo_class = app.cls(serialized=True)(Foo)

    wrapped_method = modal_foo_class().bar
    assert isinstance(wrapped_method, Function)

    serialized_class = serialize(Foo)
    revived_class = deserialize(serialized_class, client)

    assert (
        revived_class().bar() == "a"
    )  # this instantiates the underlying "user_cls", so it should work basically like a normal Python class
    assert isinstance(
        revived_class.bar, PartialFunction
    )  # but it should be a PartialFunction, so it keeps associated metadata!

    # ensure that webhook metadata is kept
    assert synchronizer._translate_in(revived_class.web).webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION


def test_cross_process_userclass_serde(supports_dir):
    res = subprocess.check_output([sys.executable, supports_dir / "serialize_class.py"])
    assert len(res) < 2000  # should be ~1300 bytes as of 2024-06-05
    revived_cls = deserialize(res, None)
    method_without_descriptor_protocol = revived_cls.__dict__["method"]
    assert isinstance(method_without_descriptor_protocol, modal.partial_function.PartialFunction)
    assert revived_cls().method() == "a"  # this should be bound to the object
