# Copyright Modal Labs 2022
import inspect
import pytest
import subprocess
import sys
import threading
import typing
from typing import TYPE_CHECKING

from typing_extensions import assert_type

import modal.partial_function
from modal import App, Cls, Function, Image, Queue, build, enter, exit, method
from modal._partial_function import (
    _find_partial_methods_for_user_cls,
    _PartialFunction,
    _PartialFunctionFlags,
)
from modal._serialization import deserialize, deserialize_params, serialize
from modal._utils.async_utils import synchronizer
from modal._utils.function_utils import FunctionInfo
from modal.exception import DeprecationError, ExecutionError, InvalidError, NotFoundError
from modal.partial_function import (
    PartialFunction,
    asgi_app,
    fastapi_endpoint,
)
from modal.runner import deploy_app
from modal.running_app import RunningApp
from modal_proto import api_pb2

from .supports.base_class import BaseCls2

app = App("app", include_source=True)


@pytest.fixture(autouse=True)
def auto_use_set_env_client(set_env_client):
    # TODO(elias): remove set_env_client fixture here if/when possible - this is required only since
    #  Client.from_env happens to inject an unused client when loading the
    #  parametrized function
    return


@app.cls()
class NoParamsCls:
    @method()
    def bar(self, x):
        return x**3

    @method()
    def baz(self, x):
        return x**2


@app.cls(cpu=42)
class Foo:
    @method()
    def bar(self, x: int) -> float:
        return x**3

    @method()
    def baz(self, y: int) -> float:
        return y**4


def test_run_class(client, servicer):
    assert len(servicer.precreated_functions) == 0
    assert servicer.n_functions == 0
    with app.run(client=client):
        method_handle_object_id = Foo._get_class_service_function().object_id  # type: ignore
        assert isinstance(Foo, Cls)
        assert isinstance(NoParamsCls, Cls)
        class_id = Foo.object_id
        class_id2 = NoParamsCls.object_id
        app_id = app.app_id

    assert len(servicer.classes) == 2 and set(servicer.classes) == {class_id, class_id2}
    assert servicer.n_functions == 2
    objects = servicer.app_objects[app_id]
    class_function_id = objects["Foo.*"]
    class_function_id2 = objects["NoParamsCls.*"]
    assert servicer.precreated_functions == {class_function_id, class_function_id2}
    assert method_handle_object_id == class_function_id  # method handle object id will probably go away
    assert len(objects) == 4  # two classes + two class service function
    assert objects["Foo"] == class_id
    assert class_function_id.startswith("fu-")
    assert servicer.app_functions[class_function_id].is_class
    assert servicer.app_functions[class_function_id].method_definitions == {
        "bar": api_pb2.MethodDefinition(
            function_name="Foo.bar",
            function_type=api_pb2.Function.FunctionType.FUNCTION_TYPE_FUNCTION,
        ),
        "baz": api_pb2.MethodDefinition(
            function_name="Foo.baz",
            function_type=api_pb2.Function.FunctionType.FUNCTION_TYPE_FUNCTION,
        ),
    }


def test_call_class_sync(client, servicer, set_env_client):
    with servicer.intercept() as ctx:
        with app.run(client=client):
            assert len(ctx.get_requests("FunctionCreate")) == 2  # one for Foo, one for NoParamsCls
            foo: NoParamsCls = NoParamsCls()
            assert len(ctx.get_requests("FunctionCreate")) == 2
            assert len(ctx.get_requests("FunctionBindParams")) == 0  # no binding, yet
            ret: float = foo.bar.remote(42)
            assert ret == 1764
            assert (
                len(ctx.get_requests("FunctionBindParams")) == 0
            )  # reuse class base function when class has no params

    function_creates_requests: list[api_pb2.FunctionCreateRequest] = ctx.get_requests("FunctionCreate")
    assert len(function_creates_requests) == 2
    assert len(ctx.get_requests("ClassCreate")) == 2
    function_creates = {fc.function.function_name: fc for fc in function_creates_requests}
    assert function_creates.keys() == {"Foo.*", "NoParamsCls.*"}
    service_function_id = servicer.app_objects["ap-1"]["NoParamsCls.*"]
    (function_map_request,) = ctx.get_requests("FunctionMap")
    assert function_map_request.function_id == service_function_id


def test_class_with_options(client, servicer):
    unhydrated_volume = modal.Volume.from_name("some_volume", create_if_missing=True)
    unhydrated_secret = modal.Secret.from_dict({"foo": "bar"})
    with servicer.intercept() as ctx:
        foo = Foo.with_options(  # type: ignore
            cpu=48, retries=5, volumes={"/vol": unhydrated_volume}, secrets=[unhydrated_secret]
        )()
        assert len(ctx.calls) == 0  # no rpcs in with_options

    with app.run(client=client):
        with servicer.intercept() as ctx:
            res = foo.bar.remote(2)
            function_bind_params: api_pb2.FunctionBindParamsRequest
            (function_bind_params,) = ctx.get_requests("FunctionBindParams")
            assert function_bind_params.function_options.retry_policy.retries == 5
            assert function_bind_params.function_options.resources.milli_cpu == 48000

            assert len(ctx.get_requests("VolumeGetOrCreate")) == 1
            assert len(ctx.get_requests("SecretGetOrCreate")) == 1

        with servicer.intercept() as ctx:
            res = foo.bar.remote(2)
            assert len(ctx.get_requests("FunctionBindParams")) == 0  # no need to rebind
            assert len(ctx.get_requests("VolumeGetOrCreate")) == 0  # no need to rehydrate
            assert len(ctx.get_requests("SecretGetOrCreate")) == 0  # no need to rehydrate

        assert res == 4
        assert len(servicer.function_options) == 1
        options: api_pb2.FunctionOptions = list(servicer.function_options.values())[0]
        assert options.resources.milli_cpu == 48_000
        assert options.retry_policy.retries == 5

        with pytest.warns(DeprecationError, match="max_containers"):
            Foo.with_options(concurrency_limit=10)()  # type: ignore


def test_with_options_from_name(servicer):
    unhydrated_volume = modal.Volume.from_name("some_volume", create_if_missing=True)
    unhydrated_secret = modal.Secret.from_dict({"foo": "bar"})

    with servicer.intercept() as ctx:
        SomeClass = modal.Cls.from_name("some_app", "SomeClass")
        OptionedClass = SomeClass.with_options(cpu=10, secrets=[unhydrated_secret], volumes={"/vol": unhydrated_volume})
        inst = OptionedClass(x=10)
        assert len(ctx.calls) == 0

    with servicer.intercept() as ctx:
        ctx.add_response("VolumeGetOrCreate", api_pb2.VolumeGetOrCreateResponse(volume_id="vo-123"))
        ctx.add_response("SecretGetOrCreate", api_pb2.SecretGetOrCreateResponse(secret_id="st-123"))
        ctx.add_response("ClassGet", api_pb2.ClassGetResponse(class_id="cs-123"))
        ctx.add_response(
            "FunctionGet",
            api_pb2.FunctionGetResponse(
                function_id="fu-123",
                handle_metadata=api_pb2.FunctionHandleMetadata(
                    method_handle_metadata={
                        "some_method": api_pb2.FunctionHandleMetadata(
                            use_function_id="fu-123",
                            use_method_name="some_method",
                            function_name="SomeClass.some_method",
                        )
                    }
                ),
            ),
        )
        ctx.add_response("FunctionBindParams", api_pb2.FunctionBindParamsResponse(bound_function_id="fu-124"))
        inst.some_method.remote()

    function_bind_params: api_pb2.FunctionBindParamsRequest
    (function_bind_params,) = ctx.get_requests("FunctionBindParams")
    assert len(function_bind_params.function_options.volume_mounts) == 1
    function_map: api_pb2.FunctionMapRequest
    (function_map,) = ctx.get_requests("FunctionMap")
    assert function_map.function_id == "fu-124"  # the bound function


# Reusing the app runs into an issue with stale function handles.
# TODO (akshat): have all the client tests use separate apps, and throw
# an exception if the user tries to reuse an app.
app_remote = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


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


app_2 = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


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
    class_function = servicer.function_by_name("FooSer.*")
    assert class_function.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED
    user_cls = deserialize(class_function.class_serialized, client)

    # Create bound method
    obj = user_cls()
    bound_bar = user_cls.bar.__get__(obj)
    # Make sure it's callable
    assert bound_bar(100) == 1000000


app_remote_2 = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


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


app_local = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


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


app_remote_3 = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


@app_remote_3.cls(cpu=42)
class NoArgRemote:
    def __init__(self) -> None:
        pass

    @method()
    def baz(self, z: int) -> float:
        return z**3.0


def test_call_cls_remote_no_args(client):
    with app_remote_3.run(client=client):
        foo_remote = NoArgRemote()
        assert foo_remote.baz.remote(8) == 64  # Mock servicer just squares the argument


if TYPE_CHECKING:
    # Check that type annotations carry through to the decorated classes
    assert_type(NoParamsCls(), NoParamsCls)
    # can't use assert_type with named arguments, as it will diff in the name
    # vs the anonymous argument in the assertion type
    # assert_type(Foo().bar, Function[[int], float])


def test_lookup(client, servicer):
    # basically same test as test_from_name_lazy_method_resolve, but assumes everything is hydrated
    deploy_app(app, "my-cls-app", client=client)

    with pytest.warns(DeprecationError, match="Cls.lookup"):
        cls: Cls = Cls.lookup("my-cls-app", "Foo", client=client)

    # objects are resolved
    assert cls.object_id.startswith("cs-")
    assert cls._get_class_service_function().object_id.startswith("fu-")

    # Check that function properties are preserved
    assert cls().bar.is_generator is False

    # Make sure we can instantiate the class
    with servicer.intercept() as ctx:
        obj = cls("foo", 234)
        assert len(ctx.calls) == 0  # no rpc requests for class instantiation

        # Make sure we can call methods
        # (mock servicer just returns the sum of the squares of the args)
        assert obj.bar.remote(42) == 1764
        assert len(ctx.get_requests("FunctionBindParams")) == 1  # bind params

        assert obj.baz.remote(41) == 1681
        assert len(ctx.get_requests("FunctionBindParams")) == 1  # call to other method shouldn't need a bind

    # Not allowed for remote classes:
    with pytest.raises(NotFoundError, match="can't be accessed for remote classes"):
        assert obj.a == "foo"

    # Make sure local calls fail
    with pytest.raises(ExecutionError):
        assert obj.bar.local(1, 2)


def test_from_name_lazy_method_hydration(client, servicer):
    deploy_app(app, "my-cls-app", client=client)
    cls: Cls = Cls.from_name("my-cls-app", "Foo")

    # Make sure we can instantiate the class
    obj = cls("foo", 234)

    # Check that function properties are preserved
    with servicer.intercept() as ctx:
        assert obj.bar.is_generator is False
        assert len(ctx.get_requests("FunctionBindParams")) == 1  # to determine this attribute, hydration is needed

    # Make sure we can methods
    # (mock servicer just returns the sum of the squares of the args)
    with servicer.intercept() as ctx:
        assert obj.bar.remote(42) == 1764
        assert len(ctx.get_requests("FunctionBindParams")) == 0

    with servicer.intercept() as ctx:
        assert obj.baz.remote(42) == 1764
        assert len(ctx.get_requests("FunctionBindParams")) == 0  # other method shouldn't rebind

    with pytest.raises(NotFoundError, match="can't be accessed for remote classes"):
        assert obj.a == 234

    # Make sure local calls fail
    with pytest.raises(ExecutionError, match="locally"):
        assert obj.bar.local(1, 2)

    # Make sure that non-existing methods fail
    with pytest.raises(NotFoundError):
        obj.non_exist.remote("hello")


def test_lookup_lazy_remote(client, servicer):
    # See #972 (PR) and #985 (revert PR): adding unit test to catch regression
    deploy_app(app, "my-cls-app", client=client)
    cls: Cls = Cls.from_name("my-cls-app", "Foo").hydrate(client=client)
    obj = cls("foo", 234)
    assert obj.bar.remote(42, 77) == 7693


def test_lookup_lazy_spawn(client, servicer):
    # See #1071
    deploy_app(app, "my-cls-app", client=client)
    cls: Cls = Cls.from_name("my-cls-app", "Foo").hydrate(client=client)
    obj = cls("foo", 234)
    function_call = obj.bar.spawn(42, 77)
    assert function_call.get() == 7693


baz_app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


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


cls_with_enter_app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


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


@pytest.mark.skip("this doesn't actually work - but issue was hidden by `entered` being an obj property")
@pytest.mark.asyncio
async def test_async_enter_on_local_modal_call():
    obj = ClsWithAsyncEnter()
    assert await obj.modal_method.local(7) == 49
    assert obj.inited
    assert obj.entered


inheritance_app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


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


inheritance_app_2 = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


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
    obj = NoParamsCls()
    assert obj.bar.local(7) == 343

    # Deploy app to get an app id
    app_id = deploy_app(app, "my-cls-app", client=client).app_id

    # Initialize a container
    container_app = RunningApp(app_id)

    # Associate app with app
    app._init_container(client, container_app)

    # Hydration shouldn't overwrite local function definition
    obj = NoParamsCls()
    assert obj.bar.local(7) == 343


app_unhydrated = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


@app_unhydrated.cls()
class FooUnhydrated:
    @method()
    def bar(self, x): ...


def test_unhydrated():
    foo = FooUnhydrated()
    with pytest.raises(ExecutionError, match="hydrated"):
        foo.bar.remote(42)


app_method_args = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


@app_method_args.cls(min_containers=5)
class XYZ:
    @method()
    def foo(self): ...

    @method()
    def bar(self): ...


def test_method_args(servicer, client):
    with app_method_args.run(client=client):
        funcs = servicer.app_functions.values()
        assert {f.function_name for f in funcs} == {"XYZ.*"}
        warm_pools = {f.function_name: f.autoscaler_settings.min_containers for f in funcs}
        assert warm_pools == {"XYZ.*": 5}


def test_cls_keep_warm(client, servicer):
    app = App()

    @app.cls(serialized=True)
    class ClsWithMethod:
        def __init__(self, arg=None):
            self.arg = arg

        @method()
        def bar(self): ...

    with app.run(client=client):
        assert len(servicer.app_functions) == 1  # only class service function
        cls_service_fun = servicer.function_by_name("ClsWithMethod.*")
        assert cls_service_fun.is_class
        assert cls_service_fun.warm_pool_size == 0

        ClsWithMethod().keep_warm(2)  # type: ignore  # Python can't do type intersection
        assert cls_service_fun.warm_pool_size == 2

        ClsWithMethod("other-instance").keep_warm(5)  # type: ignore  # Python can't do type intersection
        instance_service_function = servicer.function_by_name("ClsWithMethod.*", params=((("other-instance",), {})))
        assert len(servicer.app_functions) == 2  # + instance service function
        assert cls_service_fun.warm_pool_size == 2
        assert instance_service_function.warm_pool_size == 5


with pytest.warns(DeprecationError, match="@modal.build"):

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
    pfs: dict[str, _PartialFunction]

    pfs = _find_partial_methods_for_user_cls(ClsWithHandlers, _PartialFunctionFlags.BUILD)
    assert list(pfs.keys()) == ["my_build", "my_build_and_enter"]

    pfs = _find_partial_methods_for_user_cls(ClsWithHandlers, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT)
    assert list(pfs.keys()) == ["my_memory_snapshot"]

    pfs = _find_partial_methods_for_user_cls(ClsWithHandlers, _PartialFunctionFlags.ENTER_POST_SNAPSHOT)
    assert list(pfs.keys()) == ["my_enter", "my_build_and_enter"]

    pfs = _find_partial_methods_for_user_cls(ClsWithHandlers, _PartialFunctionFlags.EXIT)
    assert list(pfs.keys()) == ["my_exit"]


web_app_app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


@web_app_app.cls()
class WebCls:
    @fastapi_endpoint()
    def endpoint(self):
        pass

    @asgi_app()
    def asgi(self):
        pass


def test_web_cls(client):
    with web_app_app.run(client=client):
        c = WebCls()
        assert c.endpoint.web_url == "http://endpoint.internal"
        assert c.asgi.web_url == "http://asgi.internal"


handler_app = App("handler-app", include_source=True)


image = Image.debian_slim().pip_install("xyz")


with pytest.warns(DeprecationError, match="@modal.build"):

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
        service_function = servicer.function_by_name("ClsWithBuild.*")
        # The function image should have added a new layer with original image as the parent
        f_image = servicer.images[service_function.image_id]
        assert f_image.base_images[0].image_id == image.object_id
        assert servicer.force_built_images == []


other_handler_app = App("other-handler-app", include_source=True)


with pytest.warns(DeprecationError, match="@modal.build"):

    @other_handler_app.cls(image=image)
    class ClsWithForceBuild:
        @build(force=True)
        def build(self):
            pass

        @method()
        def method(self):
            pass


def test_force_build_image(client, servicer):
    with other_handler_app.run(client=client):
        service_function = servicer.function_by_name("ClsWithForceBuild.*")
        # The function image should have added a new layer with original image as the parent
        f_image = servicer.images[service_function.image_id]
        assert f_image.base_images[0].image_id == image.object_id
        assert servicer.force_built_images == ["im-3"]


build_timeout_handler_app = App("build-timeout-handler-app", include_source=True)


with pytest.warns(DeprecationError, match="@modal.build"):

    @build_timeout_handler_app.cls(image=image)
    class ClsWithBuildTimeout:
        @build(timeout=123)
        def timeout_build(self):
            pass

        @build()
        def default_timeout_build(self):
            pass

        @method()
        def method(self):
            pass


def test_build_timeout_image(client, servicer):
    with build_timeout_handler_app.run(client=client):
        service_function = servicer.function_by_name("ClsWithBuildTimeout.timeout_build")
        assert service_function.timeout_secs == 123

        service_function = servicer.function_by_name("ClsWithBuildTimeout.default_timeout_build")
        assert service_function.timeout_secs == 86400


@pytest.mark.parametrize("decorator", [enter, exit])
def test_disallow_lifecycle_decorators_with_method(decorator):
    name = decorator.__name__.split("_")[-1]  # remove synchronicity prefix
    with pytest.raises(InvalidError, match=f"Cannot use `@{name}` decorator with `@method`."):

        class ClsDecoratorMethodStack:
            @decorator()
            @method()
            def f(self):
                pass


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

        @modal.fastapi_endpoint()
        def web(self):
            pass

    assert isinstance(Foo.bar, PartialFunction)

    assert Foo().bar() == "a"  # type: ignore   # edge case - using a non-decorated class should just return the bound original method
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
    web_partial_function: _PartialFunction = synchronizer._translate_in(revived_class.web)  # type: ignore
    assert web_partial_function.webhook_config
    assert web_partial_function.webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION


def test_cross_process_userclass_serde(supports_dir):
    res = subprocess.check_output([sys.executable, supports_dir / "serialize_class.py"])
    assert len(res) < 2000  # should be ~1300 bytes as of 2024-06-05
    revived_cls = deserialize(res, None)
    method_without_descriptor_protocol = revived_cls.__dict__["method"]
    assert isinstance(method_without_descriptor_protocol, modal.partial_function.PartialFunction)
    assert revived_cls().method() == "a"  # this should be bound to the object


app2 = App("app2", include_source=True)


@app2.cls()
class UsingAnnotationParameters:
    a: int = modal.parameter()
    b: str = modal.parameter(default="hello")
    c: float = modal.parameter(init=False)

    @method()
    def get_value(self):
        return self.a


init_side_effects = []


@app2.cls()
class UsingCustomConstructor:
    # might want to deprecate this soon
    a: int

    def __init__(self, a: int):
        self._a = a
        init_side_effects.append("did_run")

    @method()
    def get_value(self):
        return self._a


def test_implicit_constructor(client, set_env_client):
    c = UsingAnnotationParameters(a=10)

    assert c.a == 10
    assert c.get_value.local() == 10
    assert c.b == "hello"

    d = UsingAnnotationParameters(a=11, b="goodbye")
    assert d.b == "goodbye"

    with pytest.raises(ValueError, match="Missing required parameter: a"):
        with app2.run(client=client):
            UsingAnnotationParameters().get_value.remote()  # type: ignore

    # check that implicit constructors trigger strict parametrization
    function_info: FunctionInfo = synchronizer._translate_in(UsingAnnotationParameters)._class_service_function._info  # type: ignore
    assert function_info.class_parameter_info().format == api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_PROTO


def test_custom_constructor():
    d = UsingCustomConstructor(10)
    assert not init_side_effects

    assert d._a == 10  # lazily run constructor when accessing non-method attributes (!)
    assert init_side_effects == ["did_run"]

    d2 = UsingCustomConstructor(11)
    assert d2.get_value.local() == 11  # run constructor before running locally
    # check that explicit constructors trigger pickle parametrization
    function_info: FunctionInfo = synchronizer._translate_in(UsingCustomConstructor)._class_service_function._info  # type: ignore
    assert function_info.class_parameter_info().format == api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_PICKLE


class ParametrizedClass1:
    def __init__(self, a):
        pass


class ParametrizedClass1Implicit:
    a: int = modal.parameter()


class ParametrizedClass2:
    def __init__(self, a: int = 1):
        pass


class ParametrizedClass2Implicit:
    a: int = modal.parameter(default=1)


class ParametrizedClass3:
    def __init__(self):
        pass


app_batched = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


def test_batched_method_duplicate_error(client):
    with pytest.raises(
        InvalidError, match="Modal class BatchedClass_1 with a modal batched function cannot have other modal methods."
    ):

        @app_batched.cls(serialized=True)
        class BatchedClass_1:
            @modal.method()
            def method(self):
                pass

            @modal.batched(max_batch_size=2, wait_ms=0)
            def batched_method(self):
                pass

    with pytest.raises(InvalidError, match="Modal class BatchedClass_2 can only have one batched function."):

        @app_batched.cls(serialized=True)
        class BatchedClass_2:
            @modal.batched(max_batch_size=2, wait_ms=0)
            def batched_method_1(self):
                pass

            @modal.batched(max_batch_size=2, wait_ms=0)
            def batched_method_2(self):
                pass


def test_cls_with_both_constructor_and_parameters_is_invalid():
    with pytest.raises(InvalidError, match="constructor"):

        @app.cls(serialized=True)
        class A:
            a: int = modal.parameter()

            def __init__(self, a):
                self.a = a


def test_unannotated_parameters_are_invalid():
    with pytest.raises(InvalidError, match="annotated"):

        @app.cls(serialized=True)
        class B:
            b = modal.parameter()  # type: ignore


def test_unsupported_type_parameters_raise_errors():
    with pytest.raises(InvalidError, match="float"):

        @app.cls(serialized=True)
        class C:
            c: float = modal.parameter()


def test_unsupported_function_decorators_on_methods():
    with pytest.raises(InvalidError, match="cannot be used on class methods"):

        @app.cls(serialized=True)
        class M:
            @app.function(serialized=True)
            @modal.fastapi_endpoint()
            def f(self):
                pass

    with pytest.raises(InvalidError, match="cannot be used on class methods"):

        @app.cls(serialized=True)
        class D:
            @app.function(serialized=True)
            def f(self):
                pass


def test_modal_object_param_uses_wrapped_type(servicer, set_env_client, client):
    with servicer.intercept() as ctx:
        with modal.Dict.ephemeral() as dct:
            with baz_app.run():
                # create bound instance:
                typing.cast(modal.Cls, Baz(x=dct)).keep_warm(1)

    req: api_pb2.FunctionBindParamsRequest = ctx.pop_request("FunctionBindParams")
    function_def: api_pb2.Function = servicer.app_functions[req.function_id]

    _client = typing.cast(modal.client._Client, synchronizer._translate_in(client))
    container_params = deserialize_params(req.serialized_params, function_def, _client)
    args, kwargs = container_params
    assert type(kwargs["x"]) is type(dct)


def test_using_method_on_uninstantiated_cls(recwarn, disable_auto_mount):
    app = App()

    @app.cls(serialized=True)
    class C:
        @method()
        def method(self):
            pass

    assert len(recwarn) == 0
    with pytest.raises(AttributeError):
        C.blah  # type: ignore   # noqa
    assert len(recwarn) == 0

    assert isinstance(C().method, Function)  # should be fine to access on an instance of the class
    assert len(recwarn) == 0

    # The following should warn since it's accessed on the class directly
    C.method  # noqa  # triggers a deprecation warning
    # TODO: this will be an AttributeError or return a non-modal unbound function in the future:
    assert len(recwarn) == 1
    warning_string = str(recwarn[0].message)
    assert "instantiate classes before using methods" in warning_string
    assert "C().method instead of C.method" in warning_string


def test_method_on_cls_access_warns():
    with pytest.warns(match="instantiate classes before using methods"):
        print(Foo.bar)
