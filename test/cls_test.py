# Copyright Modal Labs 2022
import dataclasses
import inspect
import pytest
import subprocess
import sys
import threading
import typing
from typing import TYPE_CHECKING

from typing_extensions import assert_type

import modal.experimental
import modal.partial_function
from modal import App, Cls, Function, Image, Volume, enter, exit, method
from modal._partial_function import (
    _find_partial_methods_for_user_cls,
    _PartialFunction,
    _PartialFunctionFlags,
)
from modal._serialization import deserialize, deserialize_params, serialize
from modal._utils.async_utils import synchronizer
from modal._utils.function_utils import FunctionInfo
from modal.cls import _ServiceOptions
from modal.exception import DeprecationError, ExecutionError, InvalidError, NotFoundError
from modal.partial_function import (
    PartialFunction,
    asgi_app,
    fastapi_endpoint,
    web_server,
)
from modal.runner import deploy_app
from modal.running_app import RunningApp
from modal_proto import api_pb2

from .supports.base_class import BaseCls2

app = App("app")


@app.cls()
class NoParamsCls:
    @method()
    def bar(self, x):
        return x**3

    @method()
    def baz(self, x):
        return x**2


@app.cls(cpu=42, _experimental_restrict_output=True)
class Foo:
    @method()
    def bar(self, x: int) -> float:
        return x**3

    @method()
    def baz(self, y: int) -> float:
        return y**4

    @web_server(8080)
    def web(self):
        pass


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
            function_schema=api_pb2.FunctionSchema(
                schema_type=api_pb2.FunctionSchema.FunctionSchemaType.FUNCTION_SCHEMA_V1,
                arguments=[
                    api_pb2.ClassParameterSpec(
                        name="x",
                        full_type=api_pb2.GenericPayloadType(
                            base_type=api_pb2.ParameterType.PARAM_TYPE_INT,
                        ),
                    )
                ],
                return_type=api_pb2.GenericPayloadType(
                    base_type=api_pb2.ParameterType.PARAM_TYPE_UNKNOWN,
                ),
            ),
            supported_input_formats=[api_pb2.DATA_FORMAT_PICKLE, api_pb2.DATA_FORMAT_CBOR],
            supported_output_formats=[api_pb2.DATA_FORMAT_CBOR],
        ),
        "baz": api_pb2.MethodDefinition(
            function_name="Foo.baz",
            function_type=api_pb2.Function.FunctionType.FUNCTION_TYPE_FUNCTION,
            function_schema=api_pb2.FunctionSchema(
                schema_type=api_pb2.FunctionSchema.FunctionSchemaType.FUNCTION_SCHEMA_V1,
                arguments=[
                    api_pb2.ClassParameterSpec(
                        name="y",
                        full_type=api_pb2.GenericPayloadType(
                            base_type=api_pb2.ParameterType.PARAM_TYPE_INT,
                        ),
                    )
                ],
                return_type=api_pb2.GenericPayloadType(
                    base_type=api_pb2.ParameterType.PARAM_TYPE_UNKNOWN,
                ),
            ),
            supported_input_formats=[api_pb2.DATA_FORMAT_PICKLE, api_pb2.DATA_FORMAT_CBOR],
            supported_output_formats=[api_pb2.DATA_FORMAT_CBOR],
        ),
        "web": api_pb2.MethodDefinition(
            function_name="Foo.web",
            function_type=api_pb2.Function.FunctionType.FUNCTION_TYPE_FUNCTION,
            webhook_config=api_pb2.WebhookConfig(
                type=api_pb2.WEBHOOK_TYPE_WEB_SERVER,
                async_mode=api_pb2.WEBHOOK_ASYNC_MODE_AUTO,
                web_server_port=8080,
                web_server_startup_timeout=5,
            ),
            supported_input_formats=[api_pb2.DATA_FORMAT_ASGI],
            supported_output_formats=[api_pb2.DATA_FORMAT_ASGI, api_pb2.DATA_FORMAT_GENERATOR_DONE],
            web_url="http://web.internal",
        ),
    }


def test_call_class_sync(client, servicer):
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
    unhydrated_aws_secret = modal.Secret.from_dict(
        {"AWS_ACCESS_KEY_ID": "my-key", "AWS_SECRET_ACCESS_KEY": "my-secret"}
    )
    cloud_bucket_mount = modal.CloudBucketMount(bucket_name="s3-bucket-name", secret=unhydrated_aws_secret)
    with servicer.intercept() as ctx:
        foo = Foo.with_options(  # type: ignore
            cpu=48,
            retries=5,
            volumes={"/vol": unhydrated_volume, "/cloud_mnt": cloud_bucket_mount},
            secrets=[unhydrated_secret],
            region="us-east-1",
            cloud="aws",
        )()
        assert len(ctx.calls) == 0  # no rpcs in with_options

    with app.run(client=client):
        with servicer.intercept() as ctx:
            res = foo.bar.remote(2)
            function_bind_params: api_pb2.FunctionBindParamsRequest
            (function_bind_params,) = ctx.get_requests("FunctionBindParams")
            assert function_bind_params.function_options.retry_policy.retries == 5
            assert function_bind_params.function_options.resources.milli_cpu == 48000
            assert function_bind_params.function_options.scheduler_placement.regions == ["us-east-1"]
            assert function_bind_params.function_options.cloud_provider_str == "aws"
            assert function_bind_params.function_options.replace_cloud_bucket_mounts
            assert function_bind_params.function_options.replace_secret_ids
            cloud_bucket_mounts = function_bind_params.function_options.cloud_bucket_mounts
            assert len(cloud_bucket_mounts) == 1
            assert cloud_bucket_mounts[0].mount_path == "/cloud_mnt"
            assert cloud_bucket_mounts[0].bucket_name == "s3-bucket-name"

            assert len(ctx.get_requests("VolumeGetOrCreate")) == 1
            assert len(ctx.get_requests("SecretGetOrCreate")) == 2

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


def test_class_multiple_dynamic_parameterization_methods(client, servicer):
    foo = (
        Foo.with_options(max_containers=1)  # type: ignore
        .with_batching(max_batch_size=10, wait_ms=10)  # type: ignore
        .with_concurrency(max_inputs=100)()  # type: ignore
    )

    with app.run(client=client):
        with servicer.intercept() as ctx:
            res = foo.bar.remote(2)
            function_bind_params: api_pb2.FunctionBindParamsRequest
            (function_bind_params,) = ctx.get_requests("FunctionBindParams")
            assert function_bind_params.function_options.concurrency_limit == 1
            assert function_bind_params.function_options.batch_max_size == 10
            assert function_bind_params.function_options.batch_linger_ms == 10
            assert function_bind_params.function_options.max_concurrent_inputs == 100
    assert res == 4


@pytest.mark.parametrize("read_only", [True, False])
def test_class_multiple_with_options_calls(client, servicer, read_only):
    weights_volume = Volume.from_name("weights", create_if_missing=True)

    if read_only:
        weights_volume = weights_volume.read_only()
    foo = (
        Foo.with_options(  # type: ignore
            gpu="A10:4",
            memory=1024,
            cpu=8,
            buffer_containers=2,
            max_containers=5,
            volumes={"/data": Volume.from_name("data", create_if_missing=True)},
        ).with_options(  # type: ignore
            gpu="A100",
            memory=2048,
            max_containers=10,
            volumes={"/weights": weights_volume},
        )()  # type: ignore
    )

    with app.run(client=client):
        with servicer.intercept() as ctx:
            _ = foo.bar.remote(2)
            function_bind_params: api_pb2.FunctionBindParamsRequest
            (function_bind_params,) = ctx.get_requests("FunctionBindParams")
            assert function_bind_params.function_options.resources.milli_cpu == 8000
            assert function_bind_params.function_options.resources.memory_mb == 2048
            assert function_bind_params.function_options.resources.gpu_config.gpu_type == "A100"
            assert function_bind_params.function_options.resources.gpu_config.count == 1
            assert function_bind_params.function_options.buffer_containers == 2
            assert function_bind_params.function_options.concurrency_limit == 10
            assert len(function_bind_params.function_options.volume_mounts) == 1
            assert function_bind_params.function_options.volume_mounts[0].mount_path == "/weights"
            assert function_bind_params.function_options.volume_mounts[0].read_only == read_only


def test_with_options_from_name(servicer, client):
    unhydrated_volume = modal.Volume.from_name("some_volume", create_if_missing=True)
    unhydrated_secret = modal.Secret.from_dict({"foo": "bar"})

    with servicer.intercept() as ctx:
        SomeClass = modal.Cls.from_name("some_app", "SomeClass", client=client)
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


def test_with_options_prehydrated_payload(client, servicer):
    with servicer.intercept() as ctx, app.run(client=client):
        secret = modal.Secret.from_dict({"foo": "bar"}).hydrate(client=client)
        secret_id = secret.object_id
        foo = Foo.with_options(secrets=[secret])()  # type: ignore  # cls type shadowing :(
        foo.bar.remote(2)

    function_bind_params: api_pb2.FunctionBindParamsRequest
    (function_bind_params,) = ctx.get_requests("FunctionBindParams")
    assert function_bind_params.function_options.secret_ids == [secret_id]


def test_service_options_defaults_untruthiness():
    # For `.with_options()` stacking (method-chaining) to work, the default values of the
    # internal _ServiceOptions dataclass should be be untruthy. This test just asserts that.
    # In the future we may change the implementation to use an "Unset" sentinel default, in
    # which case we wouldn't need this assertion.
    default_options = _ServiceOptions()
    for value in dataclasses.asdict(default_options).values():  # type: ignore  # synchronicity type stubs
        assert not value


# Reusing the app runs into an issue with stale function handles.
# TODO (akshat): have all the client tests use separate apps, and throw
# an exception if the user tries to reuse an app.
app_remote = App()


@app_remote.cls(cpu=42)
class FooRemote:
    x: int = modal.parameter()
    y: str = modal.parameter()

    @method()
    def bar(self, z: int):
        return z**3


def test_call_cls_remote_sync(client):
    with app_remote.run(client=client):
        foo_remote: FooRemote = FooRemote(x=3, y="hello")
        ret: float = foo_remote.bar.remote(8)
        assert ret == 64  # Mock servicer just squares the argument


def test_call_cls_remote_invalid_type(client):
    with app_remote.run(client=client):

        def my_function():
            print("Hello, world!")

        with pytest.raises(ValueError) as excinfo:
            FooRemote(x=42, y=my_function)  # type: ignore

        exc = excinfo.value
        assert "function" in str(exc)


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
    class_function = servicer.function_by_name("FooSer.*")
    assert class_function.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED
    user_cls = deserialize(class_function.class_serialized, client)

    # Create bound method
    obj = user_cls()
    bound_bar = user_cls.bar.__get__(obj)
    # Make sure it's callable
    assert bound_bar(100) == 1000000


app_remote_2 = App()


@app_remote_2.cls(cpu=42)
class BarRemote:
    x: int = modal.parameter()
    y: str = modal.parameter()

    @method()
    def baz(self, z: int):
        return z**3


@pytest.mark.asyncio
async def test_call_cls_remote_async(client):
    async with app_remote_2.run(client=client):
        bar_remote = BarRemote(x=3, y="hello")
        assert await bar_remote.baz.remote.aio(8) == 64  # Mock servicer just squares the argument


app_local = App()


@app_local.cls(cpu=42, enable_memory_snapshot=True)
class FooLocal:
    @property
    def side_effects(self):
        return self.__dict__.setdefault("_side_effects", ["__init__"])

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


def test_from_name_lazy_method_hydration(client, servicer):
    deploy_app(app, "my-cls-app", client=client)
    cls: Cls = Cls.from_name("my-cls-app", "Foo", client=client)

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


def test_lookup_lazy_remote(client):
    # See #972 (PR) and #985 (revert PR): adding unit test to catch regression
    deploy_app(app, "my-cls-app", client=client)
    cls: Cls = Cls.from_name("my-cls-app", "Foo", client=client)
    obj = cls("foo", 234)
    assert obj.bar.remote(42, 77) == 7693


def test_lookup_lazy_remote_legacy_syntax(client):
    # See #972 (PR) and #985 (revert PR): adding unit test to catch regression
    deploy_app(app, "my-cls-app", client=client)
    cls: Cls = Cls.from_name("my-cls-app", "Foo", client=client)
    obj = cls("foo", 234)
    assert obj.bar.remote(42, 77) == 7693


def test_lookup_lazy_spawn(client):
    # See #1071
    deploy_app(app, "my-cls-app", client=client)
    cls: Cls = Cls.from_name("my-cls-app", "Foo", client=client)
    obj = cls("foo", 234)
    function_call = obj.bar.spawn(42, 77)
    assert function_call.get() == 7693


def test_failed_lookup_error(client):
    with pytest.raises(NotFoundError, match="Lookup failed for Cls 'Foo' from the 'my-cls-app' app"):
        Cls.from_name("my-cls-app", "Foo", client=client).hydrate()

    with pytest.raises(NotFoundError, match="in the 'some-env' environment"):
        Cls.from_name("my-cls-app", "Foo", environment_name="some-env", client=client).hydrate()


baz_app = App()


@baz_app.cls()
class Baz:
    x: int = modal.parameter()

    def not_modal_method(self, y: int) -> int:
        return self.x * y


def test_call_not_modal_method():
    baz: Baz = Baz(x=5)
    assert baz.x == 5
    assert baz.not_modal_method(7) == 35


cls_with_enter_app = App()


def get_thread_id():
    return threading.current_thread().name


@cls_with_enter_app.cls()
class ClsWithEnter:
    local_thread_id: str = modal.parameter()
    entered: bool = modal.parameter(default=False)

    @enter()
    def enter(self):
        self.entered = True
        assert get_thread_id() == self.local_thread_id

    def not_modal_method(self, y: int) -> int:
        return y**2

    @method()
    def modal_method(self, y: int) -> int:
        return y**2


def test_dont_enter_on_local_access():
    obj = ClsWithEnter(local_thread_id=get_thread_id())
    with pytest.raises(AttributeError):
        obj.doesnt_exist  # type: ignore
    assert obj.local_thread_id == get_thread_id()
    assert not obj.entered


def test_dont_enter_on_local_non_modal_call():
    obj = ClsWithEnter(local_thread_id=get_thread_id())
    assert obj.not_modal_method(7) == 49
    assert obj.local_thread_id == get_thread_id()
    assert not obj.entered


def test_enter_on_local_modal_call():
    obj = ClsWithEnter(local_thread_id=get_thread_id())
    assert obj.modal_method.local(7) == 49
    assert obj.local_thread_id == get_thread_id()
    assert obj.entered


@cls_with_enter_app.cls()
class ClsWithAsyncEnter:
    inited: bool = modal.parameter(default=False)
    entered = False  # non parameter

    @enter()
    async def enter(self):
        self.entered = True

    @method()
    async def modal_method(self, y: int) -> int:
        return y**2


@pytest.mark.skip("this doesn't actually work - but issue was hidden by `entered` being an obj property")
@pytest.mark.asyncio
async def test_async_enter_on_local_modal_call():
    obj = ClsWithAsyncEnter(inited=True)
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


app_unhydrated = App()


@app_unhydrated.cls()
class FooUnhydrated:
    @method()
    def bar(self, x): ...


def test_unhydrated(set_env_client):
    # TODO: get rid of set_env_client here.
    #  It's needed since the Resolver will
    #  currently try to infer a Client here before it gets to the load code that
    #  raises the exception (which happens because the client has not been added
    #  to the load context by the app "run" in this case)
    #  The crux is that a Method Function is *conditionally* lazily loadable
    #  depending on if the root Cls is defined through from_name() or via @app.cls()
    #  and in the latter case if the app is already running...
    #  We would need something like the resolver checking that all dependencies of
    #  a lazy object if they are either hydrated or lazily loadable to determine
    #  if we should attempt lazy loads
    foo = FooUnhydrated()
    with pytest.raises(ExecutionError, match="hydrated"):
        foo.bar.remote(42)


app_method_args = App()


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


def test_cls_update_autoscaler(client, servicer):
    app = App()

    @app.cls(serialized=True)
    class ClsWithMethod:
        arg: str = modal.parameter(default="")

        @method()
        def bar(self): ...

    with app.run(client=client):
        assert len(servicer.app_functions) == 1  # only class service function
        cls_service_fun = servicer.function_by_name("ClsWithMethod.*")
        assert cls_service_fun.is_class
        assert cls_service_fun.warm_pool_size == 0

        empty_args_obj = typing.cast(modal.cls.Obj, ClsWithMethod())
        empty_args_obj.update_autoscaler(min_containers=2, buffer_containers=1)
        service_function_id = empty_args_obj._cached_service_function().object_id
        service_function_defn = servicer.app_functions[service_function_id]
        autoscaler_settings = service_function_defn.autoscaler_settings
        assert service_function_defn.warm_pool_size == autoscaler_settings.min_containers == 2
        assert service_function_defn._experimental_buffer_containers == autoscaler_settings.buffer_containers == 1

        param_obj = ClsWithMethod(arg="other-instance")
        param_obj.update_autoscaler(min_containers=5, max_containers=10)  # type: ignore
        assert len(servicer.app_functions) == 3  # base + 2 x instance service function
        assert cls_service_fun.warm_pool_size == 0  # base still has no warm

        instance_service_function_id = param_obj._cached_service_function().object_id  # type: ignore
        instance_service_defn = servicer.app_functions[instance_service_function_id]
        instance_autoscaler_settings = instance_service_defn.autoscaler_settings
        assert instance_service_defn.warm_pool_size == instance_autoscaler_settings.min_containers == 5
        assert instance_service_defn.concurrency_limit == instance_autoscaler_settings.max_containers == 10


def test_cls_lookup_update_autoscaler(client, servicer, set_env_client):
    # TODO: get rid of set_env_client, see `test_unhydrated`
    app = App(name := "my-cls-app")

    @app.cls(serialized=True)
    class ClsWithMethod:
        arg: str = modal.parameter(default="")

        @method()
        def bar(self): ...

    C_pre_deploy = ClsWithMethod()
    with pytest.raises(ExecutionError, match="has not been hydrated"):
        C_pre_deploy.update_autoscaler(min_containers=1)  # type: ignore

    deploy_app(app, name, client=client)

    C = Cls.from_name(name, "ClsWithMethod", client=client)
    obj = C()
    obj.update_autoscaler(min_containers=3)

    service_function_id = obj._cached_service_function().object_id
    assert servicer.app_functions[service_function_id].warm_pool_size == 3

    with servicer.intercept() as ctx:
        obj.update_autoscaler(min_containers=4)
        assert len(ctx.get_requests("FunctionBindParams")) == 0  # We did not re-bind


class ClsWithHandlers:
    @enter(snap=True)
    def my_memory_snapshot(self):
        pass

    @enter()
    def my_enter(self):
        pass

    @exit()
    def my_exit(self):
        pass


def test_handlers():
    pfs: dict[str, _PartialFunction]

    pfs = _find_partial_methods_for_user_cls(ClsWithHandlers, _PartialFunctionFlags.ENTER_PRE_SNAPSHOT)
    assert list(pfs.keys()) == ["my_memory_snapshot"]

    pfs = _find_partial_methods_for_user_cls(ClsWithHandlers, _PartialFunctionFlags.ENTER_POST_SNAPSHOT)
    assert list(pfs.keys()) == ["my_enter"]

    pfs = _find_partial_methods_for_user_cls(ClsWithHandlers, _PartialFunctionFlags.EXIT)
    assert list(pfs.keys()) == ["my_exit"]


web_app_app = App()


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
        assert c.endpoint.get_web_url() == "http://endpoint.internal"
        assert c.asgi.get_web_url() == "http://asgi.internal"


handler_app = App("handler-app")


image = Image.debian_slim().pip_install("xyz")


other_handler_app = App("other-handler-app")


@pytest.mark.parametrize("decorator", [enter, exit])
def test_disallow_lifecycle_decorators_with_method(decorator):
    with pytest.raises(InvalidError, match="cannot be combined with lifecycle decorators"):

        class HasLifecycleOnMethod:
            @decorator()
            @method()
            def f(self):
                pass

    with pytest.raises(InvalidError, match="cannot be combined with lifecycle decorators"):

        class HasMethodOnLifecycle:
            @method()
            @decorator()
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
    assert web_partial_function.params.webhook_config
    assert web_partial_function.params.webhook_config.type == api_pb2.WEBHOOK_TYPE_FUNCTION


def test_cross_process_userclass_serde(supports_dir):
    res = subprocess.check_output([sys.executable, supports_dir / "serialize_class.py"])
    assert len(res) < 2000  # should be ~1300 bytes as of 2024-06-05
    revived_cls = deserialize(res, None)
    method_without_descriptor_protocol = revived_cls.__dict__["method"]
    assert isinstance(method_without_descriptor_protocol, modal.partial_function.PartialFunction)
    assert revived_cls().method() == "a"  # this should be bound to the object


app2 = App("app2")


@app2.cls()
class UsingAnnotationParameters:
    a: int = modal.parameter()
    b: str = modal.parameter(default="hello")
    c: float = modal.parameter(init=False)
    d: bytes = modal.parameter(default=b"world")

    @method()
    def get_value(self):
        return self.a


init_side_effects = []


def test_implicit_constructor(client):
    c = UsingAnnotationParameters(a=10)

    assert c.a == 10
    assert c.get_value.local() == 10
    assert c.b == "hello"
    assert c.d == b"world"

    d = UsingAnnotationParameters(a=11, b="goodbye", d=b"bye")
    assert d.b == "goodbye"
    assert d.d == b"bye"

    with pytest.raises(InvalidError, match="Missing required parameter: a"):
        with app2.run(client=client):
            UsingAnnotationParameters().get_value.remote()  # type: ignore

    # check that implicit constructors trigger strict parametrization
    function_info: FunctionInfo = synchronizer._translate_in(UsingAnnotationParameters)._class_service_function._info  # type: ignore
    assert function_info.class_parameter_info().format == api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_PROTO


def test_custom_constructor_has_deprecation_warning():
    with pytest.warns(DeprecationError, match="non-default constructor"):

        @app2.cls(serialized=True)
        class UsingCustomConstructor:
            # might want to deprecate this soon
            a: int

            def __init__(self, a: int):
                self._a = a
                init_side_effects.append("did_run")

            @method()
            def get_value(self):
                return self._a

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


app_batched = App()


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
    with pytest.raises(InvalidError, match=r"float is not a supported modal.parameter\(\) type"):

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


def test_using_method_on_uninstantiated_cls():
    app = App()

    @app.cls(serialized=True)
    class C:
        some_non_param_variable = 10

        @method()
        def method(self):
            pass

    assert C.some_non_param_variable == 10

    with pytest.raises(AttributeError, match="blah"):
        C.blah  # type: ignore

    with pytest.raises(AttributeError, match="Did you forget to instantiate the class first?"):
        # The following should error since the class is supposed to be instantiated first
        C.method.remote()  # noqa


def test_using_method_on_uninstantiated_hydrated_cls(client):
    app = App()

    @app.cls(serialized=True)
    class C:
        some_non_param_variable = 10

        @method()
        def method(self):
            pass

    with app.run(client=client):
        assert C.some_non_param_variable == 10

        with pytest.raises(AttributeError, match="blah"):
            C.blah  # type: ignore

        with pytest.raises(AttributeError, match="Did you forget to instantiate the class first?"):
            # The following should error since the class is supposed to be instantiated first
            C.method.remote()  # noqa


def test_using_method_on_uninstantiated_remote_cls(client):
    C = modal.Cls.from_name("app", "C", client=client)

    with pytest.raises(AttributeError, match="Did you forget to instantiate the class first?"):
        # The following should error since the class is supposed to be instantiated first
        C.method.remote()  # noqa


def test_bytes_serialization_validation(servicer, client):
    app = modal.App()

    @app.cls(serialized=True)
    class C:
        foo: bytes = modal.parameter(default=b"foo")

        @method()
        def get_foo(self):
            return self.foo

    with servicer.intercept() as ctx:
        with app.run(client=client):
            with pytest.raises(TypeError, match="Expected bytes"):
                C(foo="this is a string").get_foo.spawn()  # type: ignore   # string should not be allowed, unspecified encoding

            C(foo=b"this is bytes").get_foo.spawn()  # bytes are allowed
            create_function_req: api_pb2.FunctionCreateRequest = ctx.pop_request("FunctionCreate")
            bind_req: api_pb2.FunctionBindParamsRequest = ctx.pop_request("FunctionBindParams")
            args, kwargs = deserialize_params(bind_req.serialized_params, create_function_req.function, client)
            assert kwargs["foo"] == b"this is bytes"

            C().get_foo.spawn()  # omission when using default is allowed
            bind_req: api_pb2.FunctionBindParamsRequest = ctx.pop_request("FunctionBindParams")
            args, kwargs = deserialize_params(bind_req.serialized_params, create_function_req.function, client)
            assert kwargs["foo"] == b"foo"


def test_class_can_not_use_list_parameter(client):
    # we might want to allow lists in the future though...
    app = modal.App()

    with pytest.raises(InvalidError, match="list is not a supported modal.parameter"):

        @app.cls(serialized=True)
        class A:
            p: list[int] = modal.parameter()


def test_class_can_use_073_schema_definition(servicer, client):
    # in ~0.74, we introduced the new full_type type generic that supersedes
    # the .type "flat" type. This tests that lookups on classes deployed with
    # the old proto can still be validated when instantiated.

    with servicer.intercept() as ctx:
        ctx.add_response("ClassGet", api_pb2.ClassGetResponse(class_id="cs-123"))
        ctx.add_response(
            "FunctionGet",
            api_pb2.FunctionGetResponse(
                function_id="fu-123",
                handle_metadata=api_pb2.FunctionHandleMetadata(
                    class_parameter_info=api_pb2.ClassParameterInfo(
                        format=api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_PROTO,
                        schema=[api_pb2.ClassParameterSpec(name="p", type=api_pb2.PARAM_TYPE_STRING)],
                    ),
                    method_handle_metadata={"some_method": api_pb2.FunctionHandleMetadata()},
                ),
            ),
        )
        with pytest.raises(TypeError, match="Expected str, got int"):
            # wrong type for p triggers when .remote goes off
            obj = Cls.from_name("some_app", "SomeCls", client=client)(p=10)
            obj.some_method.remote(1)


def test_class_can_use_future_full_type_only_schema(servicer, client):
    # in ~0.74, we introduced the new full_type type generic that supersedes
    # the .type "flat" type. This tests that the client can use a *future
    # version* that drops support for the .type attribute and only fills the
    # full_type in the schema

    with servicer.intercept() as ctx:
        ctx.add_response("ClassGet", api_pb2.ClassGetResponse(class_id="cs-123"))
        ctx.add_response(
            "FunctionGet",
            api_pb2.FunctionGetResponse(
                function_id="fu-123",
                handle_metadata=api_pb2.FunctionHandleMetadata(
                    class_parameter_info=api_pb2.ClassParameterInfo(
                        format=api_pb2.ClassParameterInfo.PARAM_SERIALIZATION_FORMAT_PROTO,
                        schema=[
                            api_pb2.ClassParameterSpec(
                                name="p", full_type=api_pb2.GenericPayloadType(base_type=api_pb2.PARAM_TYPE_STRING)
                            )
                        ],
                    ),
                    method_handle_metadata={"some_method": api_pb2.FunctionHandleMetadata()},
                ),
            ),
        )
        with pytest.raises(TypeError, match="Expected str, got int"):
            # wrong type for p triggers when .remote goes off
            obj = Cls.from_name("some_app", "SomeCls", client=client)(p=10)
            obj.some_method.remote(1)


def test_concurrent_decorator_on_method_error():
    app = modal.App()

    with pytest.raises(modal.exception.InvalidError, match="decorate the class"):

        @app.cls(serialized=True)
        class UsesConcurrentDecoratoronMethod:
            @modal.concurrent(max_inputs=10)
            def method(self):
                pass


def test_concurrent_decorator_stacked_with_method_decorator():
    app = modal.App()

    with pytest.raises(modal.exception.InvalidError, match="decorate the class"):

        @app.cls(serialized=True)
        class UsesMethodAndConcurrentDecorators:
            @modal.method()
            @modal.concurrent(max_inputs=10)
            def method(self):
                pass


def test_parameter_inheritance(client):
    app = modal.App("inherit-params")

    class Base:
        a: int = modal.parameter()  # parameter in base class

    @app.cls(serialized=True)
    class ConcatenatingParams(Base):
        b: str = modal.parameter()  # add additional parameter

    @app.cls(serialized=True)
    class RepeatingParams(Base):
        # In versions prior to ~1.0.5, base class parameters were not
        # included, so subclasses would always have to repeat parameters
        # from the base class.
        # We allow this as long as the definitions are the same
        a: int = modal.parameter()  # redefine base class parameter
        b: str = modal.parameter()

    @app.cls(serialized=True)
    class ChangingParameterDefinitions(Base):
        # change type of base class parameter, allowed but frowned upon
        a: str = modal.parameter()  # type: ignore  # this isn't allowed by type checkers

    with app.run(client=client):
        # use .update_autoscaler()
        ConcatenatingParams(a=10, b="hello").update_autoscaler()  # type: ignore
        RepeatingParams(a=10, b="hello").update_autoscaler()  # type: ignore
        with pytest.raises(TypeError):
            ChangingParameterDefinitions(a=10).update_autoscaler()  # type: ignore
        ChangingParameterDefinitions(a="10").update_autoscaler()  # type: ignore


def test_cls_namespace_deprecated(servicer):
    # Test from_name with namespace parameter warns
    with pytest.warns(DeprecationError, match="The `namespace` parameter for `modal.Cls.from_name` is deprecated"):
        Cls.from_name("test-app", "test-cls", namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE)

    # Test that from_name without namespace parameter doesn't warn about namespace
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Cls.from_name("test-app", "test-cls")
    # Filter out any unrelated warnings
    namespace_warnings = [w for w in record if "namespace" in str(w.message).lower()]
    assert len(namespace_warnings) == 0


clustered_app = App()


def test_clustered_cls(client, servicer):
    @clustered_app.cls(serialized=True)
    @modal.experimental.clustered(size=3, rdma=True)  # type: ignore
    class ClusteredClass:
        @method()
        def run_task(self, x):
            return x * 2

    @clustered_app.cls(serialized=True)
    class RegularClass:
        @method()
        def regular_method(self, x):
            return x * 4

    with clustered_app.run(client=client):
        assert len(servicer.app_functions) == 2

        class_function = servicer.function_by_name("ClusteredClass.*")
        assert class_function._experimental_group_size == 3
        assert class_function.i6pn_enabled is True  # clustered implies i6pn
        assert class_function.resources.rdma == 1

        obj = ClusteredClass()
        assert hasattr(obj, "run_task")

        regular_function = servicer.function_by_name("RegularClass.*")
        assert regular_function._experimental_group_size == 0  # or not set
        assert regular_function.i6pn_enabled is False
        assert regular_function.resources.rdma == 0


invalid_clustered_app = App()


def test_clustered_cls_with_multiple_methods(client, servicer):
    with pytest.raises(
        InvalidError, match="Modal class ClusteredClassMixed cannot have multiple methods when clustered."
    ):

        @invalid_clustered_app.cls(serialized=True)
        @modal.experimental.clustered(size=2)  # type: ignore
        class ClusteredClassMixed:
            @method()
            def clustered_method(self, x):
                return x * 3

            @method()
            def second_clustered_method(self, x):
                return x * 3


def test_cls_get_flash_url(servicer, client):
    """Test get_flash_url method on Cls.from_name instances"""
    cls = Cls.from_name("dummy-app", "MyClass", client=client)

    with servicer.intercept() as ctx:
        ctx.add_response(
            "ClassGet",
            api_pb2.ClassGetResponse(class_id="cs-1"),
        )
        ctx.add_response(
            "FunctionGet",
            api_pb2.FunctionGetResponse(
                function_id="fu-1",
                handle_metadata=api_pb2.FunctionHandleMetadata(
                    function_name="MyClass.*",
                    is_method=False,
                    _experimental_flash_urls=[
                        "https://flash.example.com/service1",
                        "https://flash.example.com/service2",
                    ],
                ),
            ),
        )
        flash_urls = cls._experimental_get_flash_urls()
        assert flash_urls == ["https://flash.example.com/service1", "https://flash.example.com/service2"]


timeout_app = App("timeout-app")


@timeout_app.cls(startup_timeout=30)
class Timeout:
    @enter()
    def start(self):
        pass

    @method()
    def hello(self):
        pass


def test_startup_timeout(client, servicer):
    with servicer.intercept() as ctx:
        with timeout_app.run(client=client):
            pass

    function_creates_requests: list[api_pb2.FunctionCreateRequest] = ctx.get_requests("FunctionCreate")
    assert len(function_creates_requests) == 1
    function_request = function_creates_requests[0]
    assert function_request.function.startup_timeout_secs == 30


timeout_app_default = App("timeout-app-default")


@timeout_app_default.cls(timeout=20)
class TimeoutDefault:
    @enter()
    def start(self):
        pass

    @method()
    def hello(self):
        pass


def test_startup_timeout_default_copies_timeout(client, servicer):
    with servicer.intercept() as ctx:
        with timeout_app_default.run(client=client):
            pass

    function_creates_requests: list[api_pb2.FunctionCreateRequest] = ctx.get_requests("FunctionCreate")
    assert len(function_creates_requests) == 1
    function_request = function_creates_requests[0]
    assert function_request.function.startup_timeout_secs == 20


def test_cls_load_context_transfers_to_methods():
    C = modal.Cls.from_name("dummy-app", "MyClass")
    c = C(p=1)
    d = C(p=2)
    t = synchronizer._translate_in
    # the *instance* of LoadContext from the Cls should be the same as the child
    expected_load_context = t(C)._load_context  # type: ignore
    assert t(c.some_method)._load_context is expected_load_context  # type: ignore
    assert t(c.some_method)._load_context is t(d.some_method)._load_context  # type: ignore
    assert t(C.with_options(gpu="A100")().some_method)._load_context is expected_load_context  # type: ignore


def test_cls_load_context_transfers_to_methods_local():
    app = modal.App()

    @app.cls(serialized=True)
    class C:
        p: str = modal.parameter()

        @method()
        def some_method(self):
            pass

    t = synchronizer._translate_in
    c = C(p="1")
    assert t(c.some_method)._load_context is t(C)._load_context  # type: ignore

    # the *instance* of LoadContext from the Cls should be the same as the child
    d = C(p="2")
    assert t(c.some_method)._load_context is t(d.some_method)._load_context  # type: ignore
