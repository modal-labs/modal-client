# Copyright Modal Labs 2022
import asyncio
import inspect
import os
import pytest
import time
import typing
from contextlib import contextmanager

from synchronicity.exceptions import UserCodeException

import modal
from modal import App, Image, NetworkFileSystem, Proxy, asgi_app, batched, fastapi_endpoint
from modal._utils.async_utils import synchronize_api
from modal._vendor import cloudpickle
from modal.exception import DeprecationError, ExecutionError, InvalidError
from modal.functions import Function, FunctionCall
from modal.runner import deploy_app
from modal_proto import api_pb2
from test.helpers import deploy_app_externally

app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


if os.environ.get("GITHUB_ACTIONS") == "true":
    TIME_TOLERANCE = 0.25
else:
    TIME_TOLERANCE = 0.05


@app.function()
def foo(p, q):
    return p + q + 11  # not actually used in test (servicer returns sum of square of all args)


@app.function()
async def async_foo(p, q):
    return p + q + 12


def dummy():
    pass  # not actually used in test (servicer returns sum of square of all args)


def test_run_function(client, servicer):
    assert len(servicer.cleared_function_calls) == 0
    with app.run(client=client):
        assert foo.remote(2, 4) == 20
        assert len(servicer.cleared_function_calls) == 1


def test_single_input_function_call_uses_single_rpc(client, servicer):
    with app.run(client=client):
        with servicer.intercept() as ctx:
            assert foo.remote(2, 4) == 20
        assert len(ctx.calls) == 2
        (msg1_type, msg1), (msg2_type, msg2) = ctx.calls
        assert msg1_type == "FunctionMap"
        assert msg2_type == "FunctionGetOutputs"


@pytest.mark.asyncio
async def test_call_function_locally(client, servicer):
    assert foo.local(22, 44) == 77  # call it locally
    assert await async_foo.local(22, 44) == 78

    with app.run(client=client):
        assert foo.remote(2, 4) == 20
        assert async_foo.remote(2, 4) == 20
        assert await async_foo.remote.aio(2, 4) == 20


@pytest.mark.parametrize("slow_put_inputs", [False, True])
@pytest.mark.timeout(120)
def test_map(client, servicer, slow_put_inputs):
    servicer.slow_put_inputs = slow_put_inputs

    app = App()
    dummy_modal = app.function()(dummy)

    assert len(servicer.cleared_function_calls) == 0
    with app.run(client=client):
        assert list(dummy_modal.map([5, 2], [4, 3])) == [41, 13]
        assert len(servicer.cleared_function_calls) == 1
        assert set(dummy_modal.map([5, 2], [4, 3], order_outputs=False)) == {13, 41}
        assert len(servicer.cleared_function_calls) == 2


@pytest.mark.asyncio
async def test_map_async_generator(client):
    app = App()
    dummy_modal = app.function()(dummy)

    async def gen_num():
        yield 2
        yield 3

    async with app.run(client=client):
        res = [num async for num in dummy_modal.map.aio(gen_num())]
        assert res == [4, 9]


def _pow2(x: int):
    return x**2


@contextmanager
def synchronicity_loop_delay_tracker():
    done = False

    async def _track_eventloop_blocking():
        max_dur = 0.0
        BLOCK_TIME = 0.01
        while not done:
            t0 = time.perf_counter()
            await asyncio.sleep(BLOCK_TIME)
            max_dur = max(max_dur, time.perf_counter() - t0)
        return max_dur - BLOCK_TIME  # if it takes exactly BLOCK_TIME we would have zero delay

    track_eventloop_blocking = synchronize_api(_track_eventloop_blocking)
    yield track_eventloop_blocking(_future=True)
    done = True


def test_map_blocking_iterator_blocking_synchronicity_loop(client):
    app = App()
    SLEEP_DUR = 0.5

    def blocking_iter():
        yield 1
        time.sleep(SLEEP_DUR)
        yield 2

    pow2 = app.function()(_pow2)

    with app.run(client=client):
        t0 = time.monotonic()
        with synchronicity_loop_delay_tracker() as max_delay:
            for _ in pow2.map(blocking_iter()):
                pass
        dur = time.monotonic() - t0
    assert dur >= SLEEP_DUR
    assert max_delay.result() < TIME_TOLERANCE  # should typically be much smaller than this


@pytest.mark.asyncio
async def test_map_blocking_iterator_blocking_synchronicity_loop_async(client):
    app = App()
    SLEEP_DUR = 0.5

    def blocking_iter():
        yield 1
        time.sleep(SLEEP_DUR)
        yield 2

    pow2 = app.function()(_pow2)

    async with app.run(client=client):
        t0 = time.monotonic()
        with synchronicity_loop_delay_tracker() as max_delay:
            async for _ in pow2.map.aio(blocking_iter()):
                pass
        dur = time.monotonic() - t0
    assert dur >= SLEEP_DUR
    assert max_delay.result() < TIME_TOLERANCE  # should typically be much smaller than this


_side_effect_count = 0


def side_effect(_):
    global _side_effect_count
    _side_effect_count += 1


def test_for_each(client, servicer):
    app = App()
    servicer.function_body(side_effect)
    side_effect_modal = app.function()(side_effect)
    assert _side_effect_count == 0
    with app.run(client=client):
        side_effect_modal.for_each(range(10))

    assert _side_effect_count == 10


def custom_function(x):
    if x % 2 == 0:
        return x


def test_map_none_values(client, servicer):
    app = App()
    servicer.function_body(custom_function)
    custom_function_modal = app.function()(custom_function)

    with app.run(client=client):
        assert list(custom_function_modal.map(range(4))) == [0, None, 2, None]


def test_starmap(client):
    app = App()

    dummy_modal = app.function()(dummy)
    with app.run(client=client):
        assert list(dummy_modal.starmap([[5, 2], [4, 3]])) == [29, 25]


def test_function_memory_request(client):
    app = App()
    app.function(memory=2048)(dummy)


def test_function_memory_limit(client):
    app = App()
    f = app.function(memory=(2048, 4096))(dummy)

    with app.run(client=client):
        f.remote()

    g = app.function(memory=(2048, 2048 - 1))(custom_function)
    with pytest.raises(InvalidError), app.run(client=client):
        g.remote(0)


def test_function_cpu_request(client, servicer):
    app = App()
    f = app.function(cpu=2.0)(dummy)

    with app.run(client=client):
        f.remote()
        assert servicer.app_functions["fu-1"].resources.milli_cpu == 2000
        assert servicer.app_functions["fu-1"].resources.milli_cpu_max == 0
    assert f.spec.cpu == 2.0

    app = App()
    g = app.function(cpu=7)(dummy)

    with app.run(client=client):
        g.remote()
        assert servicer.app_functions["fu-2"].resources.milli_cpu == 7000
        assert servicer.app_functions["fu-2"].resources.milli_cpu_max == 0
    assert g.spec.cpu == 7


def test_function_cpu_limit(client, servicer):
    app = App()
    f = app.function(cpu=(1, 3))(dummy)
    assert f.spec.cpu == (1, 3)

    with app.run(client=client):
        f.remote()
        assert servicer.app_functions["fu-1"].resources.milli_cpu == 1000
        assert servicer.app_functions["fu-1"].resources.milli_cpu_max == 3000

    g = app.function(cpu=(1, 0.5))(custom_function)
    with pytest.raises(InvalidError), app.run(client=client):
        g.remote(0)


def test_function_disk_request(client):
    app = App()
    app.function(ephemeral_disk=1_000_000)(dummy)


def test_scaledown_window_must_be_positive():
    app = App()
    with pytest.raises(InvalidError, match="must be > 0"):
        app.function(scaledown_window=0)(dummy)


def later():
    return "hello"


def test_function_future(client, servicer):
    app = App()

    servicer.function_body(later)
    later_modal = app.function()(later)
    with app.run(client=client):
        future = later_modal.spawn()
        assert isinstance(future, FunctionCall)

        servicer.function_is_running = True
        assert future.object_id == "fc-1"

        with pytest.raises(TimeoutError):
            future.get(0.01)

        servicer.function_is_running = False
        assert future.get(0.01) == "hello"
        assert future.object_id not in servicer.cleared_function_calls

        future = later_modal.spawn()

        servicer.function_is_running = True
        assert future.object_id == "fc-2"

        future.cancel()
        assert "fc-2" in servicer.cancelled_calls

        assert future.object_id not in servicer.cleared_function_calls

        with pytest.raises(Exception, match="Cannot iterate"):
            next(future.get_gen())


@pytest.mark.asyncio
async def test_function_future_async(client, servicer):
    app = App()

    servicer.function_body(later)
    later_modal = app.function()(later)

    async with app.run(client=client):
        future = await later_modal.spawn.aio()
        servicer.function_is_running = True

        with pytest.raises(TimeoutError):
            await future.get.aio(0.01)

        servicer.function_is_running = False
        assert await future.get.aio(0.01) == "hello"
        assert future.object_id not in servicer.cleared_function_calls  # keep results around a bit longer for futures


def later_gen():
    yield "foo"


async def async_later_gen():
    yield "foo"


@pytest.mark.asyncio
async def test_generator(client, servicer):
    app = App()

    later_gen_modal = app.function()(later_gen)

    def dummy():
        yield "bar"
        yield "baz"
        yield "boo"

    servicer.function_body(dummy)

    assert len(servicer.cleared_function_calls) == 0
    with app.run(client=client):
        assert later_gen_modal.is_generator
        res: typing.Generator = later_gen_modal.remote_gen()  # type: ignore
        # Generators fulfil the *iterator protocol*, which requires both these methods.
        # https://docs.python.org/3/library/stdtypes.html#typeiter
        assert hasattr(res, "__iter__")  # strangely inspect.isgenerator returns false
        assert hasattr(res, "__next__")
        assert next(res) == "bar"
        assert list(res) == ["baz", "boo"]
        assert len(servicer.cleared_function_calls) == 1


def test_generator_map_invalid(client, servicer):
    app = App()

    later_gen_modal = app.function()(later_gen)

    def dummy(x):
        yield x

    servicer.function_body(dummy)

    with app.run(client=client):
        with pytest.raises(InvalidError, match="A generator function cannot be called with"):
            # Support for .map() on generators was removed in version 0.57
            for _ in later_gen_modal.map([1, 2, 3]):
                pass

        with pytest.raises(InvalidError, match="A generator function cannot be called with"):
            later_gen_modal.for_each([1, 2, 3])


@pytest.mark.asyncio
async def test_generator_async(client, servicer):
    app = App()

    later_gen_modal = app.function()(async_later_gen)

    async def async_dummy():
        yield "bar"
        yield "baz"

    servicer.function_body(async_dummy)

    assert len(servicer.cleared_function_calls) == 0
    async with app.run(client=client):
        assert later_gen_modal.is_generator
        res = later_gen_modal.remote_gen.aio()
        # Async generators fulfil the *asynchronous iterator protocol*, which requires both these methods.
        # https://peps.python.org/pep-0525/#support-for-asynchronous-iteration-protocol
        assert hasattr(res, "__aiter__")
        assert hasattr(res, "__anext__")
        # TODO(Jonathon): This works outside of testing, but here gives:
        # `TypeError: cannot pickle 'async_generator' object`
        # await res.__anext__() == "bar"
        # assert len(servicer.cleared_function_calls) == 1


@pytest.mark.asyncio
async def test_generator_future(client, servicer):
    app = App()

    servicer.function_body(later_gen)
    later_modal = app.function()(later_gen)
    with app.run(client=client):
        with pytest.raises(DeprecationError):
            later_modal.spawn()


async def slo1(sleep_seconds):
    # need to use async function body in client test to run stuff in parallel
    # but calling interface is still non-asyncio
    await asyncio.sleep(sleep_seconds)
    return sleep_seconds


def test_sync_parallelism(client, servicer):
    app = App()

    servicer.function_body(slo1)
    slo1_modal = app.function()(slo1)
    with app.run(client=client):
        t0 = time.time()
        # NOTE tests breaks in macOS CI if the smaller time is smaller than ~300ms
        res = FunctionCall.gather(slo1_modal.spawn(0.31), slo1_modal.spawn(0.3))
        t1 = time.time()
        assert res == [0.31, 0.3]  # results should be ordered as inputs, not by completion time
        assert t1 - t0 < 0.6  # less than the combined runtime, make sure they run in parallel


def test_proxy(client, servicer):
    app = App()

    app.function(proxy=Proxy.from_name("my-proxy"))(dummy)
    with app.run(client=client):
        pass


class CustomException(Exception):
    pass


def failure():
    raise CustomException("foo!")


def test_function_exception(client, servicer):
    app = App()

    servicer.function_body(failure)
    failure_modal = app.function()(failure)
    with app.run(client=client):
        with pytest.raises(CustomException) as excinfo:
            failure_modal.remote()
        assert "foo!" in str(excinfo.value)


@pytest.mark.asyncio
async def test_function_exception_async(client, servicer):
    app = App()

    servicer.function_body(failure)
    failure_modal = app.function()(failure)
    async with app.run(client=client):
        with pytest.raises(CustomException) as excinfo:
            coro = failure_modal.remote.aio()
            # mostly for mypy, since output could technically be an async generator which
            # isn't awaitable in the same sense
            assert inspect.isawaitable(coro)
            await coro
        assert "foo!" in str(excinfo.value)


def custom_exception_function(x):
    if x == 4:
        raise CustomException("bad")
    return x * x


def test_map_exceptions(client, servicer):
    app = App()

    servicer.function_body(custom_exception_function)
    custom_function_modal = app.function()(custom_exception_function)

    with app.run(client=client):
        assert list(custom_function_modal.map(range(4))) == [0, 1, 4, 9]

        with pytest.raises(CustomException) as excinfo:
            list(custom_function_modal.map(range(6)))
        assert "bad" in str(excinfo.value)

        res = list(custom_function_modal.map(range(6), return_exceptions=True))
        assert res[:4] == [0, 1, 4, 9] and res[5] == 25
        assert type(res[4]) is UserCodeException and "bad" in str(res[4])


def import_failure():
    raise ImportError("attempted relative import with no known parent package")


def test_function_relative_import_hint(client, servicer):
    app = App()

    servicer.function_body(import_failure)
    import_failure_modal = app.function()(import_failure)

    with app.run(client=client):
        with pytest.raises(ImportError) as excinfo:
            import_failure_modal.remote()
        assert "HINT" in str(excinfo.value)


def test_nonglobal_function():
    app = App()

    with pytest.raises(InvalidError) as excinfo:

        @app.function()
        def f():
            pass

    assert "global scope" in str(excinfo.value)


def test_non_global_serialized_function():
    app = App()

    @app.function(serialized=True)
    def f():
        pass


def test_closure_valued_serialized_function(client, servicer):
    app = App()

    def make_function(s):
        @app.function(name=f"ret_{s}", serialized=True)
        def returner():
            return s

    for s in ["foo", "bar"]:
        make_function(s)

    with app.run(client=client):
        pass

    functions = {}
    for func in servicer.app_functions.values():
        functions[func.function_name] = cloudpickle.loads(func.function_serialized)

    assert len(functions) == 2
    assert functions["ret_foo"]() == "foo"
    assert functions["ret_bar"]() == "bar"


def test_new_hydrated_internal(client, servicer):
    obj: FunctionCall[typing.Any] = FunctionCall._new_hydrated("fc-123", client, None)
    assert obj.object_id == "fc-123"


def test_from_id(client, servicer):
    app = App()

    @app.function(serialized=True)
    def foo():
        pass

    deploy_app(app, "dummy", client=client)

    function_id = foo.object_id
    assert function_id

    function_call = foo.spawn()
    assert function_call.object_id
    # Used in a few examples to construct FunctionCall objects
    rehydrated_function_call = FunctionCall.from_id(function_call.object_id, client)
    assert rehydrated_function_call.object_id == function_call.object_id


def test_local_execution_on_web_endpoint(client, servicer):
    app = App()

    @app.function(serialized=True)
    @fastapi_endpoint()
    def foo(x: str):
        return f"{x}!"

    deploy_app(app, "dummy", client=client)

    function_id = foo.object_id
    assert function_id
    assert foo.web_url

    res = foo.local("hello")
    assert res == "hello!"


def test_local_execution_on_asgi_app(client, servicer):
    from fastapi import FastAPI

    app = App()

    @app.function(serialized=True)
    @asgi_app()
    def foo():
        from fastapi import FastAPI

        web_app = FastAPI()

        @web_app.get("/bar")
        def bar(arg="world"):
            return {"hello": arg}

        return web_app

    deploy_app(app, "dummy", client=client)

    function_id = foo.object_id
    assert function_id
    assert foo.web_url

    res = foo.local()
    assert type(res) is FastAPI


@pytest.mark.parametrize("remote_executor", ["remote", "remote_gen", "spawn"])
def test_invalid_remote_executor_on_web_endpoint(client, servicer, remote_executor):
    app = App()

    @app.function(serialized=True)
    @fastapi_endpoint()
    def foo():
        pass

    deploy_app(app, "dummy", client=client)

    function_id = foo.object_id
    assert function_id
    assert foo.web_url

    with pytest.raises(InvalidError) as excinfo:
        f = getattr(foo, remote_executor)
        res = f()
        if inspect.isgenerator(res):
            next(res)

    assert "webhook" in str(excinfo.value) and remote_executor in str(excinfo.value)


@pytest.mark.parametrize("remote_executor", ["remote", "remote_gen", "spawn"])
def test_invalid_remote_executor_on_asgi_app(client, servicer, remote_executor):
    app = App()

    @app.function(serialized=True)
    @asgi_app()
    def foo():
        from fastapi import FastAPI

        web_app = FastAPI()

        @web_app.get("/foo")
        def foo(arg="world"):
            return {"hello": arg}

        return web_app

    deploy_app(app, "dummy", client=client)

    function_id = foo.object_id
    assert function_id
    assert foo.web_url

    with pytest.raises(InvalidError) as excinfo:
        f = getattr(foo, remote_executor)
        res = f()
        if inspect.isgenerator(res):
            next(res)

    assert "webhook" in str(excinfo.value) and remote_executor in str(excinfo.value)


@pytest.mark.parametrize("is_generator", [False, True])
def test_from_id_iter_gen(client, servicer, is_generator):
    app = App()

    f = later_gen if is_generator else later

    servicer.function_body(f)
    later_modal = app.function()(f)
    with app.run(client=client):
        if is_generator:
            with pytest.raises(DeprecationError):
                later_modal.spawn()
            return

        future = later_modal.spawn()
        assert isinstance(future, FunctionCall)

    assert future.object_id
    rehydrated_function_call = FunctionCall.from_id(future.object_id, client, is_generator=is_generator)
    assert rehydrated_function_call.object_id == future.object_id

    if is_generator:
        assert next(rehydrated_function_call.get_gen()) == "foo"
    else:
        assert rehydrated_function_call.get() == "hello"


lc_app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


@lc_app.function()
def f(x):
    return x**2


def test_allow_cross_region_volumes(client, servicer):
    app = App()
    vol1 = NetworkFileSystem.from_name("xyz-1", create_if_missing=True)
    vol2 = NetworkFileSystem.from_name("xyz-2", create_if_missing=True)
    # Should pass flag for all the function's NetworkFileSystemMounts
    app.function(network_file_systems={"/sv-1": vol1, "/sv-2": vol2}, allow_cross_region_volumes=True)(dummy)

    with app.run(client=client):
        assert len(servicer.app_functions) == 1
        for func in servicer.app_functions.values():
            assert len(func.shared_volume_mounts) == 2
            for svm in func.shared_volume_mounts:
                assert svm.allow_cross_region


def test_allow_cross_region_volumes_webhook(client, servicer):
    # TODO(erikbern): this test seems a bit redundant
    app = App()
    vol1 = NetworkFileSystem.from_name("xyz-1", create_if_missing=True)
    vol2 = NetworkFileSystem.from_name("xyz-2", create_if_missing=True)
    # Should pass flag for all the function's NetworkFileSystemMounts
    app.function(network_file_systems={"/sv-1": vol1, "/sv-2": vol2}, allow_cross_region_volumes=True)(
        fastapi_endpoint()(dummy)
    )

    with app.run(client=client):
        assert len(servicer.app_functions) == 1
        for func in servicer.app_functions.values():
            assert len(func.shared_volume_mounts) == 2
            for svm in func.shared_volume_mounts:
                assert svm.allow_cross_region


def test_serialize_deserialize_function_handle(servicer, client):
    from modal._serialization import deserialize, serialize

    app = App()

    @app.function(serialized=True)
    @fastapi_endpoint()
    def my_handle():
        pass

    with pytest.raises(InvalidError, match="hasn't been hydrated"):
        serialize(my_handle)  # handle is not "live" yet! should not be serializable yet

    with app.run(client=client):
        blob = serialize(my_handle)

        rehydrated_function_handle = deserialize(blob, client)
        assert rehydrated_function_handle.object_id == my_handle.object_id
        assert isinstance(rehydrated_function_handle, Function)
        assert rehydrated_function_handle.web_url == "http://xyz.internal"


def test_default_cloud_provider(client, servicer, monkeypatch):
    app = App()

    monkeypatch.setenv("MODAL_DEFAULT_CLOUD", "xyz")
    app.function()(dummy)
    with app.run(client=client):
        object_id: str = app.registered_functions["dummy"].object_id
        f = servicer.app_functions[object_id]

    assert f.cloud_provider == api_pb2.CLOUD_PROVIDER_UNSPECIFIED  # No longer sent
    assert f.cloud_provider_str == "xyz"


def test_autoscaler_settings(client, servicer):
    app = App()

    kwargs: dict[str, typing.Any] = dict(  # No idea why we need that type hint
        min_containers=2,
        max_containers=10,
        scaledown_window=60,
    )
    f = app.function(**kwargs)(dummy)

    with app.run(client=client):
        defn = servicer.app_functions[f.object_id]
        # Test both backwards and forwards compatibility
        settings = defn.autoscaler_settings
        assert settings.min_containers == defn.warm_pool_size == kwargs["min_containers"]
        assert settings.max_containers == defn.concurrency_limit == kwargs["max_containers"]
        assert settings.scaledown_window == defn.task_idle_timeout_secs == kwargs["scaledown_window"]


@pytest.mark.parametrize(
    "new,old",
    [
        ("min_containers", "keep_warm"),
        ("max_containers", "concurrency_limit"),
        ("scaledown_window", "container_idle_timeout"),
    ],
)
def test_autoscaler_settings_deprecations(new, old):
    app = App()

    with pytest.warns(DeprecationError, match=f"{old} -> {new}"):
        app.function(**{old: 10})(dummy)  # type: ignore


def test_not_hydrated():
    with pytest.raises(ExecutionError):
        assert foo.remote(2, 4) == 20


def test_invalid_large_serialization(client):
    big_data = b"1" * 500000

    def f():
        return big_data

    with pytest.warns(UserWarning, match="larger than the recommended limit"):
        app = App()
        app.function(serialized=True)(f)
        with app.run(client=client):
            pass

    bigger_data = b"1" * 50000000

    def g():
        return bigger_data

    with pytest.raises(InvalidError):
        app = App()
        app.function(serialized=True)(g)
        with app.run(client=client):
            pass


def test_call_unhydrated_function():
    with pytest.raises(ExecutionError, match="hydrated"):
        foo.remote(123, 456)


def test_deps_explicit(client, servicer):
    app = App()

    image = Image.debian_slim()
    nfs_1 = NetworkFileSystem.from_name("nfs-1", create_if_missing=True)
    nfs_2 = NetworkFileSystem.from_name("nfs-2", create_if_missing=True)

    app.function(image=image, network_file_systems={"/nfs_1": nfs_1, "/nfs_2": nfs_2})(dummy)

    with app.run(client=client):
        object_id: str = app.registered_functions["dummy"].object_id
        f = servicer.app_functions[object_id]

    dep_object_ids = {d.object_id for d in f.object_dependencies}
    assert dep_object_ids == {image.object_id, nfs_1.object_id, nfs_2.object_id}


def assert_is_wrapped_dict(some_arg):
    assert type(some_arg) is modal.Dict  # this should not be a modal._Dict unwrapped instance!
    return some_arg


def test_calls_should_not_unwrap_modal_objects(servicer, client):
    app = App()
    foo = app.function()(assert_is_wrapped_dict)
    servicer.function_body(assert_is_wrapped_dict)

    # make sure the serialized object is an actual Dict and not a _Dict in all user code contexts
    with app.run(client=client), modal.Dict.ephemeral(client=client) as some_modal_object:
        assert type(foo.remote(some_modal_object)) is modal.Dict
        fc = foo.spawn(some_modal_object)
        assert type(fc.get()) is modal.Dict
        for ret in foo.map([some_modal_object]):
            assert type(ret) is modal.Dict
        for ret in foo.starmap([[some_modal_object]]):
            assert type(ret) is modal.Dict
        foo.for_each([some_modal_object])

    assert len(servicer.client_calls) == 5


def assert_is_wrapped_dict_gen(some_arg):
    assert type(some_arg) is modal.Dict  # this should not be a modal._Dict unwrapped instance!
    yield some_arg


def test_calls_should_not_unwrap_modal_objects_gen(servicer, client):
    app = App()
    foo = app.function()(assert_is_wrapped_dict_gen)
    servicer.function_body(assert_is_wrapped_dict_gen)

    # make sure the serialized object is an actual Dict and not a _Dict in all user code contexts
    with app.run(client=client), modal.Dict.ephemeral(client=client) as some_modal_object:
        assert type(next(foo.remote_gen(some_modal_object))) is modal.Dict
        with pytest.raises(DeprecationError):
            foo.spawn(some_modal_object)

    assert len(servicer.client_calls) == 1


def test_function_deps_have_ids(client, servicer, monkeypatch, test_dir, set_env_client, disable_auto_mount):
    monkeypatch.syspath_prepend(test_dir / "supports")
    app = App()
    app.function(
        image=modal.Image.debian_slim().add_local_python_source("pkg_a"),
        volumes={"/vol": modal.Volume.from_name("vol", create_if_missing=True)},
        network_file_systems={"/vol": modal.NetworkFileSystem.from_name("nfs", create_if_missing=True)},
        secrets=[modal.Secret.from_dict({"foo": "bar"})],
    )(dummy)

    with servicer.intercept() as ctx:
        with app.run(client=client):
            pass

    function_create = ctx.pop_request("FunctionCreate")
    assert len(function_create.function.mount_ids) == 3  # client mount, explicit mount, entrypoint mount
    for mount_id in function_create.function.mount_ids:
        assert mount_id

    for dep in function_create.function.object_dependencies:
        assert dep.object_id


def test_no_state_reuse(client, servicer, supports_dir, disable_auto_mount):
    # two separate instances of the same mount content - triggers deduplication logic

    img = (
        Image.debian_slim()
        .add_local_file(supports_dir / "pyproject.toml", "/root/")
        .add_local_file(supports_dir / "pyproject.toml", "/root/")
    )
    app = App("reuse-mount-app")
    app.function(image=img)(dummy)

    with servicer.intercept() as ctx:
        deploy_app(app, client=client)
        func_create = ctx.pop_request("FunctionCreate")
        first_deploy_mounts = set(func_create.function.mount_ids)
        assert len(first_deploy_mounts) == 3  # client mount, one of the explicit mounts, entrypoint mount

    with servicer.intercept() as ctx:
        deploy_app(app, client=client)
        func_create = ctx.pop_request("FunctionCreate")
        second_deploy_mounts = set(func_create.function.mount_ids)
        assert len(second_deploy_mounts) == 3  # client mount, one of the explicit mounts, entrypoint mount

    # mount ids should not overlap between first and second deploy, except for client mount
    assert first_deploy_mounts & second_deploy_mounts == {servicer.default_published_client_mount}


@pytest.mark.asyncio
async def test_map_large_inputs(client, servicer, monkeypatch, blob_server):
    # TODO: tests making use of mock blob server currently have to be async, since the
    #  blob server runs as an async pytest fixture which will have its event loop blocked
    #  by the test itself otherwise... Should move to its own thread.
    monkeypatch.setattr("modal._utils.function_utils.MAX_OBJECT_SIZE_BYTES", 1)
    servicer.use_blob_outputs = True
    app = App()
    dummy_modal = app.function()(dummy)

    _, blobs = blob_server
    async with app.run.aio(client=client):
        assert len(blobs) == 0
        assert [a async for a in dummy_modal.map.aio(range(100))] == [i**2 for i in range(100)]
        assert len(servicer.cleared_function_calls) == 1

    assert len(blobs) == 200  # inputs + outputs


@pytest.mark.asyncio
async def test_non_aio_map_in_async_caller_error(client):
    dummy_function = app.function()(dummy)

    with app.run(client=client):
        with pytest.raises(InvalidError, match=".map.aio"):
            for _ in dummy_function.map([1, 2, 3]):
                pass

        # using .aio should be ok:
        res = [r async for r in dummy_function.map.aio([1, 2, 3])]
        assert res == [1, 4, 9]

        # we might want to deprecate this syntax (async for ... in map without .aio),
        # but we support it for backwards compatibility for now:
        res = [r async for r in dummy_function.map([1, 2, 4])]
        assert res == [1, 4, 16]


def test_warn_on_local_volume_mount(client, servicer):
    vol = modal.Volume.from_name("my-vol")
    dummy_function = app.function(volumes={"/foo": vol})(dummy)

    assert modal.is_local()
    with pytest.warns(match="local"):
        dummy_function.local()


class X:
    def f(self): ...


def test_function_decorator_on_method():
    app = modal.App()

    with pytest.raises(InvalidError, match="@app.cls"):
        app.function()(X.f)


def test_batch_function_invalid_error():
    app = App()

    with pytest.raises(InvalidError, match="must be a positive integer"):
        app.function(batched(max_batch_size=0, wait_ms=1))(dummy)

    with pytest.raises(InvalidError, match="must be a non-negative integer"):
        app.function(batched(max_batch_size=1, wait_ms=-1))(dummy)

    with pytest.raises(InvalidError, match="must be less than"):
        app.function(batched(max_batch_size=1000, wait_ms=1))(dummy)

    with pytest.raises(InvalidError, match="must be less than"):
        app.function(batched(max_batch_size=1, wait_ms=10 * 60 * 1000))(dummy)

    with pytest.raises(InvalidError, match="cannot return generators"):

        @app.function(serialized=True)
        @batched(max_batch_size=1, wait_ms=1)
        def f(x):
            yield [x_i**2 for x_i in x]

    with pytest.raises(InvalidError, match="does not accept default arguments"):

        @app.function(serialized=True)
        @batched(max_batch_size=1, wait_ms=1)
        def g(x=1):
            return [x_i**2 for x_i in x]


def test_experimental_spawn(client, servicer):
    app = App()
    dummy_modal = app.function()(dummy)

    with servicer.intercept() as ctx:
        with app.run(client=client):
            dummy_modal._experimental_spawn(1, 2)

    # Verify the correct invocation type is set
    function_map = ctx.pop_request("FunctionMap")
    assert function_map.function_call_invocation_type == api_pb2.FUNCTION_CALL_INVOCATION_TYPE_ASYNC


def test_from_name_web_url(servicer, set_env_client):
    f = Function.from_name("dummy-app", "func")

    with servicer.intercept() as ctx:
        ctx.add_response(
            "FunctionGet",
            api_pb2.FunctionGetResponse(
                function_id="fu-1", handle_metadata=api_pb2.FunctionHandleMetadata(web_url="test.internal")
            ),
        )
        assert f.web_url == "test.internal"


@pytest.mark.parametrize(
    ["config_automount", "app_constructor_value", "function_decorator_value", "expected_mounts"],
    [
        (None, None, None, 2),  # default with no options: entrypoint + first party (automount)
        ("0", None, None, 1),  # automount=0 in config - entrypoint only. Warn about config being deprecated
        ("1", None, None, 2),  # automount=1 explicit in config. Warn about config based automount=1 going away
        ("1", "False", None, 0),  # automount=1 explicit in config. Warn about config based automount=1 going away
        ("0", "False", None, 0),
        (None, "False", None, 0),
        (None, "False", "True", 1),
        (None, "True", "False", 0),
        # "legacy" mode is currently not enabled except as the default value
        # (None, "False", "'legacy'", 2),
        # (None, "True", "'legacy'", 2),
    ],
)
def test_include_source_mode(
    app_constructor_value,
    function_decorator_value,
    config_automount,
    expected_mounts,
    servicer,
    credentials,
    tmp_path,
    monkeypatch,
):
    # a little messy since it tests the "end to end" mounting behavior for the app
    app_constructor_value = "None" if app_constructor_value is None else app_constructor_value
    function_decorator_value = "None" if function_decorator_value is None else function_decorator_value
    src = f"""
import modal
import mod  # mod.py needs to be added for this file to load, so it needs to be included as source

app = modal.App(include_source={app_constructor_value})

@app.function(include_source={function_decorator_value})
def f():
    pass
"""
    entrypoint_file = tmp_path / "main.py"
    (tmp_path / "mod.py").touch()  # some file
    entrypoint_file.write_text(src)

    monkeypatch.delenv("MODAL_AUTOMOUNT")
    if config_automount is not None:
        env = {**os.environ, "MODAL_AUTOMOUNT": config_automount}
    else:
        env = {**os.environ}
    output = deploy_app_externally(servicer, credentials, str(entrypoint_file), env=env)
    print(output)
    mounts = servicer.mounts_excluding_published_client()

    assert len(mounts) == expected_mounts
