# Copyright Modal Labs 2022
import asyncio
import inspect
import pytest
import time
import typing

import cloudpickle
from synchronicity.exceptions import UserCodeException

from modal import Image, NetworkFileSystem, Proxy, Stub, web_endpoint
from modal.exception import DeprecationError, ExecutionError, InvalidError
from modal.functions import Function, FunctionCall, gather
from modal.runner import deploy_stub
from modal_proto import api_pb2

stub = Stub()


@stub.function()
def foo(p, q):
    return p + q + 11  # not actually used in test (servicer returns sum of square of all args)


@stub.function()
async def async_foo(p, q):
    return p + q + 12


def dummy():
    pass  # not actually used in test (servicer returns sum of square of all args)


def test_run_function(client, servicer):
    assert len(servicer.cleared_function_calls) == 0
    with stub.run(client=client):
        # Old-style remote calls
        with pytest.raises(DeprecationError):
            foo.call(2, 4)

        # New-style remote calls
        assert foo.remote(2, 4) == 20
        assert len(servicer.cleared_function_calls) == 1

        # Make sure we can also call the Function object
        fun = stub.foo
        assert isinstance(fun, Function)
        assert fun.remote(2, 4) == 20
        assert len(servicer.cleared_function_calls) == 2


@pytest.mark.asyncio
async def test_call_function_locally(client, servicer):
    # Old-style local calls
    with pytest.raises(DeprecationError):
        foo(22, 44)

    with pytest.raises(DeprecationError):
        await async_foo(22, 44)

    # New-style local calls
    assert foo.local(22, 44) == 77  # call it locally
    assert await async_foo.local(22, 44) == 78

    with stub.run(client=client):
        assert foo.remote(2, 4) == 20
        with pytest.raises(DeprecationError):
            foo(22, 55)
        with pytest.raises(DeprecationError):
            await async_foo(22, 44)
        assert async_foo.remote(2, 4) == 20
        assert await async_foo.remote.aio(2, 4) == 20

        # Make sure we can also call the Function object
        assert isinstance(stub.foo, Function)
        assert isinstance(stub.async_foo, Function)
        with pytest.raises(DeprecationError):
            stub.foo(22, 55)
        with pytest.raises(DeprecationError):
            await stub.async_foo(22, 44)


@pytest.mark.parametrize("slow_put_inputs", [False, True])
@pytest.mark.timeout(120)
def test_map(client, servicer, slow_put_inputs):
    servicer.slow_put_inputs = slow_put_inputs

    stub = Stub()
    dummy_modal = stub.function()(dummy)

    assert len(servicer.cleared_function_calls) == 0
    with stub.run(client=client):
        assert list(dummy_modal.map([5, 2], [4, 3])) == [41, 13]
        assert len(servicer.cleared_function_calls) == 1
        assert set(dummy_modal.map([5, 2], [4, 3], order_outputs=False)) == {13, 41}
        assert len(servicer.cleared_function_calls) == 2

        # Make sure we can map on the Function object too
        fun = stub.dummy
        assert isinstance(fun, Function)
        assert list(fun.map([5, 2], [4, 3])) == [41, 13]
        assert len(servicer.cleared_function_calls) == 3


_side_effect_count = 0


def side_effect(_):
    global _side_effect_count
    _side_effect_count += 1


def test_for_each(client, servicer):
    stub = Stub()
    side_effect_modal = stub.function()(servicer.function_body(side_effect))
    assert _side_effect_count == 0
    with stub.run(client=client):
        side_effect_modal.for_each(range(10))

        # Call stub function too
        fun = stub.side_effect
        assert isinstance(fun, Function)
        fun.for_each(range(10))

    assert _side_effect_count == 20


def custom_function(x):
    if x % 2 == 0:
        return x


def test_map_none_values(client, servicer):
    stub = Stub()

    custom_function_modal = stub.function()(servicer.function_body(custom_function))

    with stub.run(client=client):
        assert list(custom_function_modal.map(range(4))) == [0, None, 2, None]


def test_starmap(client):
    stub = Stub()

    dummy_modal = stub.function()(dummy)
    with stub.run(client=client):
        assert list(dummy_modal.starmap([[5, 2], [4, 3]])) == [29, 25]


def test_function_memory_request(client):
    stub = Stub()
    stub.function(memory=2048)(dummy)


def test_function_cpu_request(client):
    stub = Stub()
    stub.function(cpu=2.0)(dummy)


def later():
    return "hello"


def test_function_future(client, servicer):
    stub = Stub()

    later_modal = stub.function()(servicer.function_body(later))
    with stub.run(client=client):
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


@pytest.mark.asyncio
async def test_function_future_async(client, servicer):
    stub = Stub()

    later_modal = stub.function()(servicer.function_body(later))

    async with stub.run(client=client):
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
    stub = Stub()

    later_gen_modal = stub.function()(later_gen)

    def dummy():
        yield "bar"
        yield "baz"
        yield "boo"

    servicer.function_body(dummy)

    assert len(servicer.cleared_function_calls) == 0
    with stub.run(client=client):
        assert later_gen_modal.is_generator
        res: typing.Generator = later_gen_modal.remote_gen()  # type: ignore
        # Generators fulfil the *iterator protocol*, which requires both these methods.
        # https://docs.python.org/3/library/stdtypes.html#typeiter
        assert hasattr(res, "__iter__")  # strangely inspect.isgenerator returns false
        assert hasattr(res, "__next__")
        assert next(res) == "bar"
        assert list(res) == ["baz", "boo"]
        assert len(servicer.cleared_function_calls) == 1

        # Check deprecated interface
        with pytest.raises(DeprecationError):
            later_gen_modal.call()


@pytest.mark.asyncio
async def test_generator_async(client, servicer):
    stub = Stub()

    later_gen_modal = stub.function()(async_later_gen)

    async def async_dummy():
        yield "bar"
        yield "baz"

    servicer.function_body(async_dummy)

    assert len(servicer.cleared_function_calls) == 0
    async with stub.run(client=client):
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
    stub = Stub()

    later_gen_modal = stub.function()(later_gen)
    with stub.run(client=client):
        assert later_gen_modal.spawn() is None  # until we have a nice interface for polling generator futures


def gen_with_arg(i):
    yield "foo"


@pytest.mark.asyncio
async def test_generator_map_success(client, servicer):
    stub = Stub()

    gen_with_arg_modal = stub.function()(gen_with_arg)

    def dummy(i):
        yield i, "bar"
        yield i, "baz"

    servicer.function_body(dummy)

    assert len(servicer.cleared_function_calls) == 0
    with stub.run(client=client):
        assert gen_with_arg_modal.is_generator
        res = set(gen_with_arg_modal.map([1, 2, 3]))
        assert res == {(1, "bar"), (1, "baz"), (2, "bar"), (2, "baz"), (3, "bar"), (3, "baz")}


@pytest.mark.asyncio
async def test_generator_map_exception(client, servicer):
    stub = Stub()

    gen_with_arg_modal = stub.function()(gen_with_arg)

    def dummy(i):
        yield i, "bar"
        if i == 2:
            raise CustomException("boo!")
        yield i, "baz"

    servicer.function_body(dummy)

    assert len(servicer.cleared_function_calls) == 0
    with stub.run(client=client):
        assert gen_with_arg_modal.is_generator

        with pytest.raises(CustomException) as exc_info:
            list(gen_with_arg_modal.map([1, 2, 3]))
        assert exc_info.value.args == ("boo!",)


@pytest.mark.asyncio
async def test_generator_map_return_exceptions(client, servicer):
    stub = Stub()

    gen_with_arg_modal = stub.function()(gen_with_arg)

    def dummy(i):
        yield i, "bar"
        if i == 2:
            raise CustomException("boo!")
        yield i, "baz"

    servicer.function_body(dummy)

    assert len(servicer.cleared_function_calls) == 0
    with stub.run(client=client):
        assert gen_with_arg_modal.is_generator

        results = set()
        received_exc = False
        for result in gen_with_arg_modal.map([1, 2, 3], return_exceptions=True):
            # TODO: this should just be CustomException directly.
            if isinstance(result, UserCodeException):
                assert isinstance(result.exc, CustomException)
                assert result.exc.args == ("boo!",)
                received_exc = True
            else:
                results.add(result)

        assert received_exc
        assert results == {(1, "bar"), (1, "baz"), (2, "bar"), (3, "bar"), (3, "baz")}


async def slo1(sleep_seconds):
    # need to use async function body in client test to run stuff in parallel
    # but calling interface is still non-asyncio
    await asyncio.sleep(sleep_seconds)
    return sleep_seconds


def test_sync_parallelism(client, servicer):
    stub = Stub()

    slo1_modal = stub.function()(servicer.function_body(slo1))
    with stub.run(client=client):
        t0 = time.time()
        # NOTE tests breaks in macOS CI if the smaller time is smaller than ~300ms
        res = gather(slo1_modal.spawn(0.31), slo1_modal.spawn(0.3))
        t1 = time.time()
        assert res == [0.31, 0.3]  # results should be ordered as inputs, not by completion time
        assert t1 - t0 < 0.6  # less than the combined runtime, make sure they run in parallel


def test_proxy(client, servicer):
    stub = Stub()

    stub.function(proxy=Proxy.from_name("my-proxy"))(dummy)
    with stub.run(client=client):
        pass


class CustomException(Exception):
    pass


def failure():
    raise CustomException("foo!")


def test_function_exception(client, servicer):
    stub = Stub()

    failure_modal = stub.function()(servicer.function_body(failure))
    with stub.run(client=client):
        with pytest.raises(CustomException) as excinfo:
            failure_modal.remote()
        assert "foo!" in str(excinfo.value)


@pytest.mark.asyncio
async def test_function_exception_async(client, servicer):
    stub = Stub()

    failure_modal = stub.function()(servicer.function_body(failure))
    async with stub.run(client=client):
        with pytest.raises(CustomException) as excinfo:
            coro = failure_modal.remote.aio()
            assert inspect.isawaitable(
                coro
            )  # mostly for mypy, since output could technically be an async generator which isn't awaitable in the same sense
            await coro
        assert "foo!" in str(excinfo.value)


def custom_exception_function(x):
    if x == 4:
        raise CustomException("bad")
    return x * x


def test_map_exceptions(client, servicer):
    stub = Stub()

    custom_function_modal = stub.function()(servicer.function_body(custom_exception_function))

    with stub.run(client=client):
        assert list(custom_function_modal.map(range(4))) == [0, 1, 4, 9]

        with pytest.raises(CustomException) as excinfo:
            list(custom_function_modal.map(range(6)))
        assert "bad" in str(excinfo.value)

        res = list(custom_function_modal.map(range(6), return_exceptions=True))
        assert res[:4] == [0, 1, 4, 9] and res[5] == 25
        assert type(res[4]) == UserCodeException and "bad" in str(res[4])


def import_failure():
    raise ImportError("attempted relative import with no known parent package")


def test_function_relative_import_hint(client, servicer):
    stub = Stub()

    import_failure_modal = stub.function()(servicer.function_body(import_failure))

    with stub.run(client=client):
        with pytest.raises(ImportError) as excinfo:
            import_failure_modal.remote()
        assert "HINT" in str(excinfo.value)


def test_nonglobal_function():
    stub = Stub()

    with pytest.raises(InvalidError) as excinfo:

        @stub.function()
        def f():
            pass

    assert "global scope" in str(excinfo.value)


def test_non_global_serialized_function():
    stub = Stub()

    @stub.function(serialized=True)
    def f():
        pass


def test_closure_valued_serialized_function(client, servicer):
    stub = Stub()

    def make_function(s):
        @stub.function(name=f"ret_{s}", serialized=True)
        def returner():
            return s

    for s in ["foo", "bar"]:
        make_function(s)

    with stub.run(client=client):
        pass

    functions = {}
    for func in servicer.app_functions.values():
        functions[func.function_name] = cloudpickle.loads(func.function_serialized)

    assert len(functions) == 2
    assert functions["ret_foo"]() == "foo"
    assert functions["ret_bar"]() == "bar"


def test_new_hydrated_internal(client, servicer):
    obj = FunctionCall._new_hydrated("fc-123", client, None)
    assert obj.object_id == "fc-123"


def test_from_id(client, servicer):
    stub = Stub()

    @stub.function(serialized=True)
    @web_endpoint()
    def foo():
        pass

    deploy_stub(stub, "dummy", client=client)

    function_id = foo.object_id
    assert function_id
    assert foo.web_url

    rehydrated_function = Function.from_id(function_id, client=client)
    assert isinstance(rehydrated_function, Function)

    assert rehydrated_function.object_id == function_id
    assert rehydrated_function.web_url == foo.web_url

    function_call = foo.spawn()
    assert function_call.object_id
    # Used in a few examples to construct FunctionCall objects
    rehydrated_function_call = FunctionCall.from_id(function_call.object_id, client)
    assert rehydrated_function_call.object_id == function_call.object_id


lc_stub = Stub()


@lc_stub.function()
def f(x):
    return x**2


def test_allow_cross_region_volumes(client, servicer):
    stub = Stub()
    vol1, vol2 = NetworkFileSystem.new(), NetworkFileSystem.new()
    # Should pass flag for all the function's NetworkFileSystemMounts
    stub.function(network_file_systems={"/sv-1": vol1, "/sv-2": vol2}, allow_cross_region_volumes=True)(dummy)

    with stub.run(client=client):
        assert len(servicer.app_functions) == 1
        for func in servicer.app_functions.values():
            assert len(func.shared_volume_mounts) == 2
            for svm in func.shared_volume_mounts:
                assert svm.allow_cross_region


def test_allow_cross_region_volumes_webhook(client, servicer):
    # TODO(erikbern): this stest seems a bit redundant
    stub = Stub()
    vol1, vol2 = NetworkFileSystem.new(), NetworkFileSystem.new()
    # Should pass flag for all the function's NetworkFileSystemMounts
    stub.function(network_file_systems={"/sv-1": vol1, "/sv-2": vol2}, allow_cross_region_volumes=True)(
        web_endpoint()(dummy)
    )

    with stub.run(client=client):
        assert len(servicer.app_functions) == 1
        for func in servicer.app_functions.values():
            assert len(func.shared_volume_mounts) == 2
            for svm in func.shared_volume_mounts:
                assert svm.allow_cross_region


def test_shared_volumes(client, servicer):
    stub = Stub()
    vol = NetworkFileSystem.new()
    with pytest.raises(DeprecationError):
        stub.function(shared_volumes={"/sv-1": vol})


def test_serialize_deserialize_function_handle(servicer, client):
    from modal._serialization import deserialize, serialize

    stub = Stub()

    @stub.function(serialized=True)
    @web_endpoint()
    def my_handle():
        pass

    with pytest.raises(InvalidError, match="hasn't been created"):
        serialize(my_handle)  # handle is not "live" yet! should not be serializable yet

    with stub.run(client=client):
        blob = serialize(my_handle)

        rehydrated_function_handle = deserialize(blob, client)
        assert rehydrated_function_handle.object_id == my_handle.object_id
        assert isinstance(rehydrated_function_handle, Function)
        assert rehydrated_function_handle.web_url == "http://xyz.internal"


def test_default_cloud_provider(client, servicer, monkeypatch):
    stub = Stub()

    monkeypatch.setenv("MODAL_DEFAULT_CLOUD", "oci")
    stub.function()(dummy)
    with stub.run(client=client):
        f = servicer.app_functions[stub.dummy.object_id]

    assert f.cloud_provider == api_pb2.CLOUD_PROVIDER_OCI


def test_not_hydrated():
    with pytest.raises(ExecutionError):
        assert foo.remote(2, 4) == 20


def test_invalid_large_serialization(client):
    big_data = b"1" * 500000

    def f():
        return big_data

    with pytest.warns(UserWarning, match="larger than the recommended limit"):
        stub = Stub()
        stub.function(serialized=True)(f)
        with stub.run(client=client):
            pass

    bigger_data = b"1" * 50000000

    def g():
        return bigger_data

    with pytest.raises(InvalidError):
        stub = Stub()
        stub.function(serialized=True)(g)
        with stub.run(client=client):
            pass


def test_call_unhydrated_function():
    with pytest.raises(ExecutionError, match="hydrated"):
        foo.remote(123)


def test_deps(client, servicer):
    stub = Stub()

    image = Image.debian_slim()
    nfs_1 = NetworkFileSystem.new()
    nfs_2 = NetworkFileSystem.new()

    stub.function(image=image, network_file_systems={"/nfs_1": nfs_1, "/nfs_2": nfs_2})(dummy)

    with stub.run(client=client):
        f = servicer.app_functions[stub.dummy.object_id]

    dep_object_ids = set(d.object_id for d in f.object_dependencies)
    assert dep_object_ids == set([image.object_id, nfs_1.object_id, nfs_2.object_id])
