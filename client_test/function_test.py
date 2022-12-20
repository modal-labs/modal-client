# Copyright Modal Labs 2022
import asyncio
import pytest
import time

import cloudpickle

from modal import Proxy, Stub
from modal.exception import DeprecationError, InvalidError
from modal.functions import FunctionCall, gather
from modal.stub import AioStub
from modal_proto import api_pb2

stub = Stub()


@stub.function()
def foo():
    pass  # not actually used in test (servicer returns sum of square of all args)


def dummy():
    pass  # not actually used in test (servicer returns sum of square of all args)


def test_run_function(client):
    with stub.run(client=client):
        assert foo.call(2, 4) == 20


def test_map(client):
    stub = Stub()
    dummy_modal = stub.function(dummy)

    with stub.run(client=client):
        assert list(dummy_modal.map([5, 2], [4, 3])) == [41, 13]
        assert set(dummy_modal.map([5, 2], [4, 3], order_outputs=False)) == {13, 41}


_side_effect_count = 0


def side_effect(_):
    global _side_effect_count
    _side_effect_count += 1


def test_for_each(client, servicer):
    stub = Stub()
    side_effect_modal = stub.function(servicer.function_body(side_effect))

    assert _side_effect_count == 0
    with stub.run(client=client):
        side_effect_modal.for_each(range(10))
    assert _side_effect_count == 10


def custom_function(x):
    if x % 2 == 0:
        return x


def test_map_none_values(client, servicer):
    stub = Stub()

    custom_function_modal = stub.function(servicer.function_body(custom_function))
    with stub.run(client=client):
        assert list(custom_function_modal.map(range(4))) == [0, None, 2, None]


def test_starmap(client):
    stub = Stub()

    dummy_modal = stub.function(dummy)
    with stub.run(client=client):
        assert list(dummy_modal.starmap([[5, 2], [4, 3]])) == [29, 25]


def test_function_memory_request(client):
    stub = Stub()
    stub.function(dummy, memory=2048)


def test_function_cpu_request(client):
    stub = Stub()
    stub.function(dummy, cpu=2.0)


def later():
    return "hello"


def test_function_future(client, servicer):
    stub = Stub()

    later_modal = stub.function(servicer.function_body(later))

    with stub.run(client=client):
        future = later_modal.spawn()
        assert isinstance(future, FunctionCall)

        servicer.function_is_running = True
        assert future.object_id == "fc-1"

        with pytest.raises(TimeoutError):
            future.get(0.01)

        servicer.function_is_running = False
        assert future.get(0.01) == "hello"

        with pytest.warns(DeprecationError):
            future = later_modal.submit()
            assert isinstance(future, FunctionCall)

        servicer.function_is_running = True
        assert future.object_id == "fc-2"


@pytest.mark.asyncio
async def test_function_future_async(client, servicer):
    stub = AioStub()

    later_modal = stub.function(servicer.function_body(later))

    async with stub.run(client=client):
        future = await later_modal.spawn()
        servicer.function_is_running = True

        with pytest.raises(TimeoutError):
            await future.get(0.01)

        servicer.function_is_running = False
        assert await future.get(0.01) == "hello"


def later_gen():
    yield "foo"


@pytest.mark.asyncio
async def test_generator(client, servicer):
    stub = Stub()

    with pytest.warns(DeprecationError):
        later_gen_modal = stub.generator(later_gen)
    assert later_gen_modal.is_generator

    later_gen_modal = stub.function(later_gen)
    assert later_gen_modal.is_generator


@pytest.mark.asyncio
async def test_generator_future(client, servicer):
    stub = Stub()

    later_gen_modal = stub.function(later_gen)
    with stub.run(client=client):
        assert later_gen_modal.spawn() is None  # until we have a nice interface for polling generator futures


async def slo1(sleep_seconds):
    # need to use async function body in client test to run stuff in parallel
    # but calling interface is still non-asyncio
    await asyncio.sleep(sleep_seconds)
    return sleep_seconds


def test_sync_parallelism(client, servicer):
    stub = Stub()

    slo1_modal = stub.function(servicer.function_body(slo1))
    with stub.run(client=client):
        t0 = time.time()
        # NOTE tests breaks in macOS CI if the smaller time is smaller than ~300ms
        res = gather(slo1_modal.spawn(0.31), slo1_modal.spawn(0.3))
        t1 = time.time()
        assert res == [0.31, 0.3]  # results should be ordered as inputs, not by completion time
        assert t1 - t0 < 0.6  # less than the combined runtime, make sure they run in parallel


def test_proxy(client, servicer):
    stub = Stub()

    stub.function(dummy, proxy=Proxy.from_name("my-proxy"))
    with stub.run(client=client):
        pass


class CustomException(Exception):
    pass


def failure():
    raise CustomException("foo!")


def test_function_exception(client, servicer):
    stub = Stub()

    failure_modal = stub.function(servicer.function_body(failure))

    with stub.run(client=client):
        with pytest.raises(CustomException) as excinfo:
            failure_modal.call()
        assert "foo!" in str(excinfo.value)


def import_failure():
    raise ImportError("attempted relative import with no known parent package")


def test_function_relative_import_hint(client, servicer):
    stub = Stub()

    import_failure_modal = stub.function(servicer.function_body(import_failure))
    with stub.run(client=client):
        with pytest.raises(ImportError) as excinfo:
            import_failure_modal.call()
        assert "HINT" in str(excinfo.value)


lifecycle_stub = Stub()


class Foo:
    bar = "hello"

    @lifecycle_stub.function(serialized=True)
    def run(self):
        return self.bar


def test_serialized_function_includes_lifecycle_class(client, servicer):
    with lifecycle_stub.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.definition_type == api_pb2.Function.DEFINITION_TYPE_SERIALIZED

    func = cloudpickle.loads(func_def.function_serialized)
    cls = cloudpickle.loads(func_def.class_serialized)
    assert func(cls()) == "hello"


def test_nonglobal_function():
    stub = Stub()

    with pytest.raises(InvalidError) as excinfo:

        @stub.function
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

    for s in ["foo", "bar"]:

        @stub.function(name=f"ret_{s}", serialized=True)
        def returner():
            return s

    with stub.run(client=client):
        pass

    functions = {}
    for func in servicer.app_functions.values():
        functions[func.function_name] = cloudpickle.loads(func.function_serialized)

    assert len(functions) == 2
    assert functions["ret_foo"]() == "foo"
    assert functions["ret_bar"]() == "bar"
