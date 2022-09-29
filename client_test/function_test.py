import asyncio
import pytest
import time

from modal import Proxy, Stub
from modal.functions import FunctionCall, gather
from modal.stub import AioStub

stub = Stub()


@stub.function()
def foo():
    pass  # not actually used in test (servicer returns sum of square of all args)


def test_run_function(client):
    with stub.run(client=client):
        assert foo(2, 4) == 20


def test_map(client):
    stub = Stub()

    @stub.function
    def dummy():
        pass  # not actually used in test (servicer returns sum of square of all args)

    with stub.run(client=client):
        assert list(dummy.map([5, 2], [4, 3])) == [41, 13]
        assert set(dummy.map([5, 2], [4, 3], order_outputs=False)) == {13, 41}


def test_for_each(client, servicer):
    stub = Stub()

    res = 0

    @stub.function
    @servicer.function_body
    def side_effect(_):
        nonlocal res
        res += 1

    with stub.run(client=client):
        side_effect.for_each(range(10))

    assert res == 10


def test_map_none_values(client, servicer):
    stub = Stub()

    @stub.function
    @servicer.function_body
    def custom_function(x):
        if x % 2 == 0:
            return x

    with stub.run(client=client):
        assert list(custom_function.map(range(4))) == [0, None, 2, None]


def test_starmap(client):
    stub = Stub()

    @stub.function
    def dummy():
        pass  # not actually used in test (servicer returns sum of square of all args)

    with stub.run(client=client):
        assert list(dummy.starmap([[5, 2], [4, 3]])) == [29, 25]


def test_function_memory_request(client):
    stub = Stub()

    @stub.function(memory=2048)
    def f1():
        pass


def test_function_cpu_request(client):
    stub = Stub()

    @stub.function(cpu=2.0)
    def f1():
        pass


def test_function_future(client, servicer):
    stub = Stub()

    @stub.function()
    @servicer.function_body
    def later():
        return "hello"

    with stub.run(client=client):
        future = later.submit()
        assert isinstance(future, FunctionCall)

        servicer.function_is_running = True
        assert future.object_id == "fc-1"

        with pytest.raises(TimeoutError):
            future.get(0.01)

        servicer.function_is_running = False
        assert future.get(0.01) == "hello"


@pytest.mark.asyncio
async def test_function_future_async(client, servicer):
    stub = AioStub()

    @stub.function()
    @servicer.function_body
    def later():
        return "foo"

    async with stub.run(client=client):
        future = await later.submit()
        servicer.function_is_running = True

        with pytest.raises(TimeoutError):
            await future.get(0.01)

        servicer.function_is_running = False
        assert await future.get(0.01) == "foo"


@pytest.mark.asyncio
async def test_generator_future(client, servicer):
    stub = Stub()

    @stub.generator()
    def later():
        yield "foo"

    with stub.run(client=client):
        assert later.submit() is None  # until we have a nice interface for polling generator futures


def test_sync_parallelism(client, servicer):
    stub = Stub()

    @stub.function()
    @servicer.function_body
    async def slo1(sleep_seconds):
        # need to use async function body in client test to run stuff in parallel
        # but calling interface is still non-asyncio
        await asyncio.sleep(sleep_seconds)
        return sleep_seconds

    with stub.run(client=client):
        t0 = time.time()
        # NOTE tests breaks in macOS CI if the smaller time is smaller than ~300ms
        res = gather(slo1.submit(0.31), slo1.submit(0.3))
        t1 = time.time()
        assert res == [0.31, 0.3]  # results should be ordered as inputs, not by completion time
        assert t1 - t0 < 0.6  # less than the combined runtime, make sure they run in parallel


def test_proxy(client, servicer):
    stub = Stub()

    @stub.function(proxy=Proxy.from_name("my-proxy"))
    def f():
        pass

    with stub.run(client=client):
        pass


class CustomException(Exception):
    pass


def test_function_exception(client, servicer):
    stub = Stub()

    @stub.function
    @servicer.function_body
    def failure():
        raise CustomException("foo!")

    with stub.run(client=client):
        with pytest.raises(CustomException) as excinfo:
            failure()
        assert "foo!" in str(excinfo.value)


def test_function_relative_import_hint(client, servicer):
    stub = Stub()

    @stub.function
    @servicer.function_body
    def failure():
        raise ImportError("attempted relative import with no known parent package")

    with stub.run(client=client):
        with pytest.raises(ImportError) as excinfo:
            failure()
        assert "HINT" in str(excinfo.value)
