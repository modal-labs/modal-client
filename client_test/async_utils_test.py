import asyncio
import platform
import pytest

from modal_utils.async_utils import (
    TaskContext,
    chunk_generator,
    queue_batch_iterator,
    retry,
)

skip_non_linux = pytest.mark.skipif(
    platform.system() != "Linux", reason="sleep is inaccurate on Github Actions runners."
)


class SampleException(Exception):
    pass


class FailNTimes:
    def __init__(self, n_failures):
        self.n_failures = n_failures
        self.n_calls = 0

    async def __call__(self, x):
        self.n_calls += 1
        if self.n_calls < self.n_failures:
            raise SampleException("Something bad happened")
        else:
            return x + 1


@pytest.mark.asyncio
async def test_retry():
    f_retry = retry(FailNTimes(3))
    assert await f_retry(42) == 43

    with pytest.raises(SampleException):
        f_retry = retry(FailNTimes(4))
        assert await f_retry(42) == 43

    f_retry = retry(n_attempts=5)(FailNTimes(5))
    assert await f_retry(42) == 43

    with pytest.raises(SampleException):
        f_retry = retry(n_attempts=5)(FailNTimes(6))
        assert await f_retry(42) == 43


async def unchunk_generator(generator):
    ret = []
    async for chunk in generator:
        loop_ret = []
        ret.append(loop_ret)
        try:
            async for value in chunk:
                loop_ret.append(value)
        except SampleException:
            loop_ret.append("exc")
            break
    return ret


@skip_non_linux
@pytest.mark.asyncio
async def test_chunk_generator():
    async def generator():
        try:
            for i in range(10):
                await asyncio.sleep(0.1)
                yield i
        except BaseException as exc:
            print(f"generator exc {exc}")
            raise

    ret = await unchunk_generator(chunk_generator(generator(), 0.33))
    assert ret == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


@pytest.mark.asyncio
async def test_chunk_generator_raises():
    async def generator_raises():
        yield 42
        raise SampleException("foo")

    ret = await unchunk_generator(chunk_generator(generator_raises(), 0.33))
    assert ret == [[42, "exc"]]


@pytest.mark.asyncio
async def test_task_context():
    async with TaskContext() as task_context:
        t = task_context.create_task(asyncio.sleep(0.1))
        assert not t.done()
        # await asyncio.sleep(0.0)
    await asyncio.sleep(0.0)  # just waste a loop step for the cancellation to go through
    assert t.cancelled()


@pytest.mark.asyncio
async def test_task_context_grace():
    async with TaskContext(grace=0.2) as task_context:
        u = task_context.create_task(asyncio.sleep(0.1))
        v = task_context.create_task(asyncio.sleep(0.3))
        assert not u.done()
        assert not v.done()
    await asyncio.sleep(0.0)
    assert u.done()
    assert v.cancelled()


async def raise_exception():
    raise SampleException("foo")


@skip_non_linux
@pytest.mark.asyncio
async def test_task_context_wait():
    async with TaskContext(grace=0.1) as task_context:
        u = task_context.create_task(asyncio.sleep(1.1))
        v = task_context.create_task(asyncio.sleep(1.3))
        await task_context.wait(u)

    assert u.done()
    assert v.cancelled()

    with pytest.raises(SampleException):
        async with TaskContext(grace=0.2) as task_context:
            u = task_context.create_task(asyncio.sleep(1.1))
            v = task_context.create_task(raise_exception())
            await task_context.wait(u)

    assert u.cancelled()
    assert v.done()


@skip_non_linux
@pytest.mark.asyncio
async def test_task_context_infinite_loop():
    async with TaskContext(grace=0.01) as task_context:
        counter = 0

        async def f():
            nonlocal counter
            counter += 1

        t = task_context.infinite_loop(f, sleep=0.1)
        assert not t.done()
        await asyncio.sleep(0.35)
        assert counter == 4  # at 0.00, 0.10, 0.20, 0.30
    await asyncio.sleep(0.0)  # just waste a loop step for the cancellation to go through
    assert not t.cancelled()
    assert t.done()
    assert counter == 4  # should be exited immediately


DEBOUNCE_TIME = 0.1


@pytest.mark.asyncio
async def test_queue_batch_iterator():
    queue = asyncio.Queue()
    await queue.put(1)
    drained_items = []

    async def drain_queue(logs_queue):
        async for batch in queue_batch_iterator(logs_queue, debounce_time=DEBOUNCE_TIME):
            drained_items.extend(batch)

    async with TaskContext(grace=0.0) as tc:
        tc.create_task(drain_queue(queue))

        # Make sure the queue gets drained.
        await asyncio.sleep(0.001)

        assert len(drained_items) == 1

        # Add items to the queue and a sentinel while it's still waiting for DEBOUNCE_TIME.
        await queue.put(2)
        await queue.put(3)
        await queue.put(None)

        await asyncio.sleep(DEBOUNCE_TIME + 0.001)

        assert len(drained_items) == 3
