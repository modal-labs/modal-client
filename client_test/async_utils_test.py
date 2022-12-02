# Copyright Modal Labs 2022
import asyncio
import logging
import os
import platform
import pytest

from synchronicity import Interface, Synchronizer

from modal_utils import async_utils
from modal_utils.async_utils import (
    TaskContext,
    queue_batch_iterator,
    retry,
    warn_if_generator_is_not_consumed,
)

skip_github_non_linux = pytest.mark.skipif(
    (os.environ.get("GITHUB_ACTIONS") == "true" and platform.system() != "Linux"),
    reason="sleep is inaccurate on GitHub Actions runners.",
)


class SampleException(Exception):
    pass


class FailNTimes:
    def __init__(self, n_failures, exc=SampleException("Something bad happened")):
        self.n_failures = n_failures
        self.n_calls = 0
        self.exc = exc

    async def __call__(self, x):
        self.n_calls += 1
        if self.n_calls <= self.n_failures:
            raise self.exc
        else:
            return x + 1


@pytest.mark.asyncio
async def test_retry():
    f_retry = retry(FailNTimes(2))
    assert await f_retry(42) == 43

    with pytest.raises(SampleException):
        f_retry = retry(FailNTimes(3))
        assert await f_retry(42) == 43

    f_retry = retry(n_attempts=5)(FailNTimes(4))
    assert await f_retry(42) == 43

    with pytest.raises(SampleException):
        f_retry = retry(n_attempts=5)(FailNTimes(5))
        assert await f_retry(42) == 43


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


@skip_github_non_linux
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


@skip_github_non_linux
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
    queue: asyncio.Queue = asyncio.Queue()
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


@pytest.mark.asyncio
async def test_warn_if_generator_is_not_consumed(caplog):
    @warn_if_generator_is_not_consumed
    async def my_generator():
        yield 42

    with caplog.at_level(logging.WARNING):
        g = my_generator()
        assert "my_generator" in repr(g)
        del g  # Force destructor

    assert len(caplog.records) == 1
    assert "my_generator" in caplog.text
    assert "for" in caplog.text
    assert "list" in caplog.text


def test_exit_handler():
    result = None
    sync = Synchronizer()

    async def cleanup():
        nonlocal result
        result = "bye"

    async def _setup_code():
        async_utils.on_shutdown(cleanup())

    setup_code = sync.create(_setup_code)[Interface.BLOCKING]
    setup_code()

    sync._close_loop()  # this is called on exit by synchronicity, which shuts down the event loop
    assert result == "bye"
