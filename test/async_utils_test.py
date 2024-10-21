# Copyright Modal Labs 2022
import asyncio
import functools
import logging
import os
import platform
import pytest

from synchronicity import Synchronizer

from modal._utils import async_utils
from modal._utils.async_utils import (
    TaskContext,
    aclosing,
    async_merge,
    async_zip,
    callable_to_agen,
    queue_batch_iterator,
    retry,
    sync_or_async_iter,
    synchronize_api,
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


@skip_github_non_linux
@pytest.mark.asyncio
async def test_task_context_infinite_loop_non_functions():
    async with TaskContext(grace=0.01) as task_context:

        async def f(x):
            pass

        task_context.infinite_loop(lambda: f(123))
        task_context.infinite_loop(functools.partial(f, 123))


@skip_github_non_linux
@pytest.mark.asyncio
async def test_task_context_infinite_loop_timeout(caplog):
    async with TaskContext(grace=0.01) as task_context:

        async def f():
            await asyncio.sleep(5.0)

        task_context.infinite_loop(f, timeout=0.1)
        await asyncio.sleep(0.15)

    # TODO(elias): Find the tests that leak `Task was destroyed but it is pending` warnings into this test
    # so we can assert a single record here:
    # assert len(caplog.records) == 1
    for record in caplog.records:
        if "timed out" in caplog.text:
            break
    else:
        assert False, "no timeout"


@pytest.mark.asyncio
async def test_task_context_gather():
    state = "none"

    async def t1(error=False):
        nonlocal state
        await asyncio.sleep(0.1)
        state = "t1"
        if error:
            raise ValueError()

    async def t2():
        nonlocal state
        await asyncio.sleep(0.2)
        state = "t2"

    await asyncio.gather(t1(), t2())
    assert state == "t2"

    # On t1 error: asyncio.gather() does not cancel t2, which is bad behavior.
    state = "none"
    with pytest.raises(ValueError):
        await asyncio.gather(t1(error=True), t2())
    assert state == "t1"
    await asyncio.sleep(0.2)
    assert state == "t2"  # t2 still runs because asyncio.gather() does not cancel tasks

    # On t1 error: TaskContext.gather() should cancel the remaining tasks.
    state = "none"
    with pytest.raises(ValueError):
        await TaskContext.gather(t1(error=True), t2())
    assert state == "t1"
    await asyncio.sleep(0.2)
    assert state == "t1"


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
    @warn_if_generator_is_not_consumed()
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


def test_warn_if_generator_is_not_consumed_sync(caplog):
    @warn_if_generator_is_not_consumed()
    def my_generator():
        yield 42

    with caplog.at_level(logging.WARNING):
        g = my_generator()
        assert "my_generator" in repr(g)
        del g  # Force destructor

    assert len(caplog.records) == 1
    assert "my_generator" in caplog.text
    assert "for" in caplog.text
    assert "list" in caplog.text


@pytest.mark.asyncio
async def test_no_warn_if_generator_is_consumed(caplog):
    @warn_if_generator_is_not_consumed()
    async def my_generator():
        yield 42

    with caplog.at_level(logging.WARNING):
        g = my_generator()
        async for _ in g:
            pass
        del g  # Force destructor

    assert len(caplog.records) == 0


def test_exit_handler():
    result = None
    sync = Synchronizer()

    async def cleanup():
        nonlocal result
        result = "bye"

    async def _setup_code():
        async_utils.on_shutdown(cleanup())

    setup_code = sync.create_blocking(_setup_code)
    setup_code()

    sync._close_loop()  # this is called on exit by synchronicity, which shuts down the event loop
    assert result == "bye"


def test_synchronize_api_blocking_name():
    class _MyClass:
        async def foo(self):
            await asyncio.sleep(0.1)
            return "bar"

    async def _myfunc():
        await asyncio.sleep(0.1)
        return "bar"

    MyClass = synchronize_api(_MyClass)
    assert MyClass.__name__ == "MyClass"
    assert MyClass().foo() == "bar"

    myfunc = synchronize_api(_myfunc)
    assert myfunc.__name__ == "myfunc"
    assert myfunc() == "bar"


@pytest.mark.asyncio
async def test_aclosing():
    result = []
    states = []

    async def foo():
        states.append("enter")
        try:
            yield 1
            yield 2
        finally:
            states.append("exit")

    async with aclosing(foo()) as stream:
        async for it in stream:
            result.append(it)

    assert sorted(result) == [1, 2]
    assert states == ["enter", "exit"]

    states.clear()
    result.clear()
    async with aclosing(foo()) as stream:
        async for it in stream:
            break

    assert result == []
    assert states == ["enter", "exit"]


@pytest.mark.asyncio
async def test_sync_or_async_iter():
    async def async_gen():
        yield 1
        await asyncio.sleep(0.1)
        yield 2
        await asyncio.sleep(0.1)
        yield 3

    def sync_gen():
        yield 4
        yield 5
        yield 6

    res = []
    async for i in sync_or_async_iter(async_gen()):
        res.append(i)
    assert res == [1, 2, 3]

    res = []
    async for i in sync_or_async_iter(sync_gen()):
        res.append(i)
    assert res == [4, 5, 6]


@pytest.mark.asyncio
async def test_async_zip():
    async def gen1(start, count=1):
        for i in range(start, start + count):
            yield i

    result = []
    async for item in async_zip(gen1(1), gen1(4), gen1(6)):
        result.append(item)

    assert result == [(1, 4, 6)]

    result = []
    async for item in async_zip(gen1(1, 10), gen1(5, 8), gen1(6, 2)):
        result.append(item)

    assert result == [(1, 5, 6), (2, 6, 7)]


@pytest.mark.asyncio
async def test_async_zip_parallel():
    ev = asyncio.Event()
    ev2 = asyncio.Event()

    async def gen1():
        ev.set()
        await ev2.wait()
        yield 1

    async def gen2():
        await ev.wait()
        ev2.set()
        yield 2

    result = []
    async for item in async_zip(gen1(), gen2()):
        result.append(item)

    assert result == [(1, 2)]


@pytest.mark.asyncio
async def test_async_merge():
    result = []
    states = []

    gen1_event = asyncio.Event()
    gen2_event = asyncio.Event()
    gen3_event = asyncio.Event()
    gen4_event = asyncio.Event()

    async def gen1():
        states.append("gen1 enter")
        try:
            gen1_event.set()
            await gen2_event.wait()
            yield 1
            gen3_event.set()
            await gen4_event.wait()
            yield 2
        finally:
            states.append("gen1 exit")

    async def gen2():
        states.append("gen2 enter")
        try:
            await gen1_event.wait()
            yield 3
            gen2_event.set()
            await gen3_event.wait()
            yield 4
            gen4_event.set()
        finally:
            states.append("gen2 exit")

    async for item in async_merge(gen1(), gen2()):
        result.append(item)

    assert result == [3, 1, 4, 2]
    assert sorted(states) == [
        "gen1 enter",
        "gen1 exit",
        "gen2 enter",
        "gen2 exit",
    ]

    result.clear()
    states.clear()

    async for item in async_merge(gen1(), gen2()):
        break

    assert result == []
    assert sorted(states) == [
        "gen1 enter",
        "gen1 exit",
        "gen2 enter",
        "gen2 exit",
    ]


@pytest.mark.asyncio
async def test_awaitable_to_aiter():
    async def foo():
        await asyncio.sleep(0.1)
        return 42

    result = []
    async for item in callable_to_agen(foo):
        result.append(item)
    assert result == [await foo()]
