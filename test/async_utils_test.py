# Copyright Modal Labs 2022
import asyncio
import functools
import logging
import os
import platform
import pytest

import pytest_asyncio
from synchronicity import Synchronizer

from modal._utils import async_utils
from modal._utils.async_utils import (
    TaskContext,
    aclosing,
    async_concat,
    async_merge,
    async_zip,
    callable_to_agen,
    queue_batch_iterator,
    retry,
    sync_or_async_iter,
    synchronize_api,
    warn_if_generator_is_not_consumed,
)


@pytest_asyncio.fixture(autouse=True)
async def no_dangling_tasks():
    yield
    assert not asyncio.all_tasks() - {asyncio.tasks.current_task()}


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

    # test that things are cleaned up when we fully exhaust the generator
    async with aclosing(foo()) as stream:
        async for it in stream:
            result.append(it)

    assert sorted(result) == [1, 2]
    assert states == ["enter", "exit"]

    # test that things are cleaned up when we exit the context manager without fully exhausting the generator
    states.clear()
    result.clear()
    async with aclosing(foo()) as stream:
        async for it in stream:
            break

    assert result == []
    assert states == ["enter", "exit"]


@pytest.mark.asyncio
async def test_sync_or_async_iter_sync_gen():
    result = []

    def sync_gen():
        yield 4
        yield 5
        yield 6

    async for i in sync_or_async_iter(sync_gen()):
        result.append(i)
    assert result == [4, 5, 6]


@pytest.mark.asyncio
async def test_sync_or_async_iter_async_gen():
    result = []
    states = []

    async def async_gen():
        states.append("enter")
        try:
            yield 1
            await asyncio.sleep(0.1)
            yield 2
            await asyncio.sleep(0.1)
            yield 3
        finally:
            states.append("exit")

    # test that things are cleaned up when we fully exhaust the generator
    async for i in sync_or_async_iter(async_gen()):
        result.append(i)
    assert result == [1, 2, 3]
    assert states == ["enter", "exit"]

    # test that things are cleaned up when we exit the context manager without fully exhausting the generator
    result.clear()
    states.clear()
    async with aclosing(async_gen()) as agen, aclosing(sync_or_async_iter(agen)) as stream:
        async for _ in stream:
            break
    assert states == ["enter", "exit"]
    assert result == []


@pytest.mark.asyncio
async def test_async_zip():
    states = []
    result = []

    async def gen(x):
        states.append(f"enter {x}")
        try:
            await asyncio.sleep(0.1)
            yield x
            yield x + 1
        finally:
            await asyncio.sleep(0)
            states.append(f"exit {x}")

    async with aclosing(gen(1)) as g1, aclosing(gen(5)) as g2, aclosing(gen(10)) as g3, aclosing(
        async_zip(g1, g2, g3)
    ) as stream:
        async for item in stream:
            result.append(item)

    assert result == [(1, 5, 10), (2, 6, 11)]
    assert states == ["enter 1", "enter 5", "enter 10", "exit 1", "exit 5", "exit 10"]


@pytest.mark.asyncio
async def test_async_zip_different_lengths():
    states = []
    result = []

    async def gen_short():
        states.append("enter short")
        try:
            await asyncio.sleep(0.1)
            yield 1
            yield 2
        finally:
            await asyncio.sleep(0)
            states.append("exit short")

    async def gen_long():
        states.append("enter long")
        try:
            await asyncio.sleep(0.1)
            yield 3
            yield 4
            yield 5
            yield 6

        finally:
            await asyncio.sleep(0)
            states.append("exit long")

    async with aclosing(gen_short()) as g1, aclosing(gen_long()) as g2, aclosing(async_zip(g1, g2)) as stream:
        async for item in stream:
            result.append(item)

    assert result == [(1, 3), (2, 4)]
    assert states == ["enter short", "enter long", "exit short", "exit long"]


@pytest.mark.asyncio
async def test_async_zip_exception():
    states = []
    result = []

    async def gen(x):
        states.append(f"enter {x}")
        try:
            await asyncio.sleep(0.1)
            yield x
            if x == 1:
                raise SampleException("test")
            yield x + 1
        finally:
            await asyncio.sleep(0)
            states.append(f"exit {x}")

    with pytest.raises(SampleException):
        async with aclosing(gen(1)) as g1, aclosing(gen(5)) as g2, aclosing(async_zip(g1, g2)) as stream:
            async for item in stream:
                result.append(item)

    assert result == [(1, 5)]
    assert states == ["enter 1", "enter 5", "exit 1", "exit 5"]


@pytest.mark.asyncio
async def test_async_zip_cancellation():
    ev = asyncio.Event()

    async def gen1():
        await asyncio.sleep(0.1)
        yield 1
        await ev.wait()
        raise asyncio.CancelledError()
        yield 2

    async def gen2():
        yield 3
        await asyncio.sleep(0.1)
        yield 4

    async def zip_coro():
        async for _ in async_zip(gen1(), gen2()):
            pass

    zip_task = asyncio.create_task(zip_coro())
    await asyncio.sleep(0.1)
    zip_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await zip_task


@pytest.mark.asyncio
async def test_async_zip_producer_cancellation():
    async def gen1():
        await asyncio.sleep(0.1)
        yield 1
        raise asyncio.CancelledError()
        yield 2

    async def gen2():
        yield 3
        await asyncio.sleep(0.1)
        yield 4

    await asyncio.sleep(0.1)
    with pytest.raises(asyncio.CancelledError):
        async for _ in async_zip(gen1(), gen2()):
            pass


@pytest.mark.asyncio
async def test_async_zip_parallel():
    ev1 = asyncio.Event()
    ev2 = asyncio.Event()

    async def gen1():
        await asyncio.sleep(0.1)
        ev1.set()
        yield 1
        await ev2.wait()
        yield 2

    async def gen2():
        await ev1.wait()
        yield 3
        await asyncio.sleep(0.1)
        ev2.set()
        yield 4

    result = []
    async for item in async_zip(gen1(), gen2()):
        result.append(item)

    assert result == [(1, 3), (2, 4)]


@pytest.mark.asyncio
async def test_async_merge():
    result = []
    states = []

    ev1 = asyncio.Event()
    ev2 = asyncio.Event()

    async def gen1():
        states.append("gen1 enter")
        try:
            await asyncio.sleep(0.1)
            yield 1
            ev1.set()
            await ev2.wait()
            yield 2
        finally:
            await asyncio.sleep(0)
            states.append("gen1 exit")

    async def gen2():
        states.append("gen2 enter")
        try:
            await ev1.wait()
            yield 3
            await asyncio.sleep(0.1)
            ev2.set()
            yield 4
        finally:
            await asyncio.sleep(0)
            states.append("gen2 exit")

    async for item in async_merge(gen1(), gen2()):
        result.append(item)

    assert result == [1, 3, 4, 2]
    assert states == [
        "gen1 enter",
        "gen2 enter",
        "gen2 exit",
        "gen1 exit",
    ]


@pytest.mark.asyncio
async def test_async_merge_cleanup():
    states = []

    ev1 = asyncio.Event()
    ev2 = asyncio.Event()

    async def gen1():
        states.append("gen1 enter")
        try:
            await asyncio.sleep(0.1)
            yield 1
            ev1.set()
            await ev2.wait()
            yield 2
        finally:
            await asyncio.sleep(0)
            states.append("gen1 exit")

    async def gen2():
        states.append("gen2 enter")
        try:
            await ev1.wait()
            yield 3
            await asyncio.sleep(0.1)
            ev2.set()
            yield 4
        finally:
            await asyncio.sleep(0)
            states.append("gen2 exit")

    async with aclosing(gen1()) as g1, aclosing(gen2()) as g2, aclosing(async_merge(g1, g2)) as stream:
        async for _ in stream:
            break

    assert states == [
        "gen1 enter",
        "gen2 enter",
        "gen2 exit",
        "gen1 exit",
    ]


@pytest.mark.asyncio
async def test_async_merge_exception():
    result = []
    states = []

    async def gen1():
        states.append("gen1 enter")
        try:
            await asyncio.sleep(0.1)
            yield 1
            raise SampleException("test")
        finally:
            await asyncio.sleep(0)
            states.append("gen1 exit")

    async def gen2():
        states.append("gen2 enter")
        try:
            yield 3
            await asyncio.sleep(0.1)
            yield 4
        finally:
            await asyncio.sleep(0)
            states.append("gen2 exit")

    with pytest.raises(SampleException):
        async for item in async_merge(gen1(), gen2()):
            result.append(item)

    assert sorted(result) == [1, 3, 4]
    assert sorted(states) == [
        "gen1 enter",
        "gen1 exit",
        "gen2 enter",
        "gen2 exit",
    ]


@pytest.mark.asyncio
async def test_async_merge_cancellation():
    ev = asyncio.Event()

    async def gen1():
        await asyncio.sleep(0.1)
        yield 1
        await ev.wait()
        yield 2

    async def gen2():
        yield 3
        await asyncio.sleep(0.1)
        yield 4

    async def merge_coro():
        async for _ in async_merge(gen1(), gen2()):
            pass

    merge_task = asyncio.create_task(merge_coro())
    await asyncio.sleep(0.1)
    merge_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await merge_task


@pytest.mark.asyncio
async def test_async_merge_producer_cancellation():
    async def gen1():
        await asyncio.sleep(0.1)
        yield 1
        raise asyncio.CancelledError()
        yield 2

    async def gen2():
        yield 3
        await asyncio.sleep(0.1)
        yield 4

    await asyncio.sleep(0.1)
    with pytest.raises(asyncio.CancelledError):
        async for _ in async_merge(gen1(), gen2()):
            pass


@pytest.mark.asyncio
async def test_callable_to_agen():
    async def foo():
        await asyncio.sleep(0.1)
        return 42

    result = []
    async for item in callable_to_agen(foo):
        result.append(item)
    assert result == [await foo()]


@pytest.mark.asyncio
async def test_async_concat():
    async def gen1():
        await asyncio.sleep(0.1)
        yield 1
        yield 2

    async def gen2():
        yield 3
        await asyncio.sleep(0.1)
        yield 4

    async def gen3():
        yield 5
        yield 6

    result = []
    async for item in async_concat(gen1(), gen2(), gen3()):
        result.append(item)

    assert result == [1, 2, 3, 4, 5, 6]


@pytest.mark.asyncio
async def test_async_concat_exception():
    # test exception bubbling up
    result = []
    states = []

    async def gen1():
        states.append("enter 1")
        try:
            yield 1
            yield 2
        finally:
            states.append("exit 1")

    async def gen2():
        states.append("enter 2")
        try:
            await asyncio.sleep(0.1)
            yield 3
            raise SampleException("test")
            yield 4
        finally:
            await asyncio.sleep(0)
            states.append("exit 2")

    with pytest.raises(SampleException):
        async for item in async_concat(gen1(), gen2()):
            result.append(item)

    assert result == [1, 2, 3]
    assert states == ["enter 1", "exit 1", "enter 2", "exit 2"]


@pytest.mark.asyncio
async def test_async_concat_cancellation():
    # test asyncio cancellation bubbles up

    async def gen1():
        await asyncio.sleep(0.1)
        yield 1
        raise asyncio.CancelledError()
        yield 2

    async def gen2():
        yield 3
        await asyncio.sleep(0.1)
        yield 4

    with pytest.raises(asyncio.CancelledError):
        async for _ in async_concat(gen1(), gen2()):
            pass


@pytest.mark.asyncio
async def test_async_concat_cleanup():
    # test cleanup of generators
    result = []
    states = []

    async def gen1():
        states.append("enter 1")
        try:
            await asyncio.sleep(0.1)
            yield 1
            yield 2
        finally:
            await asyncio.sleep(0)
            states.append("exit 1")

    async def gen2():
        states.append("enter 2")
        try:
            yield 3
            await asyncio.sleep(0.1)
            yield 4
        finally:
            await asyncio.sleep(0)
            states.append("exit 2")

    async with aclosing(gen1()) as g1, aclosing(gen2()) as g2, aclosing(async_concat(g1, g2)) as stream:
        async for item in stream:
            result.append(item)
            if item == 3:
                break

    assert result == [1, 2, 3]
    assert states == ["enter 1", "exit 1", "enter 2", "exit 2"]
