# Copyright Modal Labs 2022
import asyncio
import concurrent.futures
import functools
import inspect
import itertools
import time
import typing
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Iterable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
    cast,
)

import synchronicity
from synchronicity.async_utils import Runner
from synchronicity.exceptions import NestedEventLoops
from typing_extensions import ParamSpec, assert_type

from ..exception import InvalidError
from .logger import logger

T = TypeVar("T")
P = ParamSpec("P")
V = TypeVar("V")

synchronizer = synchronicity.Synchronizer()


def synchronize_api(obj, target_module=None):
    if inspect.isclass(obj) or inspect.isfunction(obj):
        blocking_name = obj.__name__.lstrip("_")
    elif isinstance(obj, TypeVar):
        blocking_name = "_BLOCKING_" + obj.__name__
    else:
        blocking_name = None
    if target_module is None:
        target_module = obj.__module__
    return synchronizer.create_blocking(obj, blocking_name, target_module=target_module)


def retry(direct_fn=None, *, n_attempts=3, base_delay=0, delay_factor=2, timeout=90):
    """Decorator that calls an async function multiple times, with a given timeout.

    If a `base_delay` is provided, the function is given an exponentially
    increasing delay on each run, up until the maximum number of attempts.

    Usage:

    ```
    @retry
    async def may_fail_default():
        # ...
        pass

    @retry(n_attempts=5, base_delay=1)
    async def may_fail_delay():
        # ...
        pass
    ```
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def f_wrapped(*args, **kwargs):
            delay = base_delay
            for i in range(n_attempts):
                t0 = time.time()
                try:
                    return await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout)
                except asyncio.CancelledError:
                    logger.debug(f"Function {fn} was cancelled")
                    raise
                except Exception as e:
                    if i >= n_attempts - 1:
                        raise
                    logger.debug(
                        f"Failed invoking function {fn}: {e}"
                        f" (took {time.time() - t0}s, sleeping {delay}s"
                        f" and trying {n_attempts - i - 1} more times)"
                    )
                await asyncio.sleep(delay)
                delay *= delay_factor

        return f_wrapped

    if direct_fn is not None:
        # It's invoked like @retry
        return decorator(direct_fn)
    else:
        # It's invoked like @retry(n_attempts=...)
        return decorator


class TaskContext:
    """A structured group that helps manage stray tasks.

    This differs from the standard library `asyncio.TaskGroup` in that it cancels all tasks still
    running after exiting the context manager, rather than waiting for them to finish.

    A `TaskContext` can have an optional `grace` period in seconds, which will wait for a certain
    amount of time before cancelling all remaining tasks. This is useful for allowing tasks to
    gracefully exit when they determine that the context is shutting down.

    Usage:

    ```python notest
    async with TaskContext() as task_context:
        task = task_context.create_task(coro())
    ```
    """

    _loops: set[asyncio.Task]

    def __init__(self, grace: Optional[float] = None):
        self._grace = grace
        self._loops = set()

    async def start(self):
        # TODO: this only exists as a standalone method because Client doesn't have a proper ctx mgr
        self._tasks: set[asyncio.Task] = set()
        self._exited: asyncio.Event = asyncio.Event()  # Used to stop infinite loops

    @property
    def exited(self) -> bool:
        return self._exited.is_set()

    async def __aenter__(self):
        await self.start()
        return self

    async def stop(self):
        self._exited.set()
        await asyncio.sleep(0)  # Causes any just-created tasks to get started
        unfinished_tasks = [t for t in self._tasks if not t.done()]
        gather_future = None
        try:
            if self._grace is not None and unfinished_tasks:
                gather_future = asyncio.gather(*unfinished_tasks, return_exceptions=True)
                await asyncio.wait_for(gather_future, timeout=self._grace)
        except asyncio.TimeoutError:
            pass
        finally:
            # asyncio.wait_for cancels the future, but the CancelledError
            # still needs to be handled
            # (https://stackoverflow.com/a/63356323/2475114)
            if gather_future:
                try:
                    await gather_future
                except asyncio.CancelledError:
                    pass

            for task in self._tasks:
                if task.done() and not task.cancelled():
                    # Raise any exceptions if they happened.
                    # Only tasks without a done_callback will still be present in self._tasks
                    task.result()

                if task.done() or task in self._loops:  # Note: Legacy code, we can probably cancel loops.
                    continue

                # Cancel any remaining unfinished tasks.
                task.cancel()
            await asyncio.sleep(0)  # wake up coroutines waiting for cancellations

    async def __aexit__(self, exc_type, value, tb):
        await self.stop()

    def create_task(self, coro_or_task) -> asyncio.Task:
        if isinstance(coro_or_task, asyncio.Task):
            task = coro_or_task
        elif asyncio.iscoroutine(coro_or_task):
            loop = asyncio.get_event_loop()
            task = loop.create_task(coro_or_task)
        else:
            raise Exception(f"Object of type {type(coro_or_task)} is not a coroutine or Task")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def infinite_loop(
        self, async_f, timeout: Optional[float] = 90, sleep: float = 10, log_exception: bool = True
    ) -> asyncio.Task:
        if isinstance(async_f, functools.partial):
            function_name = async_f.func.__qualname__
        else:
            function_name = async_f.__qualname__

        async def loop_coro() -> None:
            logger.debug(f"Starting infinite loop {function_name}")
            while not self.exited:
                try:
                    await asyncio.wait_for(async_f(), timeout=timeout)
                except Exception as exc:
                    if log_exception and isinstance(exc, asyncio.TimeoutError):
                        # Asyncio sends an empty message in this case, so let's use logger.error
                        logger.error(f"Loop attempt for {function_name} timed out")
                    elif log_exception:
                        # Propagate the exception to the logger
                        logger.exception(f"Loop attempt for {function_name} failed")
                try:
                    await asyncio.wait_for(self._exited.wait(), timeout=sleep)
                except asyncio.TimeoutError:
                    continue

            logger.debug(f"Exiting infinite loop for {function_name}")

        t = self.create_task(loop_coro())
        t.set_name(f"{function_name} loop")
        self._loops.add(t)
        t.add_done_callback(self._loops.discard)
        return t

    @staticmethod
    async def gather(*coros: Awaitable) -> Any:
        """Wait for a sequence of coroutines to finish, concurrently.

        This is similar to `asyncio.gather()`, but it uses TaskContext to cancel all remaining tasks
        if one fails with an exception other than `asyncio.CancelledError`. The native `asyncio`
        function does not cancel remaining tasks in this case, which can lead to surprises.

        For example, if you use `asyncio.gather(t1, t2, t3)` and t2 raises an exception, then t1 and
        t3 would continue running. With `TaskContext.gather(t1, t2, t3)`, they are cancelled.

        (It's still acceptable to use `asyncio.gather()` if you don't need cancellation — for
        example, if you're just gathering quick coroutines with no side-effects. Or if you're
        gathering the tasks with `return_exceptions=True`.)

        Usage:

        ```python notest
        # Example 1: Await three coroutines
        created_object, other_work, new_plumbing = await TaskContext.gather(
            create_my_object(),
            do_some_other_work(),
            fix_plumbing(),
        )

        # Example 2: Gather a list of coroutines
        coros = [a.load() for a in objects]
        results = await TaskContext.gather(*coros)
        ```
        """
        async with TaskContext() as tc:
            results = await asyncio.gather(*(tc.create_task(coro) for coro in coros))
        return results


def run_coro_blocking(coro):
    """Fairly hacky thing that's needed in some extreme cases.

    It's basically works like asyncio.run but unlike asyncio.run it also works
    with in the case an event loop is already running. It does this by basically
    moving the whole thing to a separate thread.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        fut = executor.submit(asyncio.run, coro)
        return fut.result()


class TimestampPriorityQueue(Generic[T]):
    """
    A priority queue that schedules items to be processed at specific timestamps.
    """

    _MAX_PRIORITY = float("inf")

    def __init__(self, maxsize: int = 0):
        self.condition = asyncio.Condition()
        self._queue: asyncio.PriorityQueue[tuple[float, Union[T, None]]] = asyncio.PriorityQueue(maxsize=maxsize)

    async def close(self):
        await self.put(self._MAX_PRIORITY, None)

    async def put(self, timestamp: float, item: Union[T, None]):
        """
        Add an item to the queue to be processed at a specific timestamp.
        """
        await self._queue.put((timestamp, item))
        async with self.condition:
            self.condition.notify_all()  # notify any waiting coroutines

    async def get(self) -> Union[T, None]:
        """
        Get the next item from the queue that is ready to be processed.
        """
        while True:
            async with self.condition:
                while self.empty():
                    await self.condition.wait()
                # peek at the next item
                timestamp, item = await self._queue.get()
                now = time.time()
                if timestamp < now:
                    return item
                if timestamp == self._MAX_PRIORITY:
                    return None
                # not ready yet, calculate sleep time
                sleep_time = timestamp - now
                self._queue.put_nowait((timestamp, item))  # put it back
                # wait until either the timeout or a new item is added
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=sleep_time)
                except asyncio.TimeoutError:
                    continue

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    async def clear(self):
        """
        Clear the retry queue. Used for testing to simulate reading all elements from queue using queue_batch_iterator.
        """
        while not self.empty():
            await self.get()

    def __len__(self):
        return self._queue.qsize()


async def queue_batch_iterator(
    q: Union[asyncio.Queue, TimestampPriorityQueue], max_batch_size=100, debounce_time=0.015
):
    """
    Read from a queue but return lists of items when queue is large

    Treats a None value as end of queue items
    """
    item_list: list[Any] = []

    while True:
        if q.empty() and len(item_list) > 0:
            yield item_list
            item_list = []
            await asyncio.sleep(debounce_time)

        res = await q.get()

        if len(item_list) >= max_batch_size:
            yield item_list
            item_list = []

        if res is None:
            if len(item_list) > 0:
                yield item_list
            break
        item_list.append(res)


class _WarnIfGeneratorIsNotConsumed:
    def __init__(self, gen, function_name: str):
        self.gen = gen
        self.function_name = function_name
        self.iterated = False
        self.warned = False

    def __aiter__(self):
        self.iterated = True
        return self.gen.__aiter__()

    async def __anext__(self):
        self.iterated = True
        return await self.gen.__anext__()

    async def asend(self, value):
        self.iterated = True
        return await self.gen.asend(value)

    def __repr__(self):
        return repr(self.gen)

    def __del__(self):
        if not self.iterated and not self.warned:
            self.warned = True
            logger.warning(
                f"Warning: the results of a call to {self.function_name} was not consumed, "
                "so the call will never be executed."
                f" Consider a for-loop like `for x in {self.function_name}(...)` or "
                "unpacking the generator using `list(...)`"
            )

    async def athrow(self, exc):
        return await self.gen.athrow(exc)

    async def aclose(self):
        return await self.gen.aclose()


synchronize_api(_WarnIfGeneratorIsNotConsumed)


class _WarnIfNonWrappedGeneratorIsNotConsumed(_WarnIfGeneratorIsNotConsumed):
    # used for non-synchronicity-wrapped generators and iterators
    def __iter__(self):
        self.iterated = True
        return iter(self.gen)

    def __next__(self):
        self.iterated = True
        return self.gen.__next__()

    def send(self, value):
        self.iterated = True
        return self.gen.send(value)


def warn_if_generator_is_not_consumed(function_name: Optional[str] = None):
    # https://gist.github.com/erikbern/01ae78d15f89edfa7f77e5c0a827a94d
    def decorator(gen_f):
        presented_func_name = function_name if function_name is not None else gen_f.__name__

        @functools.wraps(gen_f)
        def f_wrapped(*args, **kwargs):
            gen = gen_f(*args, **kwargs)
            if inspect.isasyncgen(gen):
                return _WarnIfGeneratorIsNotConsumed(gen, presented_func_name)
            else:
                return _WarnIfNonWrappedGeneratorIsNotConsumed(gen, presented_func_name)

        return f_wrapped

    return decorator


class AsyncOrSyncIterable:
    """Compatibility class for non-synchronicity wrapped async iterables to get
    both async and sync interfaces in the same way that synchronicity does (but on the main thread)
    so they can be "lazily" iterated using either `for _ in x` or `async for _ in x`

    nested_async_message is raised as an InvalidError if the async variant is called
    from an already async context, since that would otherwise deadlock the event loop
    """

    def __init__(self, async_iterable: typing.AsyncGenerator[Any, None], nested_async_message):
        self._async_iterable = async_iterable
        self.nested_async_message = nested_async_message

    def __aiter__(self):
        return self._async_iterable

    def __iter__(self):
        try:
            with Runner() as runner:
                yield from run_async_gen(runner, self._async_iterable)
        except NestedEventLoops:
            raise InvalidError(self.nested_async_message)

    async def aclose(self):
        if hasattr(self._async_iterable, "aclose"):
            await self._async_iterable.aclose()


_shutdown_tasks = []


def on_shutdown(coro):
    # hook into event loop shutdown when all active tasks get cancelled
    async def wrapper():
        try:
            await asyncio.sleep(1e10)  # never awake except for exceptions
        finally:
            await coro
            raise

    _shutdown_tasks.append(asyncio.create_task(wrapper()))


def asyncify(f: Callable[P, T]) -> Callable[P, typing.Coroutine[None, None, T]]:
    """Convert a blocking function into one that runs in the current loop's executor."""

    @functools.wraps(f)
    async def wrapper(*args: P.args, **kwargs: P.kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(f, *args, **kwargs))

    return wrapper


async def iterate_blocking(iterator: Iterator[T]) -> AsyncGenerator[T, None]:
    """Iterate over a blocking iterator in an async context."""

    loop = asyncio.get_running_loop()
    DONE = object()
    while True:
        obj = await loop.run_in_executor(None, next, iterator, DONE)
        if obj is DONE:
            break
        yield cast(T, obj)


@asynccontextmanager
async def asyncnullcontext(*args, **kwargs):
    """Async noop context manager.

    Note that for Python 3.10+ you can use contextlib.nullcontext() instead.

    Usage:
    async with asyncnullcontext():
        pass
    """
    yield


YIELD_TYPE = typing.TypeVar("YIELD_TYPE")
SEND_TYPE = typing.TypeVar("SEND_TYPE")


def run_async_gen(
    runner: Runner,
    gen: typing.AsyncGenerator[YIELD_TYPE, SEND_TYPE],
) -> typing.Generator[YIELD_TYPE, SEND_TYPE, None]:
    """Convert an async generator into a sync one"""
    # more or less copied from synchronicity's implementation:
    next_send: typing.Union[SEND_TYPE, None] = None
    next_yield: YIELD_TYPE
    exc: Optional[BaseException] = None
    while True:
        try:
            if exc:
                next_yield = runner.run(gen.athrow(exc))
            else:
                next_yield = runner.run(gen.asend(next_send))  # type: ignore[arg-type]
        except KeyboardInterrupt as e:
            raise e from None
        except StopAsyncIteration:
            break  # typically a graceful exit of the async generator
        try:
            next_send = yield next_yield
            exc = None
        except BaseException as err:
            exc = err


class aclosing(typing.Generic[T]):  # noqa
    # backport of Python contextlib.aclosing from Python 3.10
    def __init__(self, agen: AsyncGenerator[T, None]):
        self.agen = agen

    async def __aenter__(self) -> AsyncGenerator[T, None]:
        return self.agen

    async def __aexit__(self, exc, exc_type, tb):
        await self.agen.aclose()


async def sync_or_async_iter(iter: Union[Iterable[T], AsyncIterable[T]]) -> AsyncGenerator[T, None]:
    if hasattr(iter, "__aiter__"):
        agen = typing.cast(AsyncGenerator[T, None], iter)
        try:
            async for item in agen:
                yield item
        finally:
            if hasattr(agen, "aclose"):
                # All AsyncGenerator's have an aclose method
                # but some AsyncIterable's don't necessarily
                await agen.aclose()
    else:
        assert hasattr(iter, "__iter__"), "sync_or_async_iter requires an Iterable or AsyncGenerator"
        # This intentionally could block the event loop for the duration of calling __iter__ and __next__,
        # so in non-trivial cases (like passing lists and ranges) this could be quite a foot gun for users #
        # w/ async code (but they can work around it by always using async iterators)
        for item in typing.cast(Iterable[T], iter):
            yield item


@typing.overload
def async_zip(g1: AsyncGenerator[T, None], g2: AsyncGenerator[V, None], /) -> AsyncGenerator[tuple[T, V], None]: ...


@typing.overload
def async_zip(*generators: AsyncGenerator[T, None]) -> AsyncGenerator[tuple[T, ...], None]: ...


async def async_zip(*generators):
    tasks = []
    try:
        while True:
            try:

                async def next_item(gen):
                    return await gen.__anext__()

                tasks = [asyncio.create_task(next_item(gen)) for gen in generators]
                items = await asyncio.gather(*tasks)
                yield tuple(items)
            except StopAsyncIteration:
                break
    finally:
        cancelled_tasks = []
        for task in tasks:
            if not task.done():
                task.cancel()
                cancelled_tasks.append(task)
        try:
            await asyncio.gather(*cancelled_tasks)
        except asyncio.CancelledError:
            pass

        first_exception = None
        for gen in generators:
            try:
                await gen.aclose()
            except BaseException as e:
                if first_exception is None:
                    first_exception = e
                logger.exception(f"Error closing async generator: {e}")
        if first_exception is not None:
            raise first_exception


@dataclass
class ValueWrapper(typing.Generic[T]):
    value: T


@dataclass
class ExceptionWrapper:
    value: Exception


class StopSentinelType: ...


STOP_SENTINEL = StopSentinelType()


async def async_merge(*generators: AsyncGenerator[T, None]) -> AsyncGenerator[T, None]:
    """
    Asynchronously merges multiple async generators into a single async generator.

    This function takes multiple async generators and yields their values in the order
    they are produced. If any generator raises an exception, the exception is propagated.

    Args:
        *generators: One or more async generators to be merged.

    Yields:
        The values produced by the input async generators.

    Raises:
        Exception: If any of the input generators raises an exception, it is propagated.

    Usage:
    ```python
    import asyncio
    from modal._utils.async_utils import async_merge

    async def gen1():
        yield 1
        yield 2

    async def gen2():
        yield "a"
        yield "b"

    async def example():
        values = set()
        async for value in async_merge(gen1(), gen2()):
            values.add(value)

        return values

    # Output could be: {1, "a", 2, "b"} (order may vary)
    values = asyncio.run(example())
    assert values == {1, "a", 2, "b"}
    ```
    """
    queue: asyncio.Queue[Union[ValueWrapper[T], ExceptionWrapper]] = asyncio.Queue(maxsize=len(generators) * 10)

    async def producer(generator: AsyncGenerator[T, None]):
        try:
            async for item in generator:
                await queue.put(ValueWrapper(item))
        except Exception as e:
            await queue.put(ExceptionWrapper(e))

    tasks = {asyncio.create_task(producer(gen)) for gen in generators}
    new_output_task = asyncio.create_task(queue.get())

    try:
        while tasks:
            done, _ = await asyncio.wait(
                [*tasks, new_output_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if new_output_task in done:
                item = new_output_task.result()
                if isinstance(item, ValueWrapper):
                    yield item.value
                else:
                    assert_type(item, ExceptionWrapper)
                    raise item.value

                new_output_task = asyncio.create_task(queue.get())

            finished_producers = done & tasks
            tasks -= finished_producers
            for finished_producer in finished_producers:
                # this is done in order to catch potential raised errors/cancellations
                # from within worker tasks as soon as they happen.
                await finished_producer

        while not queue.empty():
            item = await new_output_task
            if isinstance(item, ValueWrapper):
                yield item.value
            else:
                assert_type(item, ExceptionWrapper)
                raise item.value

            new_output_task = asyncio.create_task(queue.get())

    finally:
        if not new_output_task.done():
            new_output_task.cancel()
        for task in tasks:
            if not task.done():
                try:
                    task.cancel()
                    await task
                except asyncio.CancelledError:
                    pass


async def callable_to_agen(awaitable: Callable[[], Awaitable[T]]) -> AsyncGenerator[T, None]:
    yield await awaitable()


async def gather_cancel_on_exc(*coros_or_futures):
    input_tasks = [asyncio.ensure_future(t) for t in coros_or_futures]
    try:
        return await asyncio.gather(*input_tasks)
    except BaseException:
        for t in input_tasks:
            t.cancel()
        await asyncio.gather(*input_tasks, return_exceptions=False)  # handle cancellations
        raise


async def async_map(
    input_generator: AsyncGenerator[T, None],
    async_mapper_func: Callable[[T], Awaitable[V]],
    concurrency: int,
) -> AsyncGenerator[V, None]:
    queue: asyncio.Queue[Union[ValueWrapper[T], StopSentinelType]] = asyncio.Queue(maxsize=concurrency * 2)

    async def producer() -> AsyncGenerator[V, None]:
        async for item in input_generator:
            await queue.put(ValueWrapper(item))

        for _ in range(concurrency):
            await queue.put(STOP_SENTINEL)

        if False:
            # Need it to be an async generator for async_merge
            # but we don't want to yield anything
            yield

    async def worker() -> AsyncGenerator[V, None]:
        while True:
            item = await queue.get()
            if isinstance(item, ValueWrapper):
                yield await async_mapper_func(item.value)
            elif isinstance(item, ExceptionWrapper):
                raise item.value
            else:
                assert_type(item, StopSentinelType)
                break

    async with aclosing(async_merge(*[worker() for _ in range(concurrency)], producer())) as stream:
        async for item in stream:
            yield item


async def async_map_ordered(
    input_generator: AsyncGenerator[T, None],
    async_mapper_func: Callable[[T], Awaitable[V]],
    concurrency: int,
    buffer_size: Optional[int] = None,
) -> AsyncGenerator[V, None]:
    semaphore = asyncio.Semaphore(buffer_size or concurrency)

    async def mapper_func_wrapper(tup: tuple[int, T]) -> tuple[int, V]:
        return (tup[0], await async_mapper_func(tup[1]))

    async def counter() -> AsyncGenerator[int, None]:
        for i in itertools.count():
            await semaphore.acquire()
            yield i

    next_idx = 0
    buffer = {}

    async with aclosing(async_map(async_zip(counter(), input_generator), mapper_func_wrapper, concurrency)) as stream:
        async for output_idx, output_item in stream:
            buffer[output_idx] = output_item

            while next_idx in buffer:
                yield buffer[next_idx]
                semaphore.release()
                del buffer[next_idx]
                next_idx += 1


async def async_chain(*generators: AsyncGenerator[T, None]) -> AsyncGenerator[T, None]:
    try:
        for gen in generators:
            async for item in gen:
                yield item
    finally:
        first_exception = None
        for gen in generators:
            try:
                await gen.aclose()
            except BaseException as e:
                if first_exception is None:
                    first_exception = e
                logger.exception(f"Error closing async generator: {e}")
        if first_exception is not None:
            raise first_exception
