# Copyright Modal Labs 2022
import asyncio
import concurrent.futures
import functools
import inspect
import itertools
import time
import typing
from contextlib import asynccontextmanager
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import synchronicity
from typing_extensions import ParamSpec

from ..exception import InvalidError
from .logger import logger

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

    _loops: Set[asyncio.Task]

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


async def queue_batch_iterator(q: asyncio.Queue, max_batch_size=100, debounce_time=0.015):
    """
    Read from a queue but return lists of items when queue is large

    Treats a None value as end of queue items
    """
    item_list: List[Any] = []

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

    def __init__(self, async_iterable: typing.AsyncIterable[Any], nested_async_message):
        self._async_iterable = async_iterable
        self.nested_async_message = nested_async_message

    def __aiter__(self):
        return self._async_iterable

    def __iter__(self):
        try:
            for output in run_generator_sync(self._async_iterable):  # type: ignore
                yield output
        except NestedAsyncCalls:
            raise InvalidError(self.nested_async_message)


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


T = TypeVar("T")
P = ParamSpec("P")
V = TypeVar("V")


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


class NestedAsyncCalls(Exception):
    pass


def run_generator_sync(
    gen: typing.AsyncGenerator[YIELD_TYPE, SEND_TYPE],
) -> typing.Generator[YIELD_TYPE, SEND_TYPE, None]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no event loop - this is what we expect!
    else:
        raise NestedAsyncCalls()
    loop = asyncio.new_event_loop()  # set up new event loop for the map so we can use async logic

    # more or less copied from synchronicity's implementation:
    next_send: typing.Union[SEND_TYPE, None] = None
    next_yield: YIELD_TYPE
    exc: Optional[BaseException] = None
    while True:
        try:
            if exc:
                next_yield = loop.run_until_complete(gen.athrow(exc))
            else:
                next_yield = loop.run_until_complete(gen.asend(next_send))  # type: ignore[arg-type]
        except StopAsyncIteration:
            break
        try:
            next_send = yield next_yield
            exc = None
        except BaseException as err:
            exc = err
    loop.close()


class AsyncItemType(Enum):
    VALUE = "value"
    EXCEPTION = "exception"
    STOP = "stop"


@asynccontextmanager
async def aclosing(
    agen: AsyncGenerator[T, None],
) -> AsyncGenerator[AsyncGenerator[T, None], None]:
    try:
        yield agen
    finally:
        await agen.aclose()


async def sync_or_async_iter(iterable: Union[Iterable[T], AsyncIterable[T]]) -> AsyncGenerator[T, None]:
    if hasattr(iterable, "__aiter__"):
        async for item in typing.cast(AsyncIterable[T], iterable):
            yield item
    else:
        # This intentionally could block the event loop for the duration of calling __iter__ and __next__,
        # so in non-trivial cases (like passing lists and ranges) this could be quite a foot gun for users #
        # w/ async code (but they can work around it by always using async iterators)
        for item in typing.cast(Iterable[T], iterable):
            yield item


async def async_zip(*inputs: Union[AsyncIterable[T], Iterable[T]]) -> AsyncGenerator[Tuple[T, ...], None]:
    generators = [sync_or_async_iter(it) for it in inputs]
    while True:
        try:
            items = await asyncio.gather(*(it.__anext__() for it in generators))
            yield tuple(items)
        except StopAsyncIteration:
            break


async def async_merge(*inputs: Union[AsyncIterable[T], Iterable[T]]) -> AsyncGenerator[T, None]:
    queue: asyncio.Queue[Tuple[int, Tuple[str, Union[T, Exception, None]]]] = asyncio.Queue()

    async def producer(producer_id: int, iterable: Union[AsyncIterable[T], Iterable[T]]):
        try:
            async for item in sync_or_async_iter(iterable):
                await queue.put((producer_id, (AsyncItemType.VALUE, item)))
        except Exception as e:
            await queue.put((producer_id, (AsyncItemType.EXCEPTION, e)))
        finally:
            await queue.put((producer_id, (AsyncItemType.STOP, None)))

    tasks = [asyncio.create_task(producer(i, it)) for i, it in enumerate(inputs)]
    active_producers = set(range(len(inputs)))

    try:
        while active_producers:
            producer_id, (event_type, item) = await queue.get()
            if event_type == AsyncItemType.EXCEPTION:
                raise typing.cast(Exception, item)
            elif event_type == AsyncItemType.STOP:
                active_producers.remove(producer_id)
            else:
                yield typing.cast(T, item)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def callable_to_agen(awaitable: Callable[[], Awaitable[T]]) -> AsyncGenerator[T, None]:
    yield await awaitable()


async def async_map(
    input_iterable: Union[AsyncIterable[T], Iterable[T]],
    async_mapper_func: Callable[[T], Awaitable[V]],
    concurrency: int,
) -> AsyncGenerator[V, None]:
    input_queue: asyncio.Queue[Tuple[str, Union[T, None]]] = asyncio.Queue()
    output_queue: asyncio.Queue[Tuple[str, Union[V, Exception]]] = asyncio.Queue()
    output_event = asyncio.Event()

    async def producer():
        async for item in sync_or_async_iter(input_iterable):
            await input_queue.put((AsyncItemType.VALUE, item))
        await input_queue.put((AsyncItemType.STOP, None))

    async def worker():
        while True:
            try:
                event_type, item = await input_queue.get()
                if event_type == AsyncItemType.STOP:
                    break
                if event_type == AsyncItemType.VALUE:
                    res = async_mapper_func(item)
                    if inspect.isawaitable(res):
                        result = await res
                    else:
                        result = typing.cast(V, res)
                    await output_queue.put((AsyncItemType.VALUE, result))
            except Exception as e:
                await output_queue.put((AsyncItemType.EXCEPTION, e))
            finally:
                input_queue.task_done()

    producer_task = asyncio.create_task(producer())
    worker_tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]

    async def complete_map():
        await producer_task
        await input_queue.join()

    complete_map_task = asyncio.create_task(complete_map())
    all_tasks = [*worker_tasks, complete_map_task]

    try:
        while True:
            done, _ = await asyncio.wait(
                [*all_tasks, asyncio.create_task(output_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )

            finished_workers = done & set(worker_tasks)
            for finished_worker in finished_workers:
                # this is done in order to catch potential raised errors/cancellations
                # from within worker tasks as soon as they happen.
                await finished_worker

            if complete_map_task.done():
                while not output_queue.empty():
                    event_type, item = await output_queue.get()
                    if event_type == AsyncItemType.VALUE:
                        yield typing.cast(V, item)
                    elif event_type == AsyncItemType.EXCEPTION:
                        raise typing.cast(Exception, item)
                    else:
                        raise Exception(f"Unknown event type: {event_type}")
                await complete_map_task
                break

            if output_event.is_set():
                while not output_queue.empty():
                    event_type, item = await output_queue.get()
                    if event_type == AsyncItemType.VALUE:
                        yield typing.cast(V, item)
                    elif event_type == AsyncItemType.EXCEPTION:
                        raise typing.cast(Exception, item)
                    else:
                        raise Exception(f"Unknown event type: {event_type}")
                output_event.clear()

    finally:
        for task in all_tasks:
            task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)


async def async_map_ordered(
    input_iterable: Union[AsyncIterable[T], Iterable[T]],
    async_mapper_func: Callable[[T], Awaitable[V]],
    concurrency: int,
) -> AsyncGenerator[V, None]:
    async def mapper_func_wrapper(tup: Tuple[int, T]) -> Tuple[int, V]:
        res = async_mapper_func(tup[1])
        if inspect.isawaitable(res):
            return tup[0], await res
        return tup[0], typing.cast(V, res)

    async def counter() -> AsyncGenerator[int, None]:
        for i in itertools.count():
            yield i

    next_idx = 0
    buffer = {}

    async with aclosing(counter()) as counter_gen:
        async with aclosing(async_zip(counter_gen, input_iterable)) as zipped_input:
            async with aclosing(async_map(zipped_input, mapper_func_wrapper, concurrency)) as stream:
                async for output_idx, output_item in stream:
                    buffer[output_idx] = output_item
                    if output_idx == next_idx:
                        while next_idx in buffer:
                            yield buffer[next_idx]
                            del buffer[next_idx]
                            next_idx += 1
