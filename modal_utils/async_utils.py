# Copyright Modal Labs 2022
import asyncio
import concurrent.futures
import functools
import inspect
import time
import typing
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, List, Optional, Set, TypeVar

import synchronicity
from typing_extensions import ParamSpec

from .logger import logger

synchronizer = synchronicity.Synchronizer()
# atexit.register(synchronizer.close)


def synchronize_api(obj, target_module=None):
    if inspect.isclass(obj):
        blocking_name = obj.__name__.lstrip("_")
    elif inspect.isfunction(object):
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
    """Simple thing to make sure we don't have stray tasks.

    Usage:
    async with TaskContext() as task_context:
        task = task_context.create(coro())
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
                # pre Python3.8, CancelledErrors were a subclass of exception
                except asyncio.CancelledError:
                    pass

            for task in self._tasks:
                if task.done() and not task.cancelled():
                    # Raise any exceptions if they happened.
                    # Only tasks without a done_callback will still be present in self._tasks
                    task.result()

                if task.done() or task in self._loops:
                    continue

                logger.warning(f"Canceling remaining unfinished task {task}")
                task.cancel()

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

    def infinite_loop(self, async_f, timeout: Optional[float] = 90, sleep: float = 10) -> asyncio.Task:
        function_name = async_f.__qualname__

        async def loop_coro() -> None:
            logger.debug(f"Starting infinite loop {function_name}")
            while True:
                t0 = time.time()
                try:
                    await asyncio.wait_for(async_f(), timeout=timeout)
                    # pre Python3.8, CancelledErrors were a subclass of exception
                except asyncio.CancelledError:
                    raise
                except Exception:
                    time_elapsed = time.time() - t0
                    logger.exception(f"Loop attempt failed for {function_name} (time_elapsed={time_elapsed})")
                try:
                    await asyncio.wait_for(self._exited.wait(), timeout=sleep)
                except asyncio.TimeoutError:
                    continue
                # Only reached if self._exited got set.
                logger.debug(f"Exiting infinite loop for {function_name}")
                break

        t = self.create_task(loop_coro())
        if hasattr(t, "set_name"):  # Was added in Python 3.8:
            t.set_name(f"{function_name} loop")
        self._loops.add(t)
        t.add_done_callback(self._loops.discard)
        return t

    async def wait(self, *tasks):
        # Waits until all of tasks have finished
        # This is slightly different than asyncio.wait since the `tasks` argument
        # may be a subset of all the tasks.
        # If any of the task context's task raises, throw that exception
        # This is probably O(n^2) sadly but I guess it's fine
        unfinished_tasks = set(tasks)
        while True:
            unfinished_tasks &= self._tasks
            if not unfinished_tasks:
                break
            try:
                done, pending = await asyncio.wait_for(
                    asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED), timeout=30.0
                )
            except asyncio.TimeoutError:
                continue
            for task in done:
                task.result()  # Raise exception if needed
                if task in unfinished_tasks:
                    unfinished_tasks.remove(task)
                if task in self._tasks:
                    self._tasks.remove(task)


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
    def __init__(self, gen, gen_f):
        self.gen = gen
        self.gen_f = gen_f
        self.iterated = False
        self.warned = False

    def __aiter__(self):
        self.iterated = True
        return self.gen

    async def __anext__(self):
        self.iterated = True
        return await self.gen.__anext__()

    def __repr__(self):
        return repr(self.gen)

    def __del__(self):
        if not self.iterated and not self.warned:
            self.warned = True
            name = self.gen_f.__name__
            logger.warning(
                f"Warning: the results of a call to {name} was not consumed, so the call will never be executed."
                f" Consider a for-loop like `for x in {name}(...)` or unpacking the generator using `list(...)`"
            )


synchronize_api(_WarnIfGeneratorIsNotConsumed)


def warn_if_generator_is_not_consumed(gen_f):
    # https://gist.github.com/erikbern/01ae78d15f89edfa7f77e5c0a827a94d
    @functools.wraps(gen_f)
    def f_wrapped(*args, **kwargs):
        gen = gen_f(*args, **kwargs)
        return _WarnIfGeneratorIsNotConsumed(gen, gen_f)

    return f_wrapped


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


def asyncify(f: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """Convert a blocking function into one that runs in the current loop's executor."""

    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Awaitable[T]:
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, functools.partial(f, *args, **kwargs))

    return wrapper


async def iterate_blocking(iterator: Iterator[T]) -> AsyncIterator[T]:
    """Iterate over a blocking iterator in an async context."""

    loop = asyncio.get_running_loop()
    DONE = object()
    while True:
        obj = await loop.run_in_executor(None, next, iterator, DONE)
        if obj is DONE:
            break
        yield obj


class ConcurrencyPool:
    def __init__(self, concurrency_limit: int):
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def run_coros(self, coros: typing.Iterable[typing.Coroutine], return_exceptions=False):
        async def blocking_wrapper(coro):
            # Not using async with on the semaphore is intentional here - if return_exceptions=False
            # manual release prevents starting extraneous tasks after exceptions.
            await self.semaphore.acquire()
            try:
                res = await coro
                self.semaphore.release()
                return res
            except BaseException as e:
                if return_exceptions:
                    self.semaphore.release()
                raise e

        # asyncio.gather() is weird - it doesn't cancel outstanding awaitables on exceptions when
        # return_exceptions=False --> wrap the coros in tasks are cancel them explicitly on exception.
        tasks = [asyncio.create_task(blocking_wrapper(coro)) for coro in coros]
        g = asyncio.gather(*tasks, return_exceptions=return_exceptions)
        try:
            return await g
        except BaseException as e:
            for t in tasks:
                t.cancel()
            raise e


@asynccontextmanager
async def asyncnullcontext(*args, **kwargs):
    """Async noop context manager.

    Note that for Python 3.10+ you can use contextlib.nullcontext() instead.

    Usage:
    async with asyncnullcontext():
        pass
    """
    yield
