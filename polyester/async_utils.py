import asyncio
import contextlib
import functools
import inspect
import time

import synchronicity

from .config import logger

synchronizer = synchronicity.Synchronizer()
# atexit.register(synchronizer.close)


def retry(direct_fn=None, n_attempts=3, base_delay=0, delay_factor=2, timeout=90):
    def decorator(fn):
        @functools.wraps(fn)
        async def f_wrapped(*args, **kwargs):
            delay = base_delay
            for i in range(n_attempts):
                try:
                    t0 = time.time()
                    return await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout)
                except asyncio.CancelledError:
                    logger.warning(f"Function {fn} was cancelled")
                    raise
                except Exception as e:
                    if i >= n_attempts - 1:
                        raise
                    logger.warning(
                        f"Failed invoking function {fn}: {repr(e)}"
                        f" (took {time.time() - t0}s, sleeping {delay}s"
                        f" and trying {n_attempts - i - 1} more times)"
                    )
                await asyncio.sleep(delay)
                delay *= delay_factor

        return f_wrapped

    if direct_fn is not None:
        # It's invoked like @retry\ndef f(...)
        return decorator(direct_fn)
    else:
        # It's invoked like @retry(n_attempts=...)\ndef f(...)
        return decorator


def add_traceback(obj, func_name=None):
    if func_name is None:
        func_name = repr(obj)
    if inspect.iscoroutine(obj):

        async def _wrap_coro():
            try:
                return await obj
            except Exception:
                logger.exception(f"Exception while running {func_name}")
                raise

        return _wrap_coro()
    elif inspect.isasyncgen(obj):

        async def _wrap_gen():
            try:
                async for elm in obj:
                    yield elm
            except Exception:
                logger.exception(f"Exception while running {func_name}")
                raise

        return _wrap_gen()
    else:
        raise Exception(f"{obj} is not a coro or async gen!")


def create_task(coro, /, name=None):
    return asyncio.create_task(add_traceback(coro), name=name)


def infinite_loop(async_f, timeout=90, sleep=10):
    async def loop_coro():
        logger.debug(f"Starting infinite loop {async_f}")
        while True:
            try:
                await asyncio.wait_for(async_f(), timeout=timeout)
            except Exception:
                logger.exception(f"Loop attempt failed for {async_f}")
            except asyncio.CancelledError:
                return
            await asyncio.sleep(sleep)

    # functools.partial objects have no __name__ attribute :(
    return create_task(loop_coro(), name=f"infinite_loop_{async_f}")


async def chunk_generator(generator, timeout):
    """Takes a generator and returns a generator of generator where each sub-generator only runs for a certain time.

    TODO: merge this into aiostream.
    """
    done = False
    task = None
    try:
        while not done:

            async def chunk():
                nonlocal done, task
                t0 = time.time()
                while True:
                    try:
                        attempt_timeout = t0 + timeout - time.time()
                        if task is None:
                            coro = generator.__anext__()
                            task = asyncio.create_task(coro)
                        value = await asyncio.wait_for(asyncio.shield(task), attempt_timeout)
                        yield value
                        task = None
                    except asyncio.TimeoutError:
                        return
                    except StopAsyncIteration:
                        done = True
                        return

            yield chunk()
    finally:
        if task is not None:
            task.cancel()


# TODO: maybe these methods could move into synchronizer later?


def asyncify_generator(generator_fn):
    """Takes a blocking generator and returns an async generator.

    TODO: merge into aiostream: https://github.com/vxgmichel/aiostream/issues/78
    """

    @functools.wraps(generator_fn)
    async def new_generator(*args, **kwargs):
        generator = generator_fn(*args, **kwargs)
        loop = asyncio.get_event_loop()
        done = False

        def safe_next(it):
            nonlocal done
            try:
                return next(it)
            except StopIteration as exc:
                done = True

        while True:
            ret = await loop.run_in_executor(None, safe_next, generator)
            if done:
                break
            yield ret

    return new_generator


def asyncify_function(function):
    async def asynced_function(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: function(*args, **kwargs))

    return asynced_function


class TaskContext:
    """Simple thing to make sure we don't have stray tasks.

    Usage:
    async with TaskContext() as task_context:
        task = task_context.create(coro())
    """

    def __init__(self, grace=None):
        self._grace = grace

    async def __aenter__(self):
        self._tasks = []
        return self

    async def __aexit__(self, exc_type, value, tb):
        await asyncio.sleep(0)  # Causes any just-created tasks to get started
        unfinished_tasks = [t for t in self._tasks if not t.done()]
        try:
            if self._grace is not None:
                await asyncio.wait_for(asyncio.gather(*unfinished_tasks), timeout=self._grace)
        except BaseException:
            logger.exception(f"Exception while waiting for {len(unfinished_tasks)} unfinished tasks")
        finally:
            for task in self._tasks:
                task.cancel()

    def create_task(self, coro_or_task):
        if isinstance(coro_or_task, asyncio.Task):
            task = coro_or_task
        elif inspect.iscoroutine(coro_or_task):
            task = create_task(coro_or_task)
        else:
            raise Exception(f"{coro_or_task} is not a coroutine or Task")
        self._tasks.append(task)
        return task

    async def wait(self, *tasks):
        # Waits until all of tasks have finished
        # If any of the task context's task raises, throw that exception
        # This is probably O(n^2) sadly but I guess it's fine
        unfinished_tasks = set(tasks)
        all_tasks = set(self._tasks)
        while unfinished_tasks:
            done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()  # Raise exception if needed
                if task in unfinished_tasks:
                    unfinished_tasks.remove(task)


AsyncGeneratorContextManager = synchronizer(contextlib._AsyncGeneratorContextManager)


def asynccontextmanager(func):
    """This works just like contextlib.asynccontextmanager, but also in synchronous contexts.

    TODO: merge into synchronicity
    """

    @functools.wraps(func)
    def func_in_loop(*args, **kwargs):
        # This is a bit ugly, but we need to
        # 1. Make sure this runs in the right loop
        # 2. Make sure we still have an async (not sync) generator
        return synchronizer._run_generator_async(func(*args, **kwargs))

    @functools.wraps(func)
    def helper(*args, **kwargs):
        return AsyncGeneratorContextManager(func_in_loop, args, kwargs)

    return helper
