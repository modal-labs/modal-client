import asyncio
import functools
import inspect
import synchronicity
import time

from .config import logger

synchronizer = synchronicity.Synchronizer()


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


def create_task(coro):
    return asyncio.create_task(add_traceback(coro))


def infinite_loop(async_f, timeout=90, sleep=10):
    async def loop_coro():
        logger.debug(f"Starting infinite loop {async_f}")
        while True:
            try:
                await asyncio.wait_for(async_f(), timeout=timeout)
            except Exception:
                logger.exception(f"Loop attempt failed for {async_f}")
            await asyncio.sleep(sleep)

    return create_task(loop_coro())


class GeneratorStream:
    """Utility for taking a sync/async generator and iterating over it.

    TODO: break this out into an open source package, maybe synchronizer for now
    """

    def __init__(self, generator):
        self._q = asyncio.Queue()
        self.done = False
        self._generator = generator

    async def __aenter__(self):
        if inspect.isgenerator(self._generator):
            loop = asyncio.get_event_loop()
            self._pump_task = loop.run_in_executor(None, self._pump_syncgen, self._generator)
        elif inspect.isasyncgen(self._generator):
            self._pump_task = asyncio.create_task(self._pump_asyncgen(self._generator))
        else:
            raise Exception(f"{self._generator} has to be a sync/async generator")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._pump_task.cancel()

    def _pump_syncgen(self, generator):
        try:
            for val in generator:
                self._q.put_nowait(("val", val))
        except Exception as exc:
            logger.exception(f"Exception while running {generator}")
            self._q.put_nowait(("exc", exc))
        self._q.put_nowait(("fin", None))

    async def _pump_asyncgen(self, generator):
        try:
            async for val in generator:
                await self._q.put(("val", val))
        except Exception as exc:
            logger.exception(f"Exception while running {generator}")
            await self._q.put(("exc", exc))
        await self._q.put(("fin", None))

    async def _get(self, timeout=None):
        tag, value = await asyncio.wait_for(self._q.get(), timeout=timeout)

        if tag == "val":
            return value
        elif tag == "exc":
            raise value
        elif tag == "fin":
            self.done = True
        else:
            raise Exception(f"weird tag {tag}")

    async def all(self):
        while True:
            value = await self._get()
            if self.done:
                return
            yield value

    async def chunk(self, timeout):
        """Returns an async generator that generates elements up until timeout."""
        t0 = time.time()
        while True:
            attempt_timeout = timeout - (time.time() - t0)
            try:
                value = await self._get(timeout=attempt_timeout)
            except asyncio.TimeoutError:
                break
            if self.done:
                return
            yield value


async def chunk_generator(generator, timeout):
    async with GeneratorStream(generator) as stream:
        while not stream.done:
            yield stream.chunk(timeout)


# TODO: maybe these methods could move into synchronizer later?


def asyncify_generator(generator_fn):
    """Takes a blocking generator and returns an async generator."""

    async def new_generator(*args, **kwargs):
        async with GeneratorStream(generator_fn(*args, **kwargs)) as stream:
            async for elm in stream.all():
                yield elm

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

    def create_task(self, coro):
        task = create_task(coro)
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
