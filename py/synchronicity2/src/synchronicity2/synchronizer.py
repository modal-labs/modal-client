# Copyright Modal Labs 2026
import asyncio
import atexit
import concurrent.futures
import importlib
import os
import sys
import threading
import typing
from typing import Optional

from .module import _direct_wrapper_location

if typing.TYPE_CHECKING:
    from synchronicity import Synchronizer as Synchronicity1Synchronizer  # pyright: ignore[reportAttributeAccessIssue]

T = typing.TypeVar("T")
R = typing.TypeVar("R")


class WrapperClassProtocol(typing.Protocol):
    _impl_instance: typing.Any

    @classmethod
    def _from_impl(cls, impl_instance: typing.Any) -> typing.Any: ...


WRAPPER_CLASS_T = typing.TypeVar("WRAPPER_CLASS_T", bound=WrapperClassProtocol)


# Calling asend/athrow/aclose/__anext__ on an async generator binds it to the asyncgen hooks
# (sys.set_asyncgen_hooks) of the *calling* thread, even though the returned awaitable
# isn't stepped until later. Wrapping the calls in coroutine functions defers them until
# the coroutine runs on the synchronizer loop, so that loop owns the generator and GC
# finalization (aclose) happens there, rather than on the caller's loop/thread.
async def _asend(gen: typing.AsyncGenerator[typing.Any, typing.Any], value: typing.Any) -> typing.Any:
    return await gen.asend(value)


async def _athrow(gen: typing.AsyncGenerator[typing.Any, typing.Any], exc: BaseException) -> typing.Any:
    return await gen.athrow(exc)


async def _aclose(gen: typing.AsyncGenerator[typing.Any, typing.Any]) -> None:
    await gen.aclose()


async def _anext(async_iter: typing.AsyncIterator[T]) -> T:
    return await async_iter.__anext__()


def _wrapped_from_impl(
    wrapper_cls: type[WRAPPER_CLASS_T],
    impl_instance: typing.Any,
    cache: typing.Any,
    synchronizer: "Synchronizer",
) -> WRAPPER_CLASS_T:
    """
    Create or retrieve a wrapper instance from an implementation instance.

    This helper is used by all generated _from_impl classmethods to handle
    caching and wrapper creation uniformly.

    Args:
        wrapper_cls: The wrapper class to create an instance of (only used for static typing)
        impl_instance: The implementation instance to wrap
        cache: A WeakValueDictionary cache for storing wrapper instances
        synchronizer: The synchronizer that owns runtime wrapper registrations

    Returns:
        A wrapper instance (either from cache or newly created)
    """
    resolved_wrapper_cls = typing.cast(type[WRAPPER_CLASS_T], synchronizer._resolve_wrapper_class(impl_instance))
    resolved_cache = getattr(resolved_wrapper_cls, "_instance_cache", cache)

    # Use id() as cache key since impl instances are Python objects
    cache_key = id(impl_instance)

    # Check cache first
    if cache_key in resolved_cache:
        return resolved_cache[cache_key]

    # Create new wrapper using __new__ to bypass __init__
    wrapper = resolved_wrapper_cls.__new__(resolved_wrapper_cls)
    wrapper._impl_instance = impl_instance
    # Cache it
    resolved_cache[cache_key] = wrapper

    return wrapper


class Synchronizer:
    def __init__(
        self,
        *,
        synchronicity1_synchronizer: Optional["Synchronicity1Synchronizer"] = None,
    ):
        self._synchronicity1_synchronizer = synchronicity1_synchronizer
        self._future_poll_interval = 0.1
        self._cancellation_future_transfer_seconds = 1
        self._blocking_in_async_callback = None
        self._loop = None
        self._loop_creation_lock = threading.Lock()
        self._thread = None
        self._owner_pid = None
        self._stopping = None
        self._asyncgen_finalizer_timeout_seconds = 10.0  # high default to allow async finalization in most cases

        # Special attribute we use to go from wrapped <-> original
        self._wrapped_attr = "_sync_wrapped_%d" % id(self)
        self._original_attr = "_sync_original_%d" % id(self)

        # Special attribute to mark something as non-wrappable
        self._nowrap_attr = "_sync_nonwrap_%d" % id(self)
        self._input_translation_attr = "_sync_input_translation_%d" % id(self)
        self._output_translation_attr = "_sync_output_translation_%d" % id(self)
        self._wrapper_classes: dict[type, type[WrapperClassProtocol]] = {}

        if synchronicity1_synchronizer is None:
            atexit.register(self._close_loop)

    @property
    def _synchronicity1(self) -> "Synchronicity1Synchronizer":
        if self._synchronicity1_synchronizer is None:
            raise RuntimeError("This generated wrapper requires a Synchronicity 1 synchronizer")
        return self._synchronicity1_synchronizer

    def register_wrapper_class(self, impl_type: type, wrapper_cls: type[WrapperClassProtocol]) -> None:
        existing = self._wrapper_classes.get(impl_type)
        if existing is not None and existing is not wrapper_cls:
            raise RuntimeError(
                f"Implementation type {impl_type!r} already registered to wrapper {existing!r}, "
                f"cannot replace it with {wrapper_cls!r}"
            )
        self._wrapper_classes[impl_type] = wrapper_cls
        if self._synchronicity1_synchronizer is not None:
            self._synchronicity1_synchronizer.register_external_wrapper_class(
                impl_type,
                wrapper_cls,
                translate_in=lambda wrapper: wrapper._impl_instance,
                translate_out=wrapper_cls._from_impl,
            )

    def _resolve_wrapper_class(
        self,
        impl_instance: typing.Any,
    ) -> type[WrapperClassProtocol]:
        impl_type = type(impl_instance)
        wrapper_cls = self._wrapper_classes.get(impl_type)
        if wrapper_cls is not None:
            return wrapper_cls

        location = _direct_wrapper_location(impl_type)
        if location is None:
            raise RuntimeError(f"Implementation type {impl_type!r} has no registered wrapper location")

        target_module, wrapper_name = location
        importlib.import_module(target_module)
        wrapper_cls = self._wrapper_classes.get(impl_type)
        if wrapper_cls is None:
            raise RuntimeError(
                f"Wrapper module {target_module!r} did not register wrapper {wrapper_name!r} "
                f"for implementation type {impl_type!r}"
            )
        return wrapper_cls

    def _start_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_creation_lock:
            if self._loop and self._loop.is_running():
                # in case of a race between two _start_loop, the loop might already
                # be created here by another thread
                return self._loop

            is_ready = threading.Event()

            def thread_inner() -> None:
                async def loop_inner() -> None:
                    self._loop = asyncio.get_running_loop()
                    self._stopping = asyncio.Event()
                    is_ready.set()
                    await self._stopping.wait()  # wait until told to stop

                try:
                    asyncio.run(loop_inner())
                except RuntimeError as exc:
                    # Python 3.12 raises a RuntimeError when new threads are created at shutdown.
                    # Swallowing it here is innocuous, but ideally we will revisit this after
                    # refactoring the shutdown handlers that modal uses to avoid triggering it.
                    if "can't create new thread at interpreter shutdown" not in str(exc):
                        raise exc

            self._owner_pid = os.getpid()
            thread = threading.Thread(target=thread_inner, daemon=True)
            thread.start()
            is_ready.wait()  # TODO: this might block for a very short time
            self._thread = thread
            assert self._loop
            return self._loop

    def _close_loop(self):
        if self._synchronicity1_synchronizer is not None:
            return
        # Use getattr to protect against weird gc races when we get here via __del__
        if getattr(self, "_thread", None) is not None:
            if self._loop and not self._loop.is_closed() and self._stopping:
                # This also serves the purpose of waking up an idle loop
                self._loop.call_soon_threadsafe(self._stopping.set)

            if self._thread:
                self._thread.join()

            self._thread = None
            self._loop = None
            self._owner_pid = None

    @typing.overload
    def _get_loop(self, start: typing.Literal[True]) -> asyncio.AbstractEventLoop: ...

    @typing.overload
    def _get_loop(self, start: bool = False) -> Optional[asyncio.AbstractEventLoop]: ...

    def _get_loop(self, start: bool = False) -> Optional[asyncio.AbstractEventLoop]:
        if self._synchronicity1_synchronizer is not None:
            return self._synchronicity1_synchronizer._get_loop(start=start)
        if self._thread and not self._thread.is_alive():
            if self._owner_pid == os.getpid():
                # warn - thread died without us forking
                raise RuntimeError("Synchronizer thread unexpectedly died")

            self._thread = None
            self._loop = None

        if self._loop is None and start:
            return self._start_loop()

        return self._loop

    def _get_running_loop(self):
        # TODO: delete this method
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return

    def _is_inside_loop(self):
        if self._synchronicity1_synchronizer is not None:
            return self._synchronicity1_synchronizer._is_inside_loop()
        loop = self._get_loop()
        if loop is None:
            return False
        if threading.current_thread() != self._thread:
            # gevent does something bad that causes asyncio.get_running_loop() to return self._loop
            return False
        current_loop = self._get_running_loop()
        return loop == current_loop

    def _maybe_warn_blocking_in_async(self):
        if self._blocking_in_async_callback is None or self._is_inside_loop():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._blocking_in_async_callback(None)

    def _run_function_sync(self, coro, *, _warn_if_running: bool = True):
        if self._is_inside_loop():
            raise Exception("Deadlock detected: calling a sync function from the synchronizer loop")
        if _warn_if_running:
            self._maybe_warn_blocking_in_async()

        loop = self._get_loop(start=True)

        inner_task_fut = concurrent.futures.Future()

        async def wrapper_coro():
            # this wrapper is needed since run_coroutine_threadsafe *only* accepts coroutines
            inner_task = loop.create_task(coro)
            inner_task_fut.set_result(inner_task)  # sends the task itself to the origin thread
            return await inner_task

        fut = asyncio.run_coroutine_threadsafe(wrapper_coro(), loop)
        try:
            while 1:
                try:
                    # repeated poll to give Windows a chance to abort on Ctrl-C
                    value = fut.result(timeout=self._future_poll_interval)
                    break
                except concurrent.futures.TimeoutError:
                    # concurrent.futures.TimeoutError aliases builtins TimeoutError, so a coroutine-raised
                    # TimeoutError is indistinguishable by exception type alone. Only treat it as a polling
                    # timeout while the cross-thread future is still pending.
                    if fut.done():
                        raise
        except BaseException as exc:
            # If `fut.done()` is True, then (1) the exception was raised from user code, and (2) we
            # don't need to do any cleanup as the future has already completed.
            if fut.done():
                raise exc

            # Some other exception (KeyboardInterrupt, SystemExit, an exception injected by a
            # signal handler, ...) interrupted the wait. We cancel the *underlying* coro_task
            # (unlike what fut.cancel() would do) and then wait for the *wrapper* coroutine to get
            # a result back, which happens after the cancellation resolves
            if inner_task_fut.done():
                inner_task: asyncio.Task = inner_task_fut.result()
                loop.call_soon_threadsafe(inner_task.cancel)
            else:
                # it's possible that the interrupt has raced with scheduling the task on the other
                # thread, so give it a grace period to complete
                try:
                    inner_task = inner_task_fut.result(timeout=self._cancellation_future_transfer_seconds)
                except concurrent.futures.TimeoutError:
                    pass
                else:
                    loop.call_soon_threadsafe(inner_task.cancel)

            try:
                value = fut.result()
            except concurrent.futures.CancelledError as expected_cancellation:
                # we *expect* this cancellation, but defer to the passed coro to potentially
                # intercept and treat the cancellation some other way
                expected_cancellation.__suppress_context__ = True
                raise exc  # if cancel - re-raise the original exception again

        return value  # type: ignore

    def _run_function_sync_future(self, coro) -> concurrent.futures.Future:
        loop = self._get_loop(start=True)
        return asyncio.run_coroutine_threadsafe(coro, loop)

    async def _run_function_async(self, coro):
        loop = self._get_loop(start=True)
        if self._is_inside_loop():
            value = await coro
        else:
            inner_task_fut = concurrent.futures.Future()

            async def wrapper_coro():
                inner_task = loop.create_task(coro)
                inner_task_fut.set_result(inner_task)  # sends the task itself to the origin thread
                return await inner_task

            c_fut = asyncio.run_coroutine_threadsafe(wrapper_coro(), loop)
            a_fut = asyncio.wrap_future(c_fut)

            try:
                # The polling loop in _run_function_sync is only needed so a blocking thread can
                # observe interrupts. Async callers can await the cross-thread future directly;
                # shielding keeps caller cancellation from cancelling a_fut before we cancel the
                # underlying task explicitly below.
                value = await asyncio.shield(a_fut)

            except asyncio.CancelledError:
                if a_fut.cancelled():
                    raise  # cancellation came from within c_fut
                if inner_task_fut.done():
                    inner_task: asyncio.Task = inner_task_fut.result()
                    loop.call_soon_threadsafe(inner_task.cancel)  # cancel task on synchronizer event loop
                    # wait for cancellation logic in the underlying coro to complete
                    # this should typically raise CancelledError, but in case of either:
                    # * cancellation prevention in the coro (catching the CancelledError)
                    # * coro_task resolves before the call_soon_threadsafe above is scheduled
                    # the cancellation in a_fut would be cancelled

                    await a_fut  # wait for cancellation logic to complete - this *normally* raises CancelledError
                else:
                    # it's possible that the cancellation has raced with scheduling the task on the other thread,
                    # so give it a grace period to complete
                    try:
                        inner_task = await asyncio.wait_for(
                            asyncio.shield(asyncio.wrap_future(inner_task_fut)),
                            timeout=self._cancellation_future_transfer_seconds,
                        )
                    except asyncio.TimeoutError:
                        pass
                    else:
                        loop.call_soon_threadsafe(inner_task.cancel)
                        await a_fut
                raise  # re-raise the CancelledError regardless - preventing unintended cancellation aborts

        return value  # type: ignore

    def _run_generator_sync(self, gen):
        self._maybe_warn_blocking_in_async()
        value: typing.Any = None
        is_exc = False
        try:
            while True:
                try:
                    if is_exc:
                        # When is_exc is True, value is always a BaseException
                        assert isinstance(value, BaseException)
                        value = self._run_function_sync(_athrow(gen, value), _warn_if_running=False)
                    else:
                        value = self._run_function_sync(_asend(gen, value), _warn_if_running=False)
                except StopAsyncIteration:
                    break

                try:
                    value = yield value
                    is_exc = False
                except GeneratorExit:
                    # GeneratorExit signals cleanup - don't forward via athrow, just propagate
                    raise
                except BaseException as exc:
                    value = exc
                    is_exc = True
        finally:
            self._finalize_generator_sync(gen)

    def _finalize_generator_sync(self, gen: typing.AsyncGenerator[typing.Any, typing.Any]) -> None:
        # During interpreter shutdown, blocking here can deadlock.
        if sys.is_finalizing():
            return
        try:
            # Best-effort close on the synchronizer loop. We use a future so we don't block
            # indefinitely in case the event loop closing races with this code and the aclose
            # never returns.
            finalization_fut = self._run_function_sync_future(_aclose(gen))
            if self._is_inside_loop():
                # The wrapper generator is being closed from the synchronizer thread itself
                # (e.g. by the gc) - waiting for the scheduled aclose here would deadlock.
                return
            finalization_fut.result(timeout=self._asyncgen_finalizer_timeout_seconds)
        except Exception:
            pass

    async def _run_generator_async(self, gen: typing.AsyncGenerator[typing.Any, typing.Any]):
        value: typing.Any = None
        is_exc = False
        try:
            while True:
                try:
                    if is_exc:
                        # When is_exc is True, value is always a BaseException
                        assert isinstance(value, BaseException)
                        value = await self._run_function_async(_athrow(gen, value))
                    else:
                        value = await self._run_function_async(_asend(gen, value))
                except StopAsyncIteration:
                    break

                try:
                    value = yield value
                    is_exc = False
                except GeneratorExit:
                    # GeneratorExit signals cleanup - don't forward via athrow, just propagate
                    raise
                except BaseException as exc:
                    value = exc
                    is_exc = True
        finally:
            # During interpreter shutdown, blocking here can deadlock.
            if not sys.is_finalizing():
                try:
                    # Best-effort close on the synchronizer loop, bounded so a loop that is shutting
                    # down concurrently can't block the caller forever. The shield lets the aclose
                    # keep running in the background if it outlives the timeout.
                    close_task = asyncio.create_task(self._run_function_async(_aclose(gen)))
                    await asyncio.wait_for(asyncio.shield(close_task), timeout=self._asyncgen_finalizer_timeout_seconds)
                except Exception:
                    pass

    def _run_iterator_sync(self, async_iter: typing.AsyncIterator[T]) -> typing.Generator[T, None, None]:
        """Run an async iterator in sync mode.

        Unlike generators, iterators don't have asend()/aclose(), just __aiter__() and __anext__().
        This method simply iterates using anext() without send() support.
        """
        self._maybe_warn_blocking_in_async()
        while True:
            try:
                value = self._run_function_sync(_anext(async_iter), _warn_if_running=False)
            except StopAsyncIteration:
                break
            yield value

    async def _run_iterator_async(self, async_iter):
        """Run an async iterator in async mode.

        Unlike generators, iterators don't have asend()/aclose(), just __aiter__() and __anext__().
        This method simply iterates using anext() without send() support.
        """
        try:
            while True:
                try:
                    value = await self._run_function_async(_anext(async_iter))
                except StopAsyncIteration:
                    break
                yield value
        finally:
            # Iterators don't have aclose(), so no cleanup needed
            pass
