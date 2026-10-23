# Copyright Modal Labs 2026
import asyncio
import logging
import pytest
import signal
import subprocess
import sys
import threading
import time
import typing
import weakref
from pathlib import Path

from synchronicity2.synchronizer import Synchronizer, _wrapped_from_impl


def make_synchronicity1_synchronizer():
    try:
        from synchronicity import Synchronizer as Synchronicity1Synchronizer
    except ImportError:
        pytest.skip("Synchronicity 1 is an optional dependency")
    return Synchronicity1Synchronizer()


def test_resolve_wrapper_class_requires_wrapper_location():
    class Impl:
        pass

    sync = Synchronizer()

    with pytest.raises(RuntimeError, match="has no registered wrapper location"):
        sync._resolve_wrapper_class(Impl())


def test_run_function_sync_propagates_coroutine_timeout_error(monkeypatch):
    class DoneFuture:
        def __init__(self):
            self.calls = 0

        def result(self, timeout=None):
            self.calls += 1
            if self.calls > 5:
                raise AssertionError("swallowed coroutine TimeoutError in sync polling loop")
            raise TimeoutError("inner timeout")

        def done(self):
            return True

    sync = Synchronizer()
    done_future = DoneFuture()

    monkeypatch.setattr(sync, "_is_inside_loop", lambda: False)
    monkeypatch.setattr(sync, "_get_loop", lambda start=False: object())
    monkeypatch.setattr(
        asyncio,
        "run_coroutine_threadsafe",
        lambda coro, loop: (coro.close(), done_future)[1],
    )

    with pytest.raises(TimeoutError, match="inner timeout"):
        sync._run_function_sync(object())


@pytest.mark.asyncio
async def test_run_function_async_propagates_coroutine_timeout_error(monkeypatch):
    sync = Synchronizer()

    done_future = object()
    wrapped_future = asyncio.get_running_loop().create_future()
    wrapped_future.set_exception(TimeoutError("inner timeout"))

    monkeypatch.setattr(sync, "_is_inside_loop", lambda: False)
    monkeypatch.setattr(sync, "_get_loop", lambda start=False: object())
    monkeypatch.setattr(
        asyncio,
        "run_coroutine_threadsafe",
        lambda coro, loop: (coro.close(), done_future)[1],
    )
    monkeypatch.setattr(asyncio, "wrap_future", lambda fut: wrapped_future)

    with pytest.raises(TimeoutError, match="inner timeout"):
        await asyncio.wait_for(sync._run_function_async(object()), timeout=0.5)


@pytest.mark.asyncio
async def test_run_function_async_waits_beyond_poll_interval():
    sync = Synchronizer()
    sync._future_poll_interval = 0.001

    async def slow():
        await asyncio.sleep(0.01)
        return "ok"

    try:
        assert await sync._run_function_async(slow()) == "ok"
    finally:
        sync._close_loop()


def test_synchronicity1_synchronizer_uses_shared_loop_without_closing_it():
    owner = make_synchronicity1_synchronizer()
    borrowed = Synchronizer(synchronicity1_synchronizer=owner)

    async def get_running_loop():
        return asyncio.get_running_loop()

    try:
        assert borrowed._run_function_sync(get_running_loop()) is owner._get_loop(start=True)
        borrowed._close_loop()
        assert owner._get_loop(start=False) is not None
    finally:
        owner._close_loop()


def test_synchronicity1_synchronizer_registers_external_wrappers():
    class Impl:
        pass

    class Wrapper:
        _impl_instance: object
        _instance_cache = weakref.WeakValueDictionary()

        @classmethod
        def _from_impl(cls, impl_instance):
            return _wrapped_from_impl(cls, impl_instance, cls._instance_cache, sync)

    owner = make_synchronicity1_synchronizer()
    sync = Synchronizer(synchronicity1_synchronizer=owner)
    sync.register_wrapper_class(Impl, Wrapper)

    impl = Impl()
    wrapper = typing.cast(Wrapper, owner._translate_out(impl))

    assert wrapper._impl_instance is impl
    assert owner._translate_out(Impl) is Wrapper
    assert owner._translate_in(Wrapper) is Impl
    assert owner._translate_out(impl) is wrapper
    assert owner._translate_in(wrapper) is impl
    assert _wrapped_from_impl(Wrapper, impl, Wrapper._instance_cache, sync) is wrapper


@pytest.fixture
def synchronizer():
    sync = Synchronizer()
    try:
        yield sync
    finally:
        sync._close_loop()


def _registered_asyncgens(loop, code):
    return [agen for agen in loop._asyncgens if agen.ag_code is code]


async def two_items():
    yield 1
    yield 2


def test_asyncgen_hooks_bound_to_synchronizer_loop_sync(synchronizer):
    iterator = synchronizer._run_generator_sync(two_items())
    assert next(iterator) == 1
    assert len(_registered_asyncgens(synchronizer._get_loop(), two_items.__code__)) == 1
    iterator.close()


@pytest.mark.asyncio
async def test_asyncgen_hooks_bound_to_synchronizer_loop_async(synchronizer):
    iterator = synchronizer._run_generator_async(two_items())
    assert await iterator.__anext__() == 1
    assert _registered_asyncgens(asyncio.get_running_loop(), two_items.__code__) == []
    assert len(_registered_asyncgens(synchronizer._get_loop(), two_items.__code__)) == 1
    await iterator.aclose()


@pytest.mark.asyncio
async def test_iterator_asyncgen_hooks_bound_to_synchronizer_loop_async(synchronizer):
    iterator = synchronizer._run_iterator_async(two_items())
    assert await iterator.__anext__() == 1
    assert _registered_asyncgens(asyncio.get_running_loop(), two_items.__code__) == []
    assert len(_registered_asyncgens(synchronizer._get_loop(), two_items.__code__)) == 1
    await iterator.aclose()


def _make_recording_genfunc(synchronizer: Synchronizer, states: list):
    async def genfunc(finalization_delay: float):
        states.append("init")
        try:
            for i in range(2):
                await asyncio.sleep(0.01)
                states.append(f"yield {i}")
                yield i
                states.append(f"send after {i}")
        except BaseException:
            states.append("start finalization")
            await asyncio.sleep(finalization_delay)
            on_loop = asyncio.get_running_loop() is synchronizer._get_loop()
            states.append(f"done on synchronizer loop: {on_loop}")

    return genfunc


def test_run_generator_sync_runs_aclose_on_synchronizer_loop(synchronizer):
    states: list[str] = []
    genfunc = _make_recording_genfunc(synchronizer, states)

    for _ in synchronizer._run_generator_sync(genfunc(0.01)):
        pass
    assert states == ["init", "yield 0", "send after 0", "yield 1", "send after 1"]

    states.clear()
    gen = synchronizer._run_generator_sync(genfunc(0.01))
    next(gen)
    gen.close()  # the underlying async generator is closed before close() returns
    assert states == ["init", "yield 0", "start finalization", "done on synchronizer loop: True"]

    states.clear()
    synchronizer._asyncgen_finalizer_timeout_seconds = 0.1
    gen = synchronizer._run_generator_sync(genfunc(0.5))
    next(gen)
    gen.close()
    assert states == ["init", "yield 0", "start finalization"]  # timed out waiting for finalization
    time.sleep(1)
    assert states == ["init", "yield 0", "start finalization", "done on synchronizer loop: True"]


@pytest.mark.asyncio
async def test_run_generator_async_runs_aclose_on_synchronizer_loop(synchronizer):
    states: list[str] = []
    genfunc = _make_recording_genfunc(synchronizer, states)

    async for _ in synchronizer._run_generator_async(genfunc(0.01)):
        pass
    assert states == ["init", "yield 0", "send after 0", "yield 1", "send after 1"]

    states.clear()
    gen = synchronizer._run_generator_async(genfunc(0.01))
    assert await gen.__anext__() == 0
    await gen.aclose()
    assert states == ["init", "yield 0", "start finalization", "done on synchronizer loop: True"]

    states.clear()
    synchronizer._asyncgen_finalizer_timeout_seconds = 0.1
    gen = synchronizer._run_generator_async(genfunc(0.5))
    assert await gen.__anext__() == 0
    await gen.aclose()
    assert states == ["init", "yield 0", "start finalization"]  # timed out waiting for finalization
    await asyncio.sleep(1)
    assert states == ["init", "yield 0", "start finalization", "done on synchronizer loop: True"]


def test_run_generator_sync_closed_from_synchronizer_thread(synchronizer):
    # e.g. the gc collecting an abandoned wrapper generator while running a callback on the loop
    finalized = threading.Event()

    async def genfunc():
        try:
            yield 1
        finally:
            finalized.set()

    gen = synchronizer._run_generator_sync(genfunc())
    next(gen)

    async def close_from_loop():
        gen.close()

    synchronizer._run_function_sync(close_from_loop())
    assert finalized.wait(timeout=1)


def _make_cancellation_recorder(states: list):
    async def long_running():
        try:
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            states.append("cancelled")
            raise
        finally:
            states.append("done")

    return long_running


class CustomBaseException(BaseException):
    pass


def _raise_custom(*_):
    raise CustomBaseException()


@pytest.mark.skipif(sys.platform == "win32", reason="requires SIGALRM")
def test_run_function_sync_cancels_on_arbitrary_base_exception(synchronizer):
    states: list[str] = []
    previous = signal.signal(signal.SIGALRM, _raise_custom)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.1)
        with pytest.raises(CustomBaseException):
            synchronizer._run_function_sync(_make_cancellation_recorder(states)())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert states == ["cancelled", "done"]


@pytest.mark.skipif(sys.platform == "win32", reason="requires SIGALRM")
def test_run_function_sync_interrupted_before_task_is_scheduled(synchronizer):
    states: list[str] = []
    loop = synchronizer._get_loop(start=True)
    previous = signal.signal(signal.SIGALRM, _raise_custom)
    try:
        # keep the synchronizer loop busy so the interrupt arrives before the task has been created
        loop.call_soon_threadsafe(time.sleep, 0.3)
        signal.setitimer(signal.ITIMER_REAL, 0.1)
        with pytest.raises(CustomBaseException):
            synchronizer._run_function_sync(_make_cancellation_recorder(states)())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert states == ["cancelled", "done"]


@pytest.mark.asyncio
async def test_run_function_async_cancelled_before_task_is_scheduled(synchronizer):
    states: list[str] = []
    loop = synchronizer._get_loop(start=True)
    # keep the synchronizer loop busy so the cancellation arrives before the task has been created
    loop.call_soon_threadsafe(time.sleep, 0.3)
    task = asyncio.create_task(synchronizer._run_function_async(_make_cancellation_recorder(states)()))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert states == ["cancelled", "done"]


@pytest.mark.asyncio
async def test_run_function_async_does_not_block_caller_loop_during_startup(synchronizer, monkeypatch):
    original_start_loop = synchronizer._start_loop

    def slow_start_loop():
        time.sleep(0.3)
        return original_start_loop()

    monkeypatch.setattr(synchronizer, "_start_loop", slow_start_loop)

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    async def foo():
        return "works"

    ticker_task = asyncio.create_task(ticker())
    try:
        assert await synchronizer._run_function_async(foo()) == "works"
    finally:
        ticker_task.cancel()
    # the caller's loop must keep running while the synchronizer loop starts up in the background
    assert ticks >= 5


@pytest.mark.filterwarnings("ignore")  # the synchronizer thread dies with an unhandled exception by design
def test_unexpected_thread_death(synchronizer, caplog):
    async def dummy():
        return "works"

    class CustomError(Exception):
        pass

    caplog.set_level(logging.ERROR)
    assert synchronizer._run_function_sync(dummy()) == "works"
    assert len(caplog.records) == 0

    def make_event_raise():
        # make the synchronizer thread's `await self._stopping.wait()` raise, as if the loop had died
        for waiter in synchronizer._stopping._waiters:
            if not waiter.done():
                waiter.set_exception(CustomError())

    synchronizer._loop.call_soon_threadsafe(make_event_raise)
    synchronizer._thread.join(timeout=5)
    coro = dummy()
    try:
        with pytest.raises(RuntimeError, match="Synchronizer thread unexpectedly died"):
            synchronizer._run_function_sync(coro)
    finally:
        coro.close()

    (error_log,) = caplog.records
    assert "Traceback" in error_log.message
    assert "CustomError" in error_log.message


@pytest.mark.skipif(sys.platform == "win32", reason="Windows can't fork")
def test_fork_during_start_loop_no_deadlock():
    with subprocess.Popen(
        [sys.executable, Path(__file__).parent / "_fork_lock_support.py"],
        encoding="utf8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as p:
        try:
            stdout, stderr = p.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            raise AssertionError("Fork lock test timed out (likely deadlocked)")

        assert p.returncode == 0, f"Fork lock test failed:\nstdout: {stdout}\nstderr: {stderr}"
        assert "PASS" in stdout
