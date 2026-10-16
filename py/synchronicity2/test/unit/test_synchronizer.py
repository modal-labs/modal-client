# Copyright Modal Labs 2026
import asyncio
import pytest
import typing
import weakref

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
