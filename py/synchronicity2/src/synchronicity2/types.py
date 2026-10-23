# Copyright Modal Labs 2026
"""Dual-mode iterator types that support both sync and async iteration."""

from __future__ import annotations

import typing
from typing import AsyncIterator, Callable, Generic, Iterator, TypeVar

from .synchronizer import _anext

if typing.TYPE_CHECKING:
    from .synchronizer import Synchronizer

T = TypeVar("T")


class SyncOrAsyncIterator(Generic[T]):
    """Iterator that supports both sync (for) and async (async for) iteration.

    This allows the same iterator object to be used in both sync and async contexts,
    making it convenient for code that can work in either mode.

    Args:
        async_iterator: The underlying async iterator to wrap
        synchronizer: The synchronizer to use for converting async to sync
        item_wrapper: Optional function to wrap each yielded item
    """

    def __init__(
        self,
        async_iterator: AsyncIterator[typing.Any],
        synchronizer: "Synchronizer",
        item_wrapper: Callable[[typing.Any], T] | None = None,
    ):
        self._async_iterator = async_iterator
        self._synchronizer = synchronizer
        self._item_wrapper = item_wrapper
        self._sync_iterator: Iterator[T] | None = None

    def __iter__(self) -> Iterator[T]:
        """Return self for sync iteration protocol."""
        # Create a sync iterator wrapper using the synchronizer
        if self._sync_iterator is None:
            if wrap := self._item_wrapper:
                # Wrap each item as it comes through
                def wrapped_iter():
                    for item in self._synchronizer._run_iterator_sync(self._async_iterator):
                        yield wrap(item)

                self._sync_iterator = wrapped_iter()
            else:
                self._sync_iterator = self._synchronizer._run_iterator_sync(self._async_iterator)
        return self

    def __next__(self) -> T:
        """Get next item in sync iteration."""
        if self._sync_iterator is None:
            # Initialize iterator if not already done
            self.__iter__()
        try:
            return next(self._sync_iterator)  # type: ignore
        except StopAsyncIteration:
            raise StopIteration()

    def __aiter__(self) -> AsyncIterator[T]:
        """Return self for async iteration protocol."""
        return self

    async def __anext__(self) -> T:
        """Get next item in async iteration."""
        try:
            item = await self._synchronizer._run_function_async(_anext(self._async_iterator))
            if self._item_wrapper:
                return self._item_wrapper(item)
            return item
        except StopAsyncIteration:
            raise


class SyncOrAsyncContextManager(Generic[T]):
    """Context manager that supports both sync (with) and async (async with) usage.

    This allows the same context manager object to be used in both sync and async contexts.

    Args:
        async_cm: The underlying async context manager to wrap
        synchronizer: The synchronizer to use for converting async to sync
        value_wrapper: Optional function to wrap the value returned by __aenter__
    """

    def __init__(
        self,
        async_cm: typing.Any,
        synchronizer: "Synchronizer",
        value_wrapper: Callable[[typing.Any], T] | None = None,
    ):
        self._async_cm = async_cm
        self._synchronizer = synchronizer
        self._value_wrapper = value_wrapper

    def __enter__(self) -> T:
        result = self._synchronizer._run_function_sync(self._async_cm.__aenter__())
        if self._value_wrapper:
            return self._value_wrapper(result)
        return result  # type: ignore

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: typing.Any
    ) -> bool | None:
        return self._synchronizer._run_function_sync(self._async_cm.__aexit__(exc_type, exc_val, exc_tb))

    async def __aenter__(self) -> T:
        result = await self._synchronizer._run_function_async(self._async_cm.__aenter__())
        if self._value_wrapper:
            return self._value_wrapper(result)
        return result  # type: ignore

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: typing.Any
    ) -> bool | None:
        return await self._synchronizer._run_function_async(self._async_cm.__aexit__(exc_type, exc_val, exc_tb))


class SyncOrAsyncIterable(Generic[T]):
    """Iterable that supports both sync (for) and async (async for) iteration.

    This allows the same iterable object to be used in both sync and async contexts.
    Each call to __iter__ or __aiter__ creates a new iterator.

    Args:
        async_iterable: The underlying async iterable to wrap
        synchronizer: The synchronizer to use for converting async to sync
        item_wrapper: Optional function to wrap each yielded item from iterators
    """

    def __init__(
        self,
        async_iterable: typing.AsyncIterable[typing.Any],
        synchronizer: "Synchronizer",
        item_wrapper: Callable[[typing.Any], T] | None = None,
    ):
        self._async_iterable = async_iterable
        self._synchronizer = synchronizer
        self._item_wrapper = item_wrapper

    def __iter__(self) -> Iterator[T]:
        """Return a sync iterator."""
        # Call __aiter__ on the async iterable to get an async iterator
        async_iter = self._async_iterable.__aiter__()
        if wrap := self._item_wrapper:
            # Wrap each item as it comes through
            def wrapped_iter():
                for item in self._synchronizer._run_iterator_sync(async_iter):
                    yield wrap(item)

            return wrapped_iter()
        else:
            return self._synchronizer._run_iterator_sync(async_iter)

    async def __aiter__(self) -> AsyncIterator[T]:
        """Return an async iterator (which is self as an async generator)."""
        # Get the async iterator from the underlying iterable
        async_iter = self._async_iterable.__aiter__()

        # Iterate and yield items, wrapping if needed
        async for item in self._synchronizer._run_iterator_async(async_iter):
            if self._item_wrapper:
                yield self._item_wrapper(item)
            else:
                yield item
