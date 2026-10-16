# Copyright Modal Labs 2026
"""Unit tests for _normalize_async_annotation() function."""

import collections.abc
import inspect
import typing

from synchronicity2.codegen.parsing.signatures import _normalize_async_annotation


# Test functions
async def async_func_with_return() -> int:
    return 42


async def async_func_no_annotation():
    return 42


def sync_func() -> int:
    return 42


async def async_gen_func() -> typing.AsyncGenerator[int, None]:
    yield 1


async def async_gen_func_unannotated():
    yield 1


def test_normalize_async_def_with_annotation():
    """Test that async def with return annotation becomes Awaitable[T]."""
    result = _normalize_async_annotation(async_func_with_return, int)

    # Check that it's an Awaitable
    assert typing.get_origin(result) is collections.abc.Awaitable

    # Check that the inner type is int
    args = typing.get_args(result)
    assert len(args) == 1
    assert args[0] is int


def test_normalize_async_def_no_annotation():
    """Test that async def without annotation becomes Awaitable[Any]."""
    result = _normalize_async_annotation(async_func_no_annotation, inspect.Signature.empty)

    # Check that it's an Awaitable
    assert typing.get_origin(result) is collections.abc.Awaitable

    # Check that the inner type is Any
    args = typing.get_args(result)
    assert len(args) == 1
    assert args[0] is typing.Any


def test_normalize_sync_function():
    """Test that sync functions are unchanged."""
    result = _normalize_async_annotation(sync_func, int)

    # Should return the annotation unchanged
    assert result is int


def test_normalize_async_generator():
    """Test that async generators are NOT wrapped in Awaitable."""
    annotation = typing.AsyncGenerator[int, None]
    result = _normalize_async_annotation(async_gen_func, annotation)

    # Should return the annotation unchanged (async generators are special)
    assert result == annotation


def test_normalize_async_generator_unannotated():
    """Test that unannotated async generators are unchanged."""
    result = _normalize_async_annotation(async_gen_func_unannotated, inspect.Signature.empty)

    # Should return empty signature unchanged
    assert result == inspect.Signature.empty


def test_normalize_explicit_awaitable():
    """Test that explicit Awaitable annotations are unchanged."""
    annotation = collections.abc.Awaitable[int]
    result = _normalize_async_annotation(sync_func, annotation)

    # Should return the annotation unchanged
    assert result == annotation


def test_normalize_explicit_coroutine():
    """Test that explicit Coroutine annotations are unchanged."""
    annotation = collections.abc.Coroutine[typing.Any, typing.Any, int]
    result = _normalize_async_annotation(sync_func, annotation)

    # Should return the annotation unchanged
    assert result == annotation


def test_normalize_preserves_complex_return_types():
    """Test that complex return types are preserved inside Awaitable."""

    async def async_func_complex() -> list[str]:
        return ["hello"]

    result = _normalize_async_annotation(async_func_complex, list[str])

    # Check that it's an Awaitable
    assert typing.get_origin(result) is collections.abc.Awaitable

    # Check that the inner type is list[str]
    args = typing.get_args(result)
    assert len(args) == 1
    inner_type = args[0]
    assert typing.get_origin(inner_type) is list
    inner_args = typing.get_args(inner_type)
    assert len(inner_args) == 1
    assert inner_args[0] is str
