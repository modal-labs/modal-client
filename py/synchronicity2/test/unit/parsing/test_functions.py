# Copyright Modal Labs 2026
"""Parse-only tests: live functions → :class:`~synchronicity2.codegen.ir.WrappedFunctionIR`.

No emission; integration tests cover parse+emit end-to-end.
"""

from __future__ import annotations

import typing
from typing import Awaitable, Coroutine

from synchronicity2.codegen.ir.annotations import AwaitableAnnotationIR, CoroutineAnnotationIR
from synchronicity2.codegen.parsing.declarations import parse_wrapped_function


def test_parse_sync_function_no_async_wrapper():
    def add(a: int, b: int) -> int:
        return a + b

    ir = parse_wrapped_function(add, "m", globals_dict=globals())
    assert ir.needs_async_wrapper is False
    assert ir.is_async_gen is False


def test_parse_sync_function_returning_coroutine_needs_async_wrapper():
    def create_c(x: int) -> Coroutine[typing.Any, typing.Any, str]: ...

    ir = parse_wrapped_function(create_c, "m", globals_dict=globals())
    assert ir.needs_async_wrapper is True
    assert isinstance(ir.return_annotation_ir, CoroutineAnnotationIR)


def test_parse_sync_function_returning_awaitable_needs_async_wrapper():
    def create_a(x: int) -> Awaitable[str]: ...

    ir = parse_wrapped_function(create_a, "m", globals_dict=globals())
    assert ir.needs_async_wrapper is True
    assert isinstance(ir.return_annotation_ir, AwaitableAnnotationIR)
