# Copyright Modal Labs 2026
"""Integration tests for manual-wrapper forwarding."""

import asyncio
from importlib.util import find_spec
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import manual_nowrap
    import manual_nowrap_impl

    assert manual_nowrap.manual_function(1) == "manual-function-sync:1"

    async def check_async_function() -> str:
        return await manual_nowrap.manual_function.aio(2)

    assert asyncio.run(check_async_function()) == "manual-function-aio:2"

    box = manual_nowrap.ManualBox("box")
    assert box.manual_method(3) == "box:manual-method-sync:3"
    assert box.auto_method(4) == "box:auto-method:4"

    async def check_async_methods() -> tuple[str, str]:
        manual_value = await box.manual_method.aio(5)
        auto_value = await box.auto_method.aio(6)
        return manual_value, auto_value

    assert asyncio.run(check_async_methods()) == (
        "box:manual-method-aio:5",
        "box:auto-method:6",
    )

    assert manual_nowrap.DirectBox is manual_nowrap_impl.DirectBox
    assert manual_nowrap.DirectBox("value").reveal() == "direct:value"


def test_generated_wrapper_contains_manual_reexports():
    import manual_nowrap

    source = Path(manual_nowrap.__file__).read_text()

    assert "from synchronicity2.descriptor import (" in source
    assert "MethodWithAio," in source
    assert "manual_function = manual_nowrap_impl.manual_function" in source
    assert "DirectBox = manual_nowrap_impl.DirectBox" in source
    assert "manual_method = manual_nowrap_impl.ManualBox.manual_method" in source
    assert "class _manual_function_FunctionWithAio:" not in source
    assert "class _ManualBox_manual_method_MethodWithAio:" not in source
    assert "class _ManualBox_auto_method_MethodWithAio(MethodWithAio):" in source


def test_pyright_implementation():
    import manual_nowrap_impl

    check_pyright([Path(manual_nowrap_impl.__file__)])


def test_pyright_wrapper():
    import manual_nowrap

    check_pyright([Path(manual_nowrap.__file__)])


def test_pyright_usage():
    spec = find_spec("manual_nowrap_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
