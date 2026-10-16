# Copyright Modal Labs 2026
"""Unit tests for ``compile_modules``."""

from synchronicity2 import Module
from synchronicity2.codegen.pipeline import compile_modules

wrapper_module = Module("compile_modules_target")


@wrapper_module.wrap_function()
async def add(a: int, b: int) -> int:
    return a + b


def test_compile_modules_respects_runtime_package() -> None:
    custom = "my_library._vendored_synchronicity"
    out = compile_modules(
        [wrapper_module],
        synchronizer_module="my_library._generated_synchronizer",
        runtime_package=custom,
    )
    code = "\n".join(out.values())

    assert f"import {custom}.types" in code
    assert f"from {custom}.descriptor import" in code
    assert f"from {custom}.synchronizer import Synchronizer" in code
    assert "import my_library._generated_synchronizer" in code
    assert "get_synchronizer" not in code
    assert "import synchronicity2.types" not in code
