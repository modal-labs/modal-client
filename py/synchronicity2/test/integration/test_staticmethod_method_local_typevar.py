# Copyright Modal Labs 2026
"""Integration tests for async staticmethods with method-local type variables."""

import asyncio
from importlib.util import find_spec
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime():
    import staticmethod_method_local_typevar

    assert staticmethod_method_local_typevar.EchoBox.echo(3) == 3
    assert staticmethod_method_local_typevar.EchoBox.echo("hi") == "hi"

    async def test_async():
        assert await staticmethod_method_local_typevar.EchoBox.echo.aio(4) == 4
        assert await staticmethod_method_local_typevar.EchoBox.echo.aio("bye") == "bye"

    asyncio.run(test_async())


def test_generated_wrapper_source():
    import staticmethod_method_local_typevar

    wrapper_source = Path(staticmethod_method_local_typevar.__file__).read_text()

    assert "@staticmethod_with_aio(_EchoBox_echo_MethodWithAio)" in wrapper_source
    assert "if typing.TYPE_CHECKING" not in wrapper_source


def test_pyright_implementation():
    import staticmethod_method_local_typevar_impl

    check_pyright([Path(staticmethod_method_local_typevar_impl.__file__)])


def test_pyright_wrapper():
    import staticmethod_method_local_typevar

    check_pyright([Path(staticmethod_method_local_typevar.__file__)])


def test_pyright_usage():
    spec = find_spec("staticmethod_method_local_typevar_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
