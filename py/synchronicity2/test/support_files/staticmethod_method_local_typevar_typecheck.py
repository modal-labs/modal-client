# Copyright Modal Labs 2026
"""Consumer typing checks for async staticmethods with method-local type variables."""

import staticmethod_method_local_typevar
from typing_extensions import assert_type


def _sync_usage() -> None:
    assert_type(staticmethod_method_local_typevar.EchoBox.echo(3), int)
    assert_type(staticmethod_method_local_typevar.EchoBox.echo("hi"), str)


async def _async_usage() -> None:
    assert_type(await staticmethod_method_local_typevar.EchoBox.echo.aio(4), int)
    assert_type(await staticmethod_method_local_typevar.EchoBox.echo.aio("bye"), str)
