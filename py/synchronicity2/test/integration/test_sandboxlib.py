# Copyright Modal Labs 2026
"""Integration tests for the sandboxlib package scenario."""

from __future__ import annotations

import asyncio
from pathlib import Path

from test.integration.test_utils import check_pyright


def test_runtime() -> None:
    from sandboxlib.sandbox import ContainerProcess, Sandbox

    sb = Sandbox.create()
    assert isinstance(sb, Sandbox)

    with sb:
        cp = sb.exec()
        assert isinstance(cp, ContainerProcess)
        assert cp.stdout.read() == "foo"
        cp.wait()
        assert cp.returncode == 1

        assert list(sb.stdout) == ["hello"]
        assert list(sb.stdout.iter_read()) == ["hello"]

        binary_cp = sb.exec(text=False)
        assert binary_cp.stdout.read() == b"bar"

    async def _async_usage() -> None:
        sb = await Sandbox.create.aio()

        async with sb:
            cp = await sb.exec.aio()
            await cp.wait.aio()
            assert cp.returncode == 1

            stdout_chunks: list[str] = []
            async for chunk in sb.stdout:
                stdout_chunks.append(chunk)
            assert stdout_chunks == ["hello"]

            iter_chunks: list[str] = []
            async for chunk in sb.stdout.iter_read.aio():
                iter_chunks.append(chunk)
            assert iter_chunks == ["hello"]

            assert await cp.stdout.read.aio() == "foo"

            binary_cp = await sb.exec.aio(text=False)
            assert await binary_cp.stdout.read.aio() == b"bar"

    asyncio.run(_async_usage())


def test_pyright_implementation() -> None:
    import sandboxlib._sandbox

    check_pyright([Path(sandboxlib._sandbox.__file__)])


def test_pyright_wrapper() -> None:
    import sandboxlib.sandbox

    check_pyright([Path(sandboxlib.sandbox.__file__)])


def test_pyright_usage() -> None:
    from importlib.util import find_spec

    spec = find_spec("sandboxlib_typecheck")
    assert spec and spec.origin
    check_pyright([Path(spec.origin)])
