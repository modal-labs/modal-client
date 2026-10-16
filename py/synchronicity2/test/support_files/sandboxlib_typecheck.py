# Copyright Modal Labs 2026
from __future__ import annotations

from sandboxlib.sandbox import ContainerProcess, Sandbox
from typing_extensions import assert_type


def _sync_usage() -> None:
    sb = Sandbox.create()
    assert_type(sb, Sandbox)

    with sb:
        cp = sb.exec()
        assert_type(cp, ContainerProcess[str])
        cp.wait()

        for chunk in sb.stdout:
            assert_type(chunk, str)

        for chunk in sb.stdout.iter_read():
            assert_type(chunk, str)

        assert_type(cp.stdout.read(), str)
        assert_type(sb.exec(text=False), ContainerProcess[bytes])
        assert_type(sb.exec(text=False).stdout.read(), bytes)


async def _async_usage() -> None:
    sb = await Sandbox.create.aio()
    assert_type(sb, Sandbox)

    async with sb:
        cp = await sb.exec.aio()
        assert_type(cp, ContainerProcess[str])
        await cp.wait.aio()

        async for chunk in sb.stdout:
            assert_type(chunk, str)

        async for chunk in sb.stdout.iter_read.aio():
            assert_type(chunk, str)

        assert_type(await cp.stdout.read.aio(), str)
        assert_type(await sb.exec.aio(text=False), ContainerProcess[bytes])
        assert_type(await sb.exec.aio(False), ContainerProcess[bytes])
        assert_type(await (await sb.exec.aio(text=False)).stdout.read.aio(), bytes)
