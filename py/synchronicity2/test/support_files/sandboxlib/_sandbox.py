# Copyright Modal Labs 2026
import asyncio
import typing
from typing import Literal, Optional

import typing_extensions

from synchronicity2 import Module

sandbox_module = Module("sandboxlib.sandbox")


T = typing.TypeVar("T", str, bytes)


@sandbox_module.wrap_class()
class StreamReader(typing.Generic[T]):
    _value: T

    def __init__(self, value: T):
        self._value = value

    async def read(self) -> T:
        return self._value

    async def __aiter__(self) -> typing.AsyncGenerator[T, None]:
        yield self._value

    async def iter_read(self) -> typing.AsyncGenerator[T, None]:
        yield self._value


@sandbox_module.wrap_class()
class ContainerProcess(typing.Generic[T]):
    returncode: Optional[int] = None

    def __init__(self, _stdout_val: T):
        self._stdout_val = _stdout_val

    async def wait(self):
        await asyncio.sleep(0.1)
        self.returncode = 1

    @property
    def stdout(self) -> StreamReader[T]:
        return StreamReader(self._stdout_val)


@sandbox_module.wrap_class()
class Sandbox:
    @property
    def stdout(self) -> StreamReader[str]:
        return StreamReader("hello")

    @classmethod
    async def create(cls) -> "Sandbox":
        return Sandbox()

    @typing.overload
    async def exec(self, text: Literal[True] = True) -> ContainerProcess[str]: ...

    @typing.overload
    async def exec(self, text: Literal[False]) -> ContainerProcess[bytes]: ...

    async def exec(self, text: bool = True) -> ContainerProcess[str] | ContainerProcess[bytes]:
        return ContainerProcess("foo") if text else ContainerProcess(b"bar")

    async def detach(self):
        return None

    async def __aenter__(self) -> typing_extensions.Self:
        return self

    async def __aexit__(self, *args, **kwargs) -> None:
        await self.detach()
