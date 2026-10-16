# Copyright Modal Labs 2026
import typing

import synchronicity2

P = typing.ParamSpec("P")
T = typing.TypeVar("T")

mod = synchronicity2.Module("decorator_factory")


@mod.wrap_class()
class RemoteFunction(typing.Generic[P, T]):
    def __init__(self, fn: typing.Callable[P, T]):
        self._fn = fn

    def remote(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self._fn(*args, **kwargs)


@mod.wrap_class()
class FunctionDecoratorType:
    @typing.overload
    def __call__(
        self, func: typing.Callable[P, typing.Coroutine[typing.Any, typing.Any, T]]
    ) -> RemoteFunction[P, T]: ...

    @typing.overload
    def __call__(self, func: typing.Callable[P, T]) -> RemoteFunction[P, T]: ...

    def __call__(self, func: typing.Any) -> typing.Any:
        return RemoteFunction(func)


@mod.wrap_class()
class Registry:
    def function(self) -> FunctionDecoratorType:
        return FunctionDecoratorType()
