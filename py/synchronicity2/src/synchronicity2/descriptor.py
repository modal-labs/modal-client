# Copyright Modal Labs 2026
import typing
from typing import Any, Callable, Concatenate, ParamSpec, TypeVar, overload

from typing_extensions import Self

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")
WithAio_co = TypeVar("WithAio_co", covariant=True)


class classproperty(typing.Generic[T, R]):
    """Read-only class property descriptor.

    Unlike :class:`property`, this binds a ``@classmethod`` getter and is accessed on
    the wrapper class itself. Synchronicity codegen knows how to emit translated
    wrapper accessors for this descriptor.
    """

    fget: classmethod

    def __init__(self, fget: Callable[..., R] | classmethod):
        if not isinstance(fget, classmethod):
            fget = classmethod(fget)
        self.fget = fget

    @typing.overload
    def __get__(self, obj: None, owner: type[T]) -> R: ...

    @typing.overload
    def __get__(self, obj: T, owner: type[T]) -> R: ...

    def __get__(self, obj: typing.Optional[T], owner: type[T]) -> R:
        return self.fget.__get__(None, owner)()


class FunctionWithAio:
    """Shared runtime base for custom function-with-aio classes."""

    _sync_impl: Callable[..., typing.Any]

    def __init__(self, sync_impl: Callable[..., typing.Any]):
        self._sync_impl = sync_impl


class MethodWithAio:
    """Shared runtime base for generated and custom method-with-aio classes."""

    _sync_impl: Callable[..., typing.Any]
    _wrapper_instance: typing.Any
    _wrapper_class: type
    _with_aio_from_impl: Callable[[typing.Any], typing.Any]

    def __init__(
        self,
        sync_impl: Callable[..., typing.Any],
        wrapper_instance: typing.Any,
        wrapper_class: type,
        _from_impl: Callable[[typing.Any], typing.Any],
    ):
        self._sync_impl = sync_impl
        self._wrapper_instance = wrapper_instance
        self._wrapper_class = wrapper_class
        self._with_aio_from_impl = _from_impl

    def _from_impl(self, impl_instance: typing.Any) -> typing.Any:
        return self._with_aio_from_impl(impl_instance)


class _BoundInstanceMethodWithAio(typing.Protocol[WithAio_co]):
    @overload
    def __get__(self, wrapper_instance: None, owner: type) -> Self: ...

    @overload
    def __get__(self, wrapper_instance: typing.Any, owner: type) -> WithAio_co: ...


class _BoundOnAccessWithAio(typing.Protocol[WithAio_co]):
    def __get__(self, wrapper_instance: typing.Any | None, owner: type) -> WithAio_co: ...


class MethodWithAioDescriptor(typing.Generic[WithAio_co]):
    """Descriptor that materializes a generated with-aio helper object for instance methods."""

    _synchronicity_raw_class_dict = False
    sync_wrapper: Callable[..., typing.Any]
    with_aio_factory: Callable[..., WithAio_co]

    def __init__(self, sync_wrapper: Callable[..., typing.Any], with_aio_factory: Callable[..., WithAio_co]):
        self.sync_wrapper = sync_wrapper
        self.with_aio_factory = with_aio_factory

    @overload
    def __get__(self, wrapper_instance: None, owner: type) -> Self: ...

    @overload
    def __get__(self, wrapper_instance: typing.Any, owner: type) -> WithAio_co: ...

    def __get__(self, wrapper_instance, owner):
        if wrapper_instance is None:
            return self
        return self.with_aio_factory(
            sync_impl=self.sync_wrapper.__get__(wrapper_instance, owner),
            wrapper_instance=wrapper_instance,
            wrapper_class=owner,
            _from_impl=owner._from_impl,
        )


class OnAccessWithAioDescriptor(typing.Generic[WithAio_co]):
    """Descriptor that materializes a generated with-aio helper object on each access."""

    _synchronicity_raw_class_dict = True
    sync_wrapper: Callable[..., typing.Any]
    with_aio_factory: Callable[..., WithAio_co]

    def __init__(self, sync_wrapper: Callable[..., typing.Any], with_aio_factory: Callable[..., WithAio_co]):
        self.sync_wrapper = sync_wrapper
        self.with_aio_factory = with_aio_factory

    def __get__(self, wrapper_instance: typing.Any | None, owner: type) -> WithAio_co:
        return self.with_aio_factory(
            sync_impl=self.sync_wrapper.__get__(wrapper_instance, owner),
            wrapper_instance=wrapper_instance,
            wrapper_class=owner,
            _from_impl=owner._from_impl,
        )


def function_with_aio(
    with_aio_factory: Callable[..., WithAio_co],
) -> typing.Callable[[Callable[P, R]], WithAio_co]:
    def decorator(
        sync_wrapper: Callable[P, R],
    ) -> WithAio_co:
        return with_aio_factory(sync_impl=sync_wrapper)

    return decorator


def method_with_aio(
    with_aio_factory: Callable[..., WithAio_co],
) -> typing.Callable[[Callable[Concatenate[Any, P], R]], _BoundInstanceMethodWithAio[WithAio_co]]:
    def decorator(
        sync_wrapper: Callable[Concatenate[Any, P], R],
    ) -> _BoundInstanceMethodWithAio[WithAio_co]:
        return MethodWithAioDescriptor(sync_wrapper, with_aio_factory)

    return decorator


def classmethod_with_aio(
    with_aio_factory: Callable[..., WithAio_co],
) -> typing.Callable[
    [
        Callable[Concatenate[type[T], P], R],
    ],
    _BoundOnAccessWithAio[WithAio_co],
]:
    def decorator(
        sync_wrapper: Callable[Concatenate[type[T], P], R],
    ) -> _BoundOnAccessWithAio[WithAio_co]:
        return OnAccessWithAioDescriptor(sync_wrapper, with_aio_factory)

    return decorator


def staticmethod_with_aio(
    with_aio_factory: Callable[..., WithAio_co],
) -> typing.Callable[[Callable[P, R]], _BoundOnAccessWithAio[WithAio_co]]:
    def decorator(
        sync_wrapper: Callable[P, R],
    ) -> _BoundOnAccessWithAio[WithAio_co]:
        return OnAccessWithAioDescriptor(sync_wrapper, with_aio_factory)

    return decorator
