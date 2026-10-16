# Copyright Modal Labs 2026
"""Module class for build-time registration of async code.

The Module class provides a lightweight way to mark async functions and classes
for wrapper code generation. It has zero runtime overhead - it simply tracks
what should be wrapped during the build step.
"""

import dataclasses
import inspect
import typing
from typing import Callable, Optional

F = typing.TypeVar("F")
C = typing.TypeVar("C", bound=type)
D = typing.TypeVar("D")

_IMPL_WRAPPER_LOCATION_ATTR = "__synchronicity_wrapper_location__"


def _direct_wrapper_location(impl_cls: type) -> tuple[str, str] | None:
    return impl_cls.__dict__.get(_IMPL_WRAPPER_LOCATION_ATTR)


def _inherited_wrapper_location(impl_cls: type) -> tuple[type, tuple[str, str]] | None:
    if _direct_wrapper_location(impl_cls) is not None:
        return None
    for base in impl_cls.__mro__[1:]:
        location = base.__dict__.get(_IMPL_WRAPPER_LOCATION_ATTR)
        if location is not None:
            return base, location
    return None


@dataclasses.dataclass(frozen=True)
class RegistrationInfo:
    target_module: str
    name: str
    include_underscored_methods: bool = False


@dataclasses.dataclass(frozen=True)
class ManualWrapperRef:
    module: str
    qualname: str


def _default_export_name(source_name: str, *, decorator_name: str) -> str:
    export_name = source_name.lstrip("_")
    if export_name:
        return export_name
    raise ValueError(
        f"{decorator_name}() could not derive a valid export name from source name {source_name!r}; "
        "pass name= explicitly"
    )


def _validate_registration_name(name: str | None, *, decorator_name: str, fallback: str) -> str:
    if name is None:
        return _default_export_name(fallback, decorator_name=decorator_name)
    if not isinstance(name, str):
        raise TypeError(f"{decorator_name}() name must be a string")
    if not name:
        raise ValueError(f"{decorator_name}() name must be a non-empty string")
    if not name.isidentifier():
        raise ValueError(f"{decorator_name}() name must be a valid Python identifier")
    return name


def _impl_ref_from_object(obj: object | None) -> ManualWrapperRef | None:
    if obj is None:
        return None
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return ManualWrapperRef(module=module, qualname=qualname)
    return None


def _resolve_manual_wrapper_ref(obj: object) -> ManualWrapperRef | None:
    for target in (
        getattr(obj, "sync_wrapper", None),
        getattr(obj, "_sync_impl", None),
        obj.__func__ if isinstance(obj, (classmethod, staticmethod)) else None,
        obj,
    ):
        ref = _impl_ref_from_object(target)
        if ref is not None:
            return ref
    return None


class Module:
    """Lightweight build-time registration for async code wrapper generation.

    The Module class is used to mark async functions and classes that should have
    synchronous wrappers generated. It only exists at build time and has no runtime
    dependencies on the Synchronizer class.

    Example:
        ```python
        from synchronicity2 import Module

        wrapper_module = Module("my_lib")

        @wrapper_module.wrap_function()
        async def my_function():
            return "hello"

        @wrapper_module.wrap_class()
        class MyClass:
            async def my_method(self):
                return "world"
        ```

    Attributes:
        _target_module: The module name where wrapper code will be generated
    """

    def __init__(
        self,
        target_module: Optional[str],
    ):
        """Initialize a Module for wrapper registration.

        Args:
            target_module: The module name where wrapper code will be generated.
                          Must be provided (auto-detection not yet implemented).
        Raises:
            NotImplementedError: If target_module is None (auto-detection not supported).
        """
        if not target_module:
            # TODO: Use call stack to infer calling module, if it's underscored
            # or suffixed with "_impl" -> default to output in same dir
            raise NotImplementedError("Auto module not implemented")
        self._target_module: str = target_module
        # Instance-level registries - each Module tracks its own wrapped items
        self._wrapped_classes: dict[type, RegistrationInfo] = {}
        self._wrapped_functions: dict[object, RegistrationInfo] = {}
        self._manual_wrappers: dict[int, ManualWrapperRef | None] = {}
        self._manual_exports: dict[str, ManualWrapperRef] = {}

    @property
    def target_module(self) -> str:
        """Get the target module name for code generation."""
        return self._target_module

    @property
    def _registered_classes(self) -> dict[type, RegistrationInfo]:
        """Get registered classes for this module."""
        return self._wrapped_classes

    @property
    def _registered_functions(self) -> dict[object, RegistrationInfo]:
        """Get registered functions for this module."""
        return self._wrapped_functions

    @property
    def _manual_wrapper_ids(self) -> frozenset[int]:
        """Get object identities registered for manual forwarding in this module."""
        return frozenset(self._manual_wrappers)

    def _is_manual_wrapper(self, obj: object) -> bool:
        """Return whether ``obj`` is registered for manual forwarding in this module."""
        return id(obj) in self._manual_wrappers

    def _manual_wrapper_ref(self, obj: object) -> ManualWrapperRef | None:
        """Get the implementation reference for a manually forwarded object, if any."""
        return self._manual_wrappers.get(id(obj))

    @property
    def _manual_export_refs(self) -> dict[str, ManualWrapperRef]:
        """Get named implementation exports that should be forwarded directly."""
        return self._manual_exports

    def _module_items(self) -> dict[object, RegistrationInfo]:
        """Get all registered classes and functions with their target module and name.

        Returns:
            Dict mapping registered items (classes or functions) to registration
            metadata for code generation.
        """
        result = {}
        result.update(self._registered_classes)
        result.update(self._registered_functions)
        return result

    def manual_wrapper(self) -> Callable[[D], D]:
        """Decorator to mark an entity for direct forwarding instead of wrapper generation."""

        def decorator(obj: D) -> D:
            self._manual_wrappers[id(obj)] = _resolve_manual_wrapper_ref(obj)
            return obj

        return decorator

    def manual_export(
        self,
        name: str,
        *,
        source_name: str | None = None,
        source_module: str | None = None,
    ) -> None:
        """Mark a module-level implementation name for direct forwarding.

        This is intended for constants, type aliases, and other public values that do not have
        a decorator target but should still appear in the generated public module.
        """

        export_name = _validate_registration_name(name, decorator_name="manual_export", fallback=name)
        impl_name = source_name or export_name
        _validate_registration_name(impl_name, decorator_name="manual_export", fallback=impl_name)

        if source_module is None:
            caller_frame = inspect.currentframe()
            if caller_frame is None or caller_frame.f_back is None:
                raise RuntimeError("manual_export() could not determine caller module")
            source_module_value = caller_frame.f_back.f_globals.get("__name__")
            if not isinstance(source_module_value, str):
                raise RuntimeError("manual_export() could not determine caller module")
            source_module = source_module_value
        if not isinstance(source_module, str) or not source_module:
            raise ValueError("manual_export() source_module must be a non-empty string")

        self._manual_exports[export_name] = ManualWrapperRef(module=source_module, qualname=impl_name)

    def wrap_function(self, *, name: str | None = None) -> Callable[[F], F]:
        """Decorator to mark a function for wrapper generation."""

        def decorator(fn: F) -> F:
            function_name = getattr(fn, "__name__", None)
            if not isinstance(function_name, str):
                ref = self._manual_wrapper_ref(fn)
                if ref is None:
                    raise TypeError(
                        "wrap_function() expects a function or a @module.manual_wrapper()-registered "
                        "with-aio helper with an underlying implementation reference"
                    )
                function_name = ref.qualname.rpartition(".")[2]
            export_name = _validate_registration_name(
                name,
                decorator_name="wrap_function",
                fallback=function_name,
            )
            self._wrapped_functions[fn] = RegistrationInfo(
                target_module=self._target_module,
                name=export_name,
            )
            return fn

        return decorator

    def wrap_class(
        self,
        *,
        name: str | None = None,
        include_underscored_methods: bool = False,
    ) -> Callable[[C], C]:
        """Decorator to mark a class for wrapper generation."""

        def decorator(impl_cls: C) -> C:
            export_name = _validate_registration_name(
                name,
                decorator_name="wrap_class",
                fallback=impl_cls.__name__,
            )
            registration = RegistrationInfo(
                target_module=self._target_module,
                name=export_name,
                include_underscored_methods=include_underscored_methods,
            )
            self._wrapped_classes[impl_cls] = registration

            if self._is_manual_wrapper(impl_cls):
                return impl_cls

            wrapper_location = (self._target_module, export_name)
            existing_location = impl_cls.__dict__.get(_IMPL_WRAPPER_LOCATION_ATTR)
            if existing_location is not None and existing_location != wrapper_location:
                raise RuntimeError(
                    f"Implementation class {impl_cls!r} already has wrapper location {existing_location!r}, "
                    f"cannot replace it with {wrapper_location!r}"
                )

            setattr(impl_cls, _IMPL_WRAPPER_LOCATION_ATTR, wrapper_location)
            return impl_cls

        return decorator
