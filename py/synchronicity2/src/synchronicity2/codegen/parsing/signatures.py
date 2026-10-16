# Copyright Modal Labs 2026
"""Parse callable annotations and signatures into declaration IR."""

from __future__ import annotations

import collections.abc
import inspect
import sys
import types
import typing
import warnings

from ..ir.annotations import (
    AnnotationIR,
    CallableAnnotationIR,
    ParameterizedWrappedClassRefIR,
    SelfAnnotationIR,
    Synchronicity1WrappedClassRefIR,
    WrappedClassRefIR,
    walk_annotation_irs,
)
from ..ir.declarations import ParameterIR
from .annotations import parse_annotation
from .defaults import resolve_parameter_default_expressions

if typing.TYPE_CHECKING:
    from synchronicity import Synchronizer as Synchronicity1Synchronizer


def is_async_generator(func_or_method: object, return_annotation: object) -> bool:
    """Whether a callable or its annotation represents an async generator."""
    if inspect.isasyncgenfunction(func_or_method):
        return True
    if return_annotation == inspect.Signature.empty:
        return False
    return hasattr(return_annotation, "__origin__") and return_annotation.__origin__ is collections.abc.AsyncGenerator


def returns_awaitable(return_annotation: object) -> bool:
    """Whether an annotation represents a directly awaitable result."""
    if return_annotation == inspect.Signature.empty:
        return False
    awaitable_origins = (collections.abc.Coroutine, collections.abc.Awaitable)
    if hasattr(return_annotation, "__origin__"):
        return return_annotation.__origin__ in awaitable_origins
    if return_annotation in awaitable_origins:
        return True
    return typing.get_origin(return_annotation) in awaitable_origins


def returns_async_iterable_type(return_annotation: object) -> bool:
    """Whether an annotation represents an async iterator or iterable."""
    if return_annotation == inspect.Signature.empty:
        return False
    async_iterable_origins = (collections.abc.AsyncIterator, collections.abc.AsyncIterable)
    if hasattr(return_annotation, "__origin__"):
        return return_annotation.__origin__ in async_iterable_origins
    if return_annotation in async_iterable_origins:
        return True
    return typing.get_origin(return_annotation) in async_iterable_origins


def _normalize_async_annotation(func, return_annotation):
    """
    Normalize async function annotations to Awaitable[T] for uniform handling.

    Converts `async def f() -> T` into `def f() -> Awaitable[T]` at the annotation level,
    allowing annotation code generation to handle async/sync output uniformly.

    Args:
        func: The function or method object to check
        return_annotation: The return type annotation (may be inspect.Signature.empty)

    Returns:
        The normalized annotation (wrapped in Awaitable if async, otherwise unchanged)

    Note:
        Async generators are NOT wrapped in Awaitable - they remain as AsyncGenerator[T].
    """
    # Check if it's an async generator - these stay as-is
    # Async generators are special: they're defined with `async def` but they're NOT awaitable
    if is_async_generator(func, return_annotation):
        return return_annotation

    # Check if it's an async function (async def)
    if inspect.iscoroutinefunction(func):
        # Wrap in Awaitable[T]
        if return_annotation == inspect.Signature.empty:
            # No annotation -> Awaitable[Any]
            return collections.abc.Awaitable[typing.Any]
        else:
            # Has annotation T -> Awaitable[T]
            return collections.abc.Awaitable[return_annotation]

    # Already has explicit Awaitable/Coroutine, or is sync - return as-is
    return return_annotation


def _safe_get_annotations(obj, globals_dict=None, *, forbidden_wrapper_modules: frozenset[str] | None = None):
    """
    Safely get annotations, with fallback for forward references under TYPE_CHECKING.

    For forward references that can't be resolved (NameError), we try to import the
    module from fully qualified names (e.g., "my_mod.SomeType").
    """
    try:
        return inspect.get_annotations(obj, eval_str=True, globals=globals_dict)
    except NameError:
        # Forward reference can't be resolved - try importing from qualified names
        # Get raw string annotations
        raw_annotations = inspect.get_annotations(obj, eval_str=False, globals=globals_dict)

        # Build an extended globals dict with imports for qualified names
        extended_globals = (globals_dict or {}).copy()

        for key, annotation_str in raw_annotations.items():
            if isinstance(annotation_str, str) and "." in annotation_str:
                # Extract module path from qualified name (e.g., "my_mod.sub.SomeType" -> "my_mod.sub")
                parts = annotation_str.split(".")
                if len(parts) >= 2:
                    # Import the full module path (all parts except the last, which is the class name)
                    module_path = ".".join(parts[:-1])
                    try:
                        # Try to import the module
                        import importlib

                        if forbidden_wrapper_modules is not None and module_path in forbidden_wrapper_modules:
                            raise TypeError(
                                f"Implementation annotations may not import generated wrapper module {module_path!r}. "
                                "Import implementation modules instead."
                            )
                        imported_module = importlib.import_module(module_path)
                        # Ensure parent packages expose the imported submodule as an attribute
                        # so expressions like "modal._app._App" can be evaluated.
                        if "." in module_path:
                            parent_path, child_name = module_path.rsplit(".", 1)
                            parent_module = sys.modules.get(parent_path)
                            if parent_module is not None and not hasattr(parent_module, child_name):
                                setattr(parent_module, child_name, imported_module)
                        # Add the top-level module to extended_globals
                        # For "a.b.c.Class", add "a" -> sys.modules["a"]
                        top_level_module = parts[0]
                        if top_level_module not in extended_globals:
                            extended_globals[top_level_module] = sys.modules.get(top_level_module)
                    except ImportError:
                        pass  # Skip if module can't be imported

        # Try again with extended globals
        try:
            return inspect.get_annotations(obj, eval_str=True, globals=extended_globals)
        except (NameError, AttributeError):
            # Still can't resolve - return string annotations
            return raw_annotations


def _contains_self_type(annotation) -> bool:
    """Check if a type annotation contains typing.Self.

    Args:
        annotation: Type annotation to check

    Returns:
        True if typing.Self is found anywhere in the annotation
    """
    # Check for typing.Self directly
    if annotation is typing.Self:
        return True

    # Check for generic types with typing.Self as an argument
    origin = typing.get_origin(annotation)
    if origin is not None:
        args = typing.get_args(annotation)
        for arg in args:
            if _contains_self_type(arg):
                return True

    return False


def _extract_typevars_from_annotation(annotation, collected: dict[str, typing.TypeVar | typing.ParamSpec]) -> None:
    """Recursively extract TypeVar and ParamSpec instances from a type annotation."""
    # Handle TypeVar and ParamSpec directly
    if isinstance(annotation, typing.TypeVar):
        collected[annotation.__name__] = annotation
        return
    if isinstance(annotation, typing.ParamSpec):
        collected[annotation.__name__] = annotation
        return

    # Recursively process generic types
    args = typing.get_args(annotation)

    if args:
        for arg in args:
            _extract_typevars_from_annotation(arg, collected)


def _extract_typevars_from_function(
    f: types.FunctionType, annotations: dict[str, typing.Any]
) -> dict[str, typing.TypeVar | typing.ParamSpec]:
    """Extract all TypeVar and ParamSpec instances used in a function's signature."""
    collected: dict[str, typing.TypeVar | typing.ParamSpec] = {}

    # Extract from all annotations (parameters and return type)
    for annotation in annotations.values():
        _extract_typevars_from_annotation(annotation, collected)

    return collected


def parse_parameters_to_ir(
    func: types.FunctionType,
    sig: inspect.Signature,
    annotations: dict,
    *,
    impl_module: types.ModuleType,
    skip_first_param: bool = False,
    owner_impl_type: type | None = None,
    owner_has_type_parameters: bool = False,
    impl_modules: frozenset[str] | None = None,
    source_label_prefix: str | None = None,
    synchronicity1_synchronizer: Synchronicity1Synchronizer | None = None,
) -> tuple[ParameterIR, ...]:
    """Collect :class:`ParameterIR` from a signature (no emission strings)."""
    resolved_defaults = resolve_parameter_default_expressions(
        func,
        sig,
        impl_module=impl_module,
        source_label_prefix=source_label_prefix,
    )

    result: list[ParameterIR] = []
    for i, (name, param) in enumerate(sig.parameters.items()):
        if i == 0 and skip_first_param:
            continue

        param_annotation = annotations.get(name, param.annotation)
        if param_annotation is inspect.Signature.empty or param_annotation is param.empty:
            annotation_ir = None
        else:
            annotation_ir = parse_annotation(
                param_annotation,
                owner_impl_type=owner_impl_type,
                owner_has_type_parameters=owner_has_type_parameters,
                impl_modules=impl_modules,
                source_label=(f"{source_label_prefix} parameter {name!r}" if source_label_prefix is not None else None),
                synchronicity1_synchronizer=synchronicity1_synchronizer,
            )
            if isinstance(annotation_ir, CallableAnnotationIR) and _annotation_ir_contains_wrapped_refs(annotation_ir):
                warnings.warn(
                    (
                        f"{source_label_prefix or func.__qualname__} parameter {name!r} is a callable containing "
                        "synchronized types. Callable-valued parameters are passed through unchanged at runtime, "
                        "so generated wrapper annotations use implementation types inside the callable signature."
                    ),
                    UserWarning,
                    stacklevel=2,
                )

        default_expr: str | None = None
        default_import_modules: tuple[str, ...] = ()
        if param.default is not inspect.Parameter.empty:
            resolved_default = resolved_defaults[name]
            default_expr = resolved_default.expression
            default_import_modules = resolved_default.import_modules

        result.append(
            ParameterIR(
                name=name,
                kind=int(param.kind),
                annotation_ir=annotation_ir,
                default_expr=default_expr,
                default_import_modules=default_import_modules,
            )
        )
    return tuple(result)


def _annotation_ir_contains_wrapped_refs(ir: AnnotationIR) -> bool:
    wrapped_ref_types = (
        WrappedClassRefIR,
        Synchronicity1WrappedClassRefIR,
        ParameterizedWrappedClassRefIR,
        SelfAnnotationIR,
    )
    return any(isinstance(referenced_ir, wrapped_ref_types) for referenced_ir in walk_annotation_irs(ir))
