# Copyright Modal Labs 2026
"""Parse live Python annotations into data-only annotation IR."""

from __future__ import annotations

import collections.abc
import contextlib
import inspect
import pathlib
import types
import typing
import warnings

from synchronicity2.module import (
    _direct_wrapper_location,
    _inherited_wrapper_location,
)

from ..ir.annotations import (
    AnnotationIR,
    AsyncContextManagerAnnotationIR,
    AsyncGeneratorAnnotationIR,
    AsyncIterableAnnotationIR,
    AsyncIteratorAnnotationIR,
    AwaitableAnnotationIR,
    CallableAnnotationIR,
    CollectionAnnotationIR,
    CoroutineAnnotationIR,
    DictAnnotationIR,
    ListAnnotationIR,
    OptionalAnnotationIR,
    ParameterizedWrappedClassRefIR,
    PlainAnnotationIR,
    SelfAnnotationIR,
    SequenceAnnotationIR,
    SyncGeneratorAnnotationIR,
    Synchronicity1WrappedClassRefIR,
    SyncIteratorAnnotationIR,
    TupleAnnotationIR,
    TypeVarRefIR,
    UnionAnnotationIR,
    WrappedClassRefIR,
)
from ..ir.references import ObjectReferenceIR

if typing.TYPE_CHECKING:
    from synchronicity import Synchronizer as Synchronicity1Synchronizer


_CANONICAL_TYPE_REFS: dict[type, tuple[str, str]] = {
    pathlib.Path: ("pathlib", "Path"),
    pathlib.PurePath: ("pathlib", "PurePath"),
    pathlib.PurePosixPath: ("pathlib", "PurePosixPath"),
    pathlib.PureWindowsPath: ("pathlib", "PureWindowsPath"),
    pathlib.PosixPath: ("pathlib", "PosixPath"),
    pathlib.WindowsPath: ("pathlib", "WindowsPath"),
}


def _canonical_type_ref(annotation: type) -> tuple[str, str]:
    return _CANONICAL_TYPE_REFS.get(annotation, (annotation.__module__, annotation.__qualname__))


def _is_self_annotation(annotation: object) -> bool:
    if annotation is typing.Self:
        return True
    try:
        import typing_extensions

        return annotation is typing_extensions.Self
    except (ImportError, AttributeError):
        return False


def _format_annotation(annotation: object) -> str:
    """Format an unwrapped Python annotation for generated source."""
    if annotation == inspect.Signature.empty:
        return ""
    if annotation is type(None) or annotation is None:
        return "None"
    if annotation is Ellipsis:
        return "..."
    if isinstance(annotation, (typing.TypeVar, typing.ParamSpec)):
        return annotation.__name__
    if isinstance(annotation, list):
        return f"[{', '.join(_format_annotation(item) for item in annotation)}]"

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is not None:
        if not args:
            return repr(annotation)
        formatted_args = [_format_annotation(arg) for arg in args]
        origin_name = origin.__name__ if hasattr(origin, "__name__") else str(origin)
        origin_module = getattr(origin, "__module__", None)
        if origin in (list, dict, tuple, set, frozenset, type):
            return f"{origin_name}[{', '.join(formatted_args)}]"
        if origin_module in ("typing", "collections.abc"):
            return f"typing.{origin_name}[{', '.join(formatted_args)}]"
        if isinstance(origin, type) and origin_module not in ("builtins", "__builtin__"):
            canonical_module, canonical_qualname = _canonical_type_ref(origin)
            return f"{canonical_module}.{canonical_qualname}[{', '.join(formatted_args)}]"
        return f"{origin_name}[{', '.join(formatted_args)}]"

    if isinstance(annotation, type):
        if annotation.__module__ in ("builtins", "__builtin__"):
            return annotation.__name__
        canonical_module, canonical_qualname = _canonical_type_ref(annotation)
        return f"{canonical_module}.{canonical_qualname}"
    return repr(annotation)


def impl_qualified(t: type) -> ObjectReferenceIR:
    return ObjectReferenceIR(module=t.__module__, qualname=t.__qualname__)


def _get_wrapper_location(impl_type: type) -> tuple[str, str] | None:
    """Read the wrapper location from the marker attribute set by ``Module.wrap_class``."""
    return _direct_wrapper_location(impl_type)


def _is_synchronicity2_wrapped_impl(t: type) -> bool:
    return _direct_wrapper_location(t) is not None


def _synchronicity1_wrapper_ref(
    impl_type: type,
    synchronicity1_synchronizer: Synchronicity1Synchronizer | None,
) -> ObjectReferenceIR | None:
    if synchronicity1_synchronizer is None:
        return None
    wrapper_cls = synchronicity1_synchronizer._translate_out(impl_type)
    if wrapper_cls is impl_type:
        return None
    if not isinstance(wrapper_cls, type):
        raise TypeError(f"Synchronicity 1 translated implementation class {impl_type!r} to non-class {wrapper_cls!r}")
    return ObjectReferenceIR(wrapper_cls.__module__, wrapper_cls.__name__)


def _warn_if_inherited_wrapper_reference(annotation: object, source_label: str | None) -> None:
    if not isinstance(annotation, type):
        return
    inherited = _inherited_wrapper_location(annotation)
    if inherited is None:
        return
    base, _location = inherited
    prefix = f"{source_label}: " if source_label else ""
    warnings.warn(
        prefix
        + "type annotation references subclass "
        + f"{annotation.__module__}.{annotation.__qualname__} of wrapped implementation class "
        + f"{base.__module__}.{base.__qualname__}, but the subclass is not directly wrapped; "
        + "treating it as an unwrapped identity type",
        UserWarning,
        stacklevel=3,
    )


def _wrapper_ref_from_type(impl_type: type) -> ObjectReferenceIR:
    loc = _get_wrapper_location(impl_type)
    assert loc is not None
    return ObjectReferenceIR(*loc)


def resolve_typevar_bound_to_wrapped_impl(
    tv: typing.TypeVar,
    known_impl_types: frozenset[type],
    impl_modules: frozenset[str] | None,
    synchronicity1_synchronizer: Synchronicity1Synchronizer | None = None,
) -> ObjectReferenceIR | None:
    """Return the impl ref when *tv*'s bound is a synchronized class or forward-refers to one by name."""
    bound = getattr(tv, "__bound__", None)
    if bound is None:
        return None
    if isinstance(bound, type):
        if _is_synchronicity2_wrapped_impl(bound) or _synchronicity1_wrapper_ref(bound, synchronicity1_synchronizer):
            return impl_qualified(bound)
        return None
    bound_name: str | None = None
    if hasattr(bound, "__forward_arg__"):
        bound_name = bound.__forward_arg__  # type: ignore[assignment]
    elif isinstance(bound, str):
        bound_name = bound
    if bound_name is None:
        return None
    for t in known_impl_types:
        ref = impl_qualified(t)
        if ref.qualname.split(".")[-1] != bound_name:
            continue
        if impl_modules is not None and ref.module not in impl_modules:
            continue
        return ref
    return None


def annotation_import_modules(annotation: object) -> frozenset[str]:
    """Return non-builtin modules that the emitted annotation text will reference."""
    modules: set[str] = set()
    if annotation in (inspect.Signature.empty, None, type(None)):
        return frozenset()
    if isinstance(annotation, (typing.TypeVar, typing.ParamSpec)):
        return frozenset()
    if isinstance(annotation, list):
        for item in annotation:
            modules.update(annotation_import_modules(item))
        return frozenset(modules)

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is not None:
        origin_module = getattr(origin, "__module__", None)
        if isinstance(origin, type) and origin_module not in (
            None,
            "builtins",
            "__builtin__",
            "typing",
            "collections.abc",
        ):
            canonical_origin_module, _canonical_origin_qualname = _canonical_type_ref(origin)
            modules.add(canonical_origin_module)
        for arg in args:
            modules.update(annotation_import_modules(arg))
        return frozenset(modules)

    if isinstance(annotation, type):
        annotation_module, _annotation_qualname = _canonical_type_ref(annotation)
        if annotation_module not in (None, "builtins", "__builtin__", "typing"):
            modules.add(annotation_module)

    return frozenset(modules)


def _identity_ir_from_annotation(annotation: object) -> PlainAnnotationIR:
    return PlainAnnotationIR(
        signature_text=_format_annotation(annotation),
        import_modules=tuple(sorted(annotation_import_modules(annotation))),
    )


def _union_ir_contains_wrapped_runtime_arm(ir: AnnotationIR) -> bool:
    if isinstance(
        ir,
        (
            WrappedClassRefIR,
            Synchronicity1WrappedClassRefIR,
            ParameterizedWrappedClassRefIR,
            SelfAnnotationIR,
        ),
    ):
        return True
    if isinstance(ir, OptionalAnnotationIR):
        return _union_ir_contains_wrapped_runtime_arm(ir.inner_ir)
    return False


def _warn_if_callable_union_arm_precedes_public_annotation(
    item_irs: list[AnnotationIR], source_label: str | None
) -> None:
    first_wrapped_index = next(
        (index for index, ir in enumerate(item_irs) if _union_ir_contains_wrapped_runtime_arm(ir)),
        None,
    )
    if first_wrapped_index is None:
        return

    first_callable_index = next(
        (index for index, ir in enumerate(item_irs) if isinstance(ir, CallableAnnotationIR)), None
    )
    if first_callable_index is None or first_callable_index > first_wrapped_index:
        return

    prefix = f"{source_label}: " if source_label else ""
    warnings.warn(
        prefix
        + "Callable union arm appears before a wrapped-type arm. Union runtime resolution is greedy, so "
        + "a callable value may match the Callable arm before a synchronized wrapper arm. Put Callable arms "
        + "after wrapped types to avoid misclassification.",
        UserWarning,
        stacklevel=3,
    )


def parse_annotation(
    annotation: object,
    *,
    owner_impl_type: type | None = None,
    owner_has_type_parameters: bool = False,
    impl_modules: frozenset[str] | None = None,
    source_label: str | None = None,
    synchronicity1_synchronizer: Synchronicity1Synchronizer | None = None,
) -> AnnotationIR:
    """Build data-only :class:`AnnotationIR` from a resolved annotation."""
    if annotation == inspect.Signature.empty:
        return PlainAnnotationIR("")
    if annotation is None:
        return _identity_ir_from_annotation(None)

    if hasattr(annotation, "__forward_arg__"):
        forward_str = annotation.__forward_arg__  # type: ignore
        raise TypeError(
            f"Found unresolved forward reference '{forward_str}' in type annotation. "
            f"Use inspect.get_annotations(eval_str=True) to resolve forward references."
        )

    _warn_if_inherited_wrapper_reference(annotation, source_label)

    if isinstance(annotation, type) and _is_synchronicity2_wrapped_impl(annotation):
        return WrappedClassRefIR(impl_qualified(annotation), _wrapper_ref_from_type(annotation))

    if isinstance(annotation, type):
        synchronicity1_wrapper_ref = _synchronicity1_wrapper_ref(annotation, synchronicity1_synchronizer)
        if synchronicity1_wrapper_ref is not None:
            return Synchronicity1WrappedClassRefIR(impl_qualified(annotation), synchronicity1_wrapper_ref)

    if isinstance(annotation, typing.TypeVar):
        return TypeVarRefIR(name=annotation.__name__)

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is None:
        if (
            _is_self_annotation(annotation)
            and owner_impl_type is not None
            and _is_synchronicity2_wrapped_impl(owner_impl_type)
        ):
            return SelfAnnotationIR(impl_qualified(owner_impl_type), _wrapper_ref_from_type(owner_impl_type))
        return _identity_ir_from_annotation(annotation)

    if origin is list:
        if args:
            return ListAnnotationIR(
                parse_annotation(
                    args[0],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                )
            )
        return _identity_ir_from_annotation(annotation)

    if origin is dict:
        if len(args) >= 2:
            return DictAnnotationIR(
                parse_annotation(
                    args[0],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                ),
                parse_annotation(
                    args[1],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                ),
            )
        return _identity_ir_from_annotation(annotation)

    if origin is collections.abc.Sequence:
        if args:
            return SequenceAnnotationIR(
                parse_annotation(
                    args[0],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                )
            )
        return _identity_ir_from_annotation(annotation)

    if origin is collections.abc.Collection:
        if args:
            return CollectionAnnotationIR(
                parse_annotation(
                    args[0],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                )
            )
        return _identity_ir_from_annotation(annotation)

    if origin is tuple:
        if args:
            if Ellipsis in args:
                return TupleAnnotationIR(
                    (
                        parse_annotation(
                            args[0],
                            owner_impl_type=owner_impl_type,
                            owner_has_type_parameters=owner_has_type_parameters,
                            impl_modules=impl_modules,
                            source_label=source_label,
                            synchronicity1_synchronizer=synchronicity1_synchronizer,
                        ),
                    ),
                    variadic=True,
                )
            return TupleAnnotationIR(
                tuple(
                    parse_annotation(
                        arg,
                        owner_impl_type=owner_impl_type,
                        owner_has_type_parameters=owner_has_type_parameters,
                        impl_modules=impl_modules,
                        source_label=source_label,
                        synchronicity1_synchronizer=synchronicity1_synchronizer,
                    )
                    for arg in args
                ),
                variadic=False,
            )
        return _identity_ir_from_annotation(annotation)

    if origin in (typing.Union, types.UnionType):
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1 and type(None) in args:
            return OptionalAnnotationIR(
                parse_annotation(
                    non_none_args[0],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                )
            )
        item_irs = [
            parse_annotation(
                arg,
                owner_impl_type=owner_impl_type,
                owner_has_type_parameters=owner_has_type_parameters,
                impl_modules=impl_modules,
                source_label=source_label,
                synchronicity1_synchronizer=synchronicity1_synchronizer,
            )
            for arg in args
        ]
        _warn_if_callable_union_arm_precedes_public_annotation(item_irs, source_label)
        return UnionAnnotationIR(
            tuple(item_irs),
            source_label=source_label,
        )

    if origin is collections.abc.Generator:
        if args:
            return SyncGeneratorAnnotationIR(
                yield_annotation_ir=parse_annotation(
                    args[0],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                ),
                send_annotation_ir=parse_annotation(
                    args[1] if len(args) > 1 else type(None),
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                ),
                return_annotation_ir=parse_annotation(
                    args[2] if len(args) > 2 else type(None),
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                ),
            )
        return _identity_ir_from_annotation(annotation)

    if origin is collections.abc.Iterator:
        if args:
            return SyncIteratorAnnotationIR(
                parse_annotation(
                    args[0],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                )
            )
        return _identity_ir_from_annotation(annotation)

    if origin is collections.abc.AsyncIterator:
        if args:
            return AsyncIteratorAnnotationIR(
                parse_annotation(
                    args[0],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                )
            )
        return AsyncIteratorAnnotationIR(_identity_ir_from_annotation(typing.Any))

    if origin is collections.abc.AsyncIterable:
        if args:
            return AsyncIterableAnnotationIR(
                parse_annotation(
                    args[0],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                )
            )
        return AsyncIterableAnnotationIR(_identity_ir_from_annotation(typing.Any))

    if origin is collections.abc.AsyncGenerator:
        if len(args) >= 1:
            return AsyncGeneratorAnnotationIR(
                yield_annotation_ir=parse_annotation(
                    args[0],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                ),
                send_annotation_ir=(
                    parse_annotation(
                        args[1],
                        owner_impl_type=owner_impl_type,
                        owner_has_type_parameters=owner_has_type_parameters,
                        impl_modules=impl_modules,
                        source_label=source_label,
                        synchronicity1_synchronizer=synchronicity1_synchronizer,
                    )
                    if len(args) > 1
                    else None
                ),
            )
        return _identity_ir_from_annotation(annotation)

    if origin is collections.abc.Coroutine:
        if args and len(args) >= 3:
            return CoroutineAnnotationIR(
                parse_annotation(
                    args[2],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                )
            )
        return CoroutineAnnotationIR(_identity_ir_from_annotation(typing.Any))

    if origin is collections.abc.Awaitable:
        if args:
            return AwaitableAnnotationIR(
                parse_annotation(
                    args[0],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                )
            )
        return AwaitableAnnotationIR(_identity_ir_from_annotation(typing.Any))

    if origin is contextlib.AbstractAsyncContextManager:
        if args:
            return AsyncContextManagerAnnotationIR(
                value_ir=parse_annotation(
                    args[0],
                    owner_impl_type=owner_impl_type,
                    owner_has_type_parameters=owner_has_type_parameters,
                    impl_modules=impl_modules,
                    source_label=source_label,
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                )
            )
        return AsyncContextManagerAnnotationIR(value_ir=_identity_ir_from_annotation(typing.Any))

    if origin is collections.abc.Callable:
        if len(args) >= 2:
            params = args[0]
            return_type = parse_annotation(
                args[1],
                owner_impl_type=owner_impl_type,
                owner_has_type_parameters=owner_has_type_parameters,
                impl_modules=impl_modules,
                source_label=source_label,
                synchronicity1_synchronizer=synchronicity1_synchronizer,
            )
            if params is Ellipsis:
                return CallableAnnotationIR(None, return_type)
            if isinstance(params, list | tuple):
                return CallableAnnotationIR(
                    tuple(
                        parse_annotation(
                            param,
                            owner_impl_type=owner_impl_type,
                            owner_has_type_parameters=owner_has_type_parameters,
                            impl_modules=impl_modules,
                            source_label=source_label,
                            synchronicity1_synchronizer=synchronicity1_synchronizer,
                        )
                        for param in params
                    ),
                    return_type,
                )
            params_identity_ir = _identity_ir_from_annotation(params)
            return CallableAnnotationIR(
                None,
                return_type,
                params_signature_text=params_identity_ir.signature_text,
                params_signature_import_modules=params_identity_ir.import_modules,
            )
        return _identity_ir_from_annotation(annotation)

    # Subscripted wrapped class, e.g. SomeContainer[WrappedType]
    if isinstance(origin, type) and args:
        arg_irs = tuple(
            parse_annotation(
                arg,
                owner_impl_type=owner_impl_type,
                owner_has_type_parameters=owner_has_type_parameters,
                impl_modules=impl_modules,
                source_label=source_label,
                synchronicity1_synchronizer=synchronicity1_synchronizer,
            )
            for arg in args
        )
        if _is_synchronicity2_wrapped_impl(origin):
            return ParameterizedWrappedClassRefIR(
                WrappedClassRefIR(impl_qualified(origin), _wrapper_ref_from_type(origin)),
                arg_irs,
            )
        synchronicity1_wrapper_ref = _synchronicity1_wrapper_ref(origin, synchronicity1_synchronizer)
        if synchronicity1_wrapper_ref is not None:
            return ParameterizedWrappedClassRefIR(
                Synchronicity1WrappedClassRefIR(impl_qualified(origin), synchronicity1_wrapper_ref),
                arg_irs,
            )
    return _identity_ir_from_annotation(annotation)
