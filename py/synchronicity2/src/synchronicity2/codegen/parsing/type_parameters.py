# Copyright Modal Labs 2026
"""Parse TypeVar and ParamSpec declarations into IR."""

from __future__ import annotations

import typing
import warnings

from synchronicity2.module import (
    _direct_wrapper_location,
    _inherited_wrapper_location,
)

from ..ir.annotations import AnnotationIR, Synchronicity1WrappedClassRefIR, WrappedClassRefIR
from ..ir.declarations import TypeParameterIR
from ..ir.references import ObjectReferenceIR
from .annotations import (
    _canonical_type_ref,
    _synchronicity1_wrapper_ref,
    annotation_import_modules,
    resolve_typevar_bound_to_wrapped_impl,
)

if typing.TYPE_CHECKING:
    from synchronicity import Synchronizer as Synchronicity1Synchronizer


def _get_wrapper_location(t: type) -> tuple[str, str] | None:
    return _direct_wrapper_location(t)


def _warn_if_inherited_wrapper_bound(bound: type) -> None:
    inherited = _inherited_wrapper_location(bound)
    if inherited is None:
        return
    base, _location = inherited
    warnings.warn(
        "TypeVar bound references subclass "
        + f"{bound.__module__}.{bound.__qualname__} of wrapped implementation class "
        + f"{base.__module__}.{base.__qualname__}, but the subclass is not directly wrapped; "
        + "treating it as an unwrapped identity type",
        UserWarning,
        stacklevel=3,
    )


def translate_typevar_bound(
    bound: type | str,
    known_impl_types: frozenset[type],
    target_module: str,
    *,
    impl_modules: frozenset[str] | None = None,
) -> str:
    def _iter_matching_name(name: str):
        for impl_type in known_impl_types:
            if impl_type.__name__ != name:
                continue
            if impl_modules is not None and impl_type.__module__ not in impl_modules:
                continue
            loc = _get_wrapper_location(impl_type)
            if loc is not None:
                yield impl_type, loc

    if hasattr(bound, "__forward_arg__"):
        forward_str = bound.__forward_arg__  # type: ignore
        matches = list(_iter_matching_name(forward_str))
        if matches:
            _, (wrapper_target_module, wrapper_name) = matches[0]
            if wrapper_target_module == target_module:
                return f'"{wrapper_name}"'
            return f'"{wrapper_target_module}.{wrapper_name}"'
        return f'"{forward_str}"'

    if isinstance(bound, str):
        matches = list(_iter_matching_name(bound))
        if matches:
            _, (wrapper_target_module, wrapper_name) = matches[0]
            if wrapper_target_module == target_module:
                return wrapper_name
            return f"{wrapper_target_module}.{wrapper_name}"
        return f'"{bound}"'

    if isinstance(bound, type):
        _warn_if_inherited_wrapper_bound(bound)
        loc = _get_wrapper_location(bound)
        if loc is not None:
            wrapper_target_module, wrapper_name = loc
            if wrapper_target_module == target_module:
                return f'"{wrapper_name}"'
            return f'"{wrapper_target_module}.{wrapper_name}"'
        canonical_module, canonical_qualname = _canonical_type_ref(bound)
        if canonical_module == "builtins":
            return bound.__name__
        return f"{canonical_module}.{canonical_qualname}"

    return repr(bound)


def type_parameter_irs_from_collected(
    module_typevars: dict[str, typing.TypeVar | typing.ParamSpec],
    known_impl_types: frozenset[type],
    target_module: str,
    *,
    impl_modules: frozenset[str] | None = None,
    synchronicity1_synchronizer: Synchronicity1Synchronizer | None = None,
) -> tuple[TypeParameterIR, ...]:
    specs: list[TypeParameterIR] = []
    for name in sorted(module_typevars.keys()):
        tv = module_typevars[name]
        if isinstance(tv, typing.ParamSpec):
            specs.append(
                TypeParameterIR(
                    name=name,
                    is_paramspec=True,
                    constraint_parts=(),
                    bound_value=None,
                    covariant=False,
                    contravariant=False,
                    bound_annotation_ir=None,
                    import_modules=(),
                )
            )
            continue

        constraint_parts: list[str] = []
        import_modules: set[str] = set()
        if hasattr(tv, "__constraints__") and tv.__constraints__:
            for constraint in tv.__constraints__:
                import_modules.update(annotation_import_modules(constraint))
                if isinstance(constraint, type):
                    loc = _get_wrapper_location(constraint)
                    if loc is not None:
                        wrapper_target_module, wrapper_name = loc
                        if wrapper_target_module == target_module:
                            constraint_parts.append(wrapper_name)
                        else:
                            constraint_parts.append(f"{wrapper_target_module}.{wrapper_name}")
                    else:
                        canonical_module, canonical_qualname = _canonical_type_ref(constraint)
                        constraint_parts.append(
                            constraint.__name__
                            if canonical_module == "builtins"
                            else f"{canonical_module}.{canonical_qualname}"
                        )
                else:
                    constraint_parts.append(repr(constraint))

        bound_value: str | None = None
        if hasattr(tv, "__bound__") and tv.__bound__ is not None:
            import_modules.update(annotation_import_modules(tv.__bound__))
            bound_value = translate_typevar_bound(
                tv.__bound__, known_impl_types, target_module, impl_modules=impl_modules
            )

        bound_annotation_ir: AnnotationIR | None = None
        impl_ref = resolve_typevar_bound_to_wrapped_impl(
            tv, known_impl_types, impl_modules, synchronicity1_synchronizer
        )
        if impl_ref is not None:
            impl_type = _get_type_from_ref(impl_ref, known_impl_types)
            if impl_type is not None:
                loc = _get_wrapper_location(impl_type)
                if loc is not None:
                    bound_annotation_ir = WrappedClassRefIR(impl_ref, ObjectReferenceIR(*loc))
                else:
                    synchronicity1_wrapper_ref = _synchronicity1_wrapper_ref(impl_type, synchronicity1_synchronizer)
                    if synchronicity1_wrapper_ref is not None:
                        bound_annotation_ir = Synchronicity1WrappedClassRefIR(impl_ref, synchronicity1_wrapper_ref)

        specs.append(
            TypeParameterIR(
                name=name,
                is_paramspec=False,
                constraint_parts=tuple(constraint_parts),
                bound_value=bound_value,
                covariant=bool(getattr(tv, "__covariant__", False)),
                contravariant=bool(getattr(tv, "__contravariant__", False)),
                bound_annotation_ir=bound_annotation_ir,
                import_modules=tuple(sorted(import_modules)),
            )
        )
    return tuple(specs)


def _get_type_from_ref(
    impl_ref,
    known_impl_types: frozenset[type],
) -> type | None:
    for t in known_impl_types:
        if t.__module__ == impl_ref.module and t.__qualname__ == impl_ref.qualname:
            return t
    return None
