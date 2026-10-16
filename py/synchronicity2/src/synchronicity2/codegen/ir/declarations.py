# Copyright Modal Labs 2026
"""IR declarations for generated modules, wrapped classes, and callables.

Signature types use data-only :class:`AnnotationIR` nodes and qualified references
instead of live implementation objects. Emission consumes this IR without reparsing
the original declarations.
"""

from __future__ import annotations

import dataclasses
import enum

from .annotations import AnnotationIR
from .references import ObjectReferenceIR


class MethodBindingKind(str, enum.Enum):
    """How an implementation method is bound on the class (mirrors ``classmethod`` / ``staticmethod``)."""

    INSTANCE = "instance"
    CLASSMETHOD = "classmethod"
    STATICMETHOD = "staticmethod"


@dataclasses.dataclass(frozen=True)
class TypeParameterIR:
    """Enough information to emit ``TypeVar`` / ``ParamSpec`` definitions.

    ``bound_annotation_ir`` is set when the bound is a synchronized implementation class; it is the
    single source for how values typed as this type parameter translate at wrapper/impl boundaries.
    """

    name: str
    is_paramspec: bool
    constraint_parts: tuple[str, ...]
    bound_value: str | None
    covariant: bool
    contravariant: bool
    bound_annotation_ir: AnnotationIR | None = None
    import_modules: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ParameterIR:
    """One formal parameter: kind + optional type as :class:`AnnotationIR` (emit unwraps from this)."""

    name: str
    kind: int
    annotation_ir: AnnotationIR | None
    default_expr: str | None
    default_import_modules: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class SignatureIR:
    """Callable signature shape shared by overloads and concrete function/method implementations."""

    parameters: tuple[ParameterIR, ...]
    return_annotation_ir: AnnotationIR


@dataclasses.dataclass(frozen=True)
class ModuleIR:
    """Planned contents of one generated wrapper module (before any text emission)."""

    target_module: str
    synchronizer_module: str
    impl_modules: frozenset[str]
    cross_module_imports: dict[str, frozenset[str]]
    typevar_specs: tuple[TypeParameterIR, ...]
    wrapped_classes: tuple[WrappedClassIR, ...]
    wrapped_functions: tuple[WrappedFunctionIR, ...]
    manual_reexports: tuple[ManualReexportIR, ...] = ()

    @property
    def has_wrapped_classes(self) -> bool:
        return bool(self.wrapped_classes)

    @property
    def class_refs(self) -> tuple[ObjectReferenceIR, ...]:
        return tuple(c.impl_ref for c in self.wrapped_classes)

    @property
    def function_refs(self) -> tuple[ObjectReferenceIR, ...]:
        return tuple(f.impl_ref for f in self.wrapped_functions)


@dataclasses.dataclass(frozen=True)
class WrappedFunctionIR:
    """Parsed module-level function and its wrapper-generation intent."""

    impl_ref: ObjectReferenceIR
    needs_async_wrapper: bool
    is_async_gen: bool
    parameters: tuple[ParameterIR, ...]
    return_annotation_ir: AnnotationIR
    overloads: tuple[SignatureIR, ...] = ()
    docstring: str | None = None
    export_name: str | None = None


@dataclasses.dataclass(frozen=True)
class ManualReexportIR:
    """A module-level name that should be re-exported directly from the impl module."""

    impl_ref: ObjectReferenceIR
    export_name: str


@dataclasses.dataclass(frozen=True)
class WrappedMethodIR:
    """Parsed method and its wrapper-generation intent; owner context is separate."""

    method_name: str
    method_type: MethodBindingKind
    parameters: tuple[ParameterIR, ...]
    is_async_gen: bool
    is_async: bool
    return_annotation_ir: AnnotationIR
    overloads: tuple[SignatureIR, ...] = ()
    docstring: str | None = None


@dataclasses.dataclass(frozen=True)
class WrappedPropertyIR:
    """Parsed @property: name, getter return type, and optional setter value type."""

    name: str
    return_annotation_ir: AnnotationIR | None
    has_setter: bool
    setter_annotation_ir: AnnotationIR | None


@dataclasses.dataclass(frozen=True)
class WrappedClassPropertyIR:
    """Parsed @classproperty: name and getter return type."""

    name: str
    return_annotation_ir: AnnotationIR | None


class ManualClassAttributeAccessKind(str, enum.Enum):
    """How to reference a manual class attribute in emitted wrapper code."""

    ATTRIBUTE = "attribute"
    RAW_CLASS_DICT = "raw_class_dict"


@dataclasses.dataclass(frozen=True)
class ManualClassAttributeIR:
    """A class attribute that should be copied into the generated wrapper class unchanged."""

    name: str
    access_kind: ManualClassAttributeAccessKind


@dataclasses.dataclass(frozen=True)
class WrappedClassIR:
    """Parsed class: everything needed to emit the wrapper without live ``type`` objects.

    The implementation identity is ``impl_ref`` (``__module__`` + ``__qualname__``). The generated
    wrapper module name is not stored here; emission takes it from :class:`ModuleIR` or the
    compile API.

    ``attributes`` holds public instance attribute names with :class:`AnnotationIR`
    for each annotation (no pre-rendered wrapper type strings).

    ``methods`` includes public methods, ``__init__`` (when not ``object.__init__``), and async
    iterator protocol dunders (``__aiter__``, ``__anext__``). The emitter partitions by
    ``method_name`` (e.g. iterator dunders map to ``__iter__``/``__aiter__``, ``__next__``/``__anext__``).

    ``wrapped_bases`` lists implementation bases that participate in synchronicity
    inheritance, each carrying both the impl ref and the resolved wrapper location.
    ``generic_type_parameters`` holds ``TypeVar`` / ``ParamSpec`` **names** for the
    ``typing.Generic[...]`` base; the emitter formats that base string.
    """

    impl_ref: ObjectReferenceIR
    wrapper_ref: ObjectReferenceIR
    wrapped_bases: tuple[tuple[ObjectReferenceIR, ObjectReferenceIR], ...]
    generic_type_parameters: tuple[str, ...] | None
    attributes: tuple[tuple[str, AnnotationIR | None], ...]
    properties: tuple[WrappedPropertyIR, ...]
    methods: tuple[WrappedMethodIR, ...]
    class_properties: tuple[WrappedClassPropertyIR, ...] = ()
    manual_attributes: tuple[ManualClassAttributeIR, ...] = ()
