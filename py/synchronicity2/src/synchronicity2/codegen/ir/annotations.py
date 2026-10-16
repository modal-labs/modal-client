# Copyright Modal Labs 2026
"""Data-only IR for parsed Python annotations.

Wrapped implementation classes and ``Self`` owners are referenced by
``(module, qualname)``. Everything else composes structurally. Emission turns
these nodes into source-generation behavior with ``codegen_for_annotation``.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Iterator

from .references import ObjectReferenceIR


class AnnotationIR(abc.ABC):
    """Base class for parsed annotation IR nodes."""

    @abc.abstractmethod
    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        """Return annotation IR nodes directly referenced by this node."""

    def direct_import_modules(self) -> tuple[str, ...]:
        """Return modules referenced directly by this node, excluding descendants."""

        return ()

    def required_import_modules(self) -> frozenset[str]:
        """Return modules required to emit this annotation tree."""

        modules = set(self.direct_import_modules())
        for referenced_ir in self.referenced_irs():
            modules.update(referenced_ir.required_import_modules())
        return frozenset(modules)


@dataclasses.dataclass(frozen=True)
class PlainAnnotationIR(AnnotationIR):
    """Non-wrapped annotation; ``signature_text`` is the type as it should appear in source."""

    signature_text: str
    import_modules: tuple[str, ...] = ()

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return ()

    def direct_import_modules(self) -> tuple[str, ...]:
        return self.import_modules


@dataclasses.dataclass(frozen=True)
class WrappedClassRefIR(AnnotationIR):
    """Registered wrapped class with resolved wrapper location."""

    impl: ObjectReferenceIR
    wrapper: ObjectReferenceIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return ()

    def direct_import_modules(self) -> tuple[str, ...]:
        return (self.wrapper.module,)


@dataclasses.dataclass(frozen=True)
class Synchronicity1WrappedClassRefIR(AnnotationIR):
    """Class wrapped by the optional Synchronicity 1 compatibility synchronizer."""

    impl: ObjectReferenceIR
    wrapper: ObjectReferenceIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return ()

    def direct_import_modules(self) -> tuple[str, ...]:
        return (self.impl.module, self.wrapper.module)


@dataclasses.dataclass(frozen=True)
class TypeVarRefIR(AnnotationIR):
    """Reference to a module-level ``typing.TypeVar`` by name.

    Bound translation behavior comes from :class:`~ir.TypeParameterIR`.
    """

    name: str

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return ()


@dataclasses.dataclass(frozen=True)
class SelfAnnotationIR(AnnotationIR):
    """``typing.Self`` tied to a wrapped owner class."""

    owner_impl: ObjectReferenceIR
    wrapper: ObjectReferenceIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return ()

    def required_import_modules(self) -> frozenset[str]:
        # Emitted as ``typing_extensions.Self`` so generated code works on Python < 3.11.
        return super().required_import_modules() | {"typing_extensions"}


@dataclasses.dataclass(frozen=True)
class ListAnnotationIR(AnnotationIR):
    item_ir: AnnotationIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return (self.item_ir,)


@dataclasses.dataclass(frozen=True)
class DictAnnotationIR(AnnotationIR):
    key_ir: AnnotationIR
    value_ir: AnnotationIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return (self.key_ir, self.value_ir)


@dataclasses.dataclass(frozen=True)
class SequenceAnnotationIR(AnnotationIR):
    item_ir: AnnotationIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return (self.item_ir,)


@dataclasses.dataclass(frozen=True)
class CollectionAnnotationIR(AnnotationIR):
    item_ir: AnnotationIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return (self.item_ir,)


@dataclasses.dataclass(frozen=True)
class TupleAnnotationIR(AnnotationIR):
    """Fixed ``tuple[T1, T2]`` or variadic ``tuple[T, ...]`` (``variadic=True``, single element)."""

    element_irs: tuple[AnnotationIR, ...]
    variadic: bool

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return self.element_irs


@dataclasses.dataclass(frozen=True)
class OptionalAnnotationIR(AnnotationIR):
    inner_ir: AnnotationIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return (self.inner_ir,)


@dataclasses.dataclass(frozen=True)
class UnionAnnotationIR(AnnotationIR):
    arm_irs: tuple[AnnotationIR, ...]
    source_label: str | None = None

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return self.arm_irs


@dataclasses.dataclass(frozen=True)
class AsyncGeneratorAnnotationIR(AnnotationIR):
    yield_annotation_ir: AnnotationIR
    send_annotation_ir: AnnotationIR | None

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        if self.send_annotation_ir is None:
            return (self.yield_annotation_ir,)
        return (self.yield_annotation_ir, self.send_annotation_ir)


@dataclasses.dataclass(frozen=True)
class SyncGeneratorAnnotationIR(AnnotationIR):
    yield_annotation_ir: AnnotationIR
    send_annotation_ir: AnnotationIR
    return_annotation_ir: AnnotationIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return (self.yield_annotation_ir, self.send_annotation_ir, self.return_annotation_ir)


@dataclasses.dataclass(frozen=True)
class SyncIteratorAnnotationIR(AnnotationIR):
    item_ir: AnnotationIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return (self.item_ir,)


@dataclasses.dataclass(frozen=True)
class AsyncIteratorAnnotationIR(AnnotationIR):
    item_ir: AnnotationIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return (self.item_ir,)


@dataclasses.dataclass(frozen=True)
class AsyncIterableAnnotationIR(AnnotationIR):
    item_ir: AnnotationIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return (self.item_ir,)


@dataclasses.dataclass(frozen=True)
class CoroutineAnnotationIR(AnnotationIR):
    return_annotation_ir: AnnotationIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return (self.return_annotation_ir,)


@dataclasses.dataclass(frozen=True)
class AwaitableAnnotationIR(AnnotationIR):
    inner_ir: AnnotationIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return (self.inner_ir,)


@dataclasses.dataclass(frozen=True)
class AsyncContextManagerAnnotationIR(AnnotationIR):
    value_ir: AnnotationIR

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return (self.value_ir,)


@dataclasses.dataclass(frozen=True)
class CallableAnnotationIR(AnnotationIR):
    parameter_irs: tuple[AnnotationIR, ...] | None
    return_annotation_ir: AnnotationIR
    # Non-expanded parameter syntax (notably ParamSpec) is emitted verbatim. Like declaration
    # defaults retained as source strings, its imports must travel beside the text because there
    # is no child IR from which to recover them later.
    params_signature_text: str | None = None
    params_signature_import_modules: tuple[str, ...] = ()

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        parameter_irs = self.parameter_irs or ()
        return (*parameter_irs, self.return_annotation_ir)

    def direct_import_modules(self) -> tuple[str, ...]:
        return self.params_signature_import_modules


@dataclasses.dataclass(frozen=True)
class ParameterizedWrappedClassRefIR(AnnotationIR):
    """Wrapped class subscripted with type arguments, e.g. ``SomeContainer[WrappedType]``."""

    wrapped_class_ir: WrappedClassRefIR | Synchronicity1WrappedClassRefIR
    type_argument_irs: tuple[AnnotationIR, ...]

    def referenced_irs(self) -> tuple[AnnotationIR, ...]:
        return (self.wrapped_class_ir, *self.type_argument_irs)


def walk_annotation_irs(annotation_ir: AnnotationIR) -> Iterator[AnnotationIR]:
    """Yield an annotation IR tree in depth-first, pre-order traversal."""

    stack = [annotation_ir]
    while stack:
        current_ir = stack.pop()
        yield current_ir
        stack.extend(reversed(current_ir.referenced_irs()))
