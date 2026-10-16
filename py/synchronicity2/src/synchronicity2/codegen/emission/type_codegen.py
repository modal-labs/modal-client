# Copyright Modal Labs 2026
"""Generate annotation and call-boundary source expressions from annotation IR.

Each TypeCodegen encapsulates:
1. Type signature generation (public_annotation)
2. Unwrap expressions (wrapper → impl)
3. Wrap expressions (impl → wrapper)

Codegens compose through nesting for complex types like list[Person].
"""

from __future__ import annotations

import dataclasses
import re
import uuid
from abc import ABC, abstractmethod

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
from ..ir.declarations import MethodBindingKind, TypeParameterIR
from ..ir.references import ObjectReferenceIR


@dataclasses.dataclass
class TypeCodegenContext:
    """Type-parameter declarations needed while generating annotation code."""

    type_parameters_by_name: dict[str, TypeParameterIR] | None = None


def _wrapper_ref_dotted(wrapper: ObjectReferenceIR) -> str:
    return f"{wrapper.module}.{wrapper.qualname}"


def _wrapper_ref_runtime_expr(wrapper: ObjectReferenceIR, target_module: str) -> str:
    if wrapper.module == target_module:
        return wrapper.qualname
    return _wrapper_ref_dotted(wrapper)


class TypeCodegen(ABC):
    """Base class for type codegens that handle type signatures and translation."""

    @abstractmethod
    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        """Return the type signature string for generated wrapper code.

        Args:
            target_module: Current target module (for local vs cross-module refs)
            is_async: Whether we're in an async context (affects async generator return types)

        Returns:
            Type string like "Person", "list[str]", "foo.bar.Person"
        """
        pass

    @abstractmethod
    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        """Generate Python expression to unwrap from wrapper → impl.

        Args:
            var_name: Variable name to unwrap

        Returns:
            Expression string that unwraps the value through the generated synchronizer binding.
        """
        pass

    @abstractmethod
    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        """Generate Python expression to wrap from impl → wrapper.

        Args:
            target_module: Current target module (for local vs cross-module refs)
            var_name: Variable name to wrap
            is_async: Whether we're in an async context (affects generator wrapping)

        Returns:
            Expression string like "Person._from_impl(value)" or "[Person._from_impl(x) for x in value]"
        """
        pass

    def requires_boundary_translation(self) -> bool:
        """Check if this type needs unwrap/wrap translation.

        Returns:
            True if this type contains wrapped classes that need translation
        """
        return False

    def helper_definitions(
        self,
        target_module: str,
        indent: str = "    ",
    ) -> dict[str, str]:
        """Generate inline helper functions needed for wrapping this type.

        Returns a dict to enable automatic deduplication.

        Args:
            target_module: Current target module
            indent: Indentation string for the helper functions

        Returns:
            Dict mapping helper function name to helper function code.
            Empty dict for types that don't need helpers.
        """
        return {}

    def references_wrapper_class(self) -> bool:
        """Whether this annotation contains a generated wrapper class reference."""
        return False

    def annotation_type(self, target_module: str, is_async: bool = True) -> str:
        """Return the string to emit in generated annotations.

        This defaults to ``public_annotation`` and quotes the whole expression whenever it contains
        a wrapper reference that may otherwise be unavailable during module import.
        """
        wrapped = self.public_annotation(target_module, is_async)
        if self.references_wrapper_class():
            return f'"{wrapped}"'
        return wrapped

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        """Return the impl-facing annotation string for passthrough values.

        This is used for container annotations like ``Callable`` where Synchronicity does not
        currently wrap or unwrap the inner runtime value, so the public wrapper annotation needs
        to describe the implementation-facing contract rather than the translated wrapper-facing one.
        """
        return self.annotation_type(target_module, is_async)


class PlainTypeCodegen(TypeCodegen):
    """Code generator for a preformatted annotation that needs no boundary translation."""

    def __init__(self, signature_text: str):
        self._signature_text = signature_text

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        return self._signature_text

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        return var_name

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        return var_name

    def requires_boundary_translation(self) -> bool:
        return False


class WrappedClassTypeCodegen(TypeCodegen):
    """Codegen for wrapped class types."""

    def __init__(self, impl: ObjectReferenceIR, wrapper: ObjectReferenceIR):
        self.impl_ref = impl
        self._wrapper = wrapper

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        """Return the wrapper class name (local or fully qualified)."""
        if self._wrapper.module == target_module:
            return self._wrapper.qualname
        else:
            return f"{self._wrapper.module}.{self._wrapper.qualname}"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        """Unwrap a native generated wrapper."""
        return f"{var_name}._impl_instance"

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        """Wrap by calling WrapperClass._from_impl()."""
        if self._wrapper.module == target_module:
            return f"{self._wrapper.qualname}._from_impl({var_name})"
        else:
            return f"{self._wrapper.module}.{self._wrapper.qualname}._from_impl({var_name})"

    def requires_boundary_translation(self) -> bool:
        return True

    def references_wrapper_class(self) -> bool:
        return True

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        return _impl_ref_dotted(self.impl_ref)


class Synchronicity1WrappedClassTypeCodegen(TypeCodegen):
    """Codegen for implementation classes wrapped by Synchronicity 1."""

    def __init__(self, impl: ObjectReferenceIR, wrapper: ObjectReferenceIR):
        self.impl_ref = impl
        self._wrapper = wrapper

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        return _wrapper_ref_runtime_expr(self._wrapper, target_module)

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        impl_type = _impl_ref_dotted(self.impl_ref)
        return f"typing.cast({impl_type}, _synchronizer._synchronicity1._translate_in({var_name}))"

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        wrapper_type = _wrapper_ref_runtime_expr(self._wrapper, target_module)
        return f"typing.cast({wrapper_type}, _synchronizer._synchronicity1._translate_out({var_name}))"

    def requires_boundary_translation(self) -> bool:
        return True

    def references_wrapper_class(self) -> bool:
        return True

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        return _impl_ref_dotted(self.impl_ref)


class ParameterizedWrappedClassTypeCodegen(TypeCodegen):
    """Wrapped class subscripted with type args, e.g. ``SomeContainer[WrappedType]``.

    Unwrap/wrap delegates to the base :class:`WrappedClassTypeCodegen`; the type
    arguments only affect the annotation string.
    """

    def __init__(
        self,
        inner: WrappedClassTypeCodegen | Synchronicity1WrappedClassTypeCodegen,
        type_arg_codegens: list[TypeCodegen],
    ):
        self._inner = inner
        self._type_arg_codegens = type_arg_codegens

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        base = self._inner.public_annotation(target_module, is_async)
        args = ", ".join(t.public_annotation(target_module, is_async) for t in self._type_arg_codegens)
        return f"{base}[{args}]"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        return self._inner.wrapper_to_impl_expr(var_name, target_module)

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        return self._inner.impl_to_wrapper_expr(target_module, var_name, is_async)

    def requires_boundary_translation(self) -> bool:
        return True

    def references_wrapper_class(self) -> bool:
        return self._inner.references_wrapper_class() or any(
            t.references_wrapper_class() for t in self._type_arg_codegens
        )

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        base = _impl_ref_dotted(self._inner.impl_ref)
        args = ", ".join(t.implementation_annotation(target_module, is_async) for t in self._type_arg_codegens)
        return f"{base}[{args}]"


class BoundTypeVarCodegen(TypeCodegen):
    """Type variable: signature shows *name* (e.g. ``T``); unwrap/wrap follows the bound type codegen."""

    def __init__(self, name: str, bound_codegen: TypeCodegen):
        self._name = name
        self._bound = bound_codegen

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        return self._name

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        return self._bound.wrapper_to_impl_expr(var_name, target_module)

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        inner = self._bound.impl_to_wrapper_expr(target_module, var_name, is_async)
        return f"typing.cast({self._name}, {inner})"

    def requires_boundary_translation(self) -> bool:
        return True

    def helper_definitions(
        self,
        target_module: str,
        indent: str = "    ",
    ) -> dict[str, str]:
        return self._bound.helper_definitions(target_module, indent)

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        return self._name


class SelfTypeCodegen(TypeCodegen):
    """``typing.Self`` on a synchronized class: unwrap/wrap like the wrapper type, but keep ``typing.Self``
    in signatures.

    Emitting the concrete wrapper name would break ``Self`` binding on subclasses and on generic classes.
    """

    def __init__(self, impl: ObjectReferenceIR, wrapper: ObjectReferenceIR):
        self._impl = WrappedClassTypeCodegen(impl, wrapper)

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        return "typing_extensions.Self"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        return self._impl.wrapper_to_impl_expr(var_name, target_module)

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        return self._impl.impl_to_wrapper_expr(target_module, var_name, is_async)

    def impl_to_wrapper_expr_for_method(
        self,
        target_module: str,
        var_name: str,
        *,
        is_async: bool,
        method_type: MethodBindingKind,
    ) -> str:
        """Runtime wrap for ``typing.Self`` (``_from_impl`` is a classmethod; call via ``self`` / ``cls``)."""
        if method_type == MethodBindingKind.CLASSMETHOD:
            binding = "cls"
        elif method_type == MethodBindingKind.STATICMETHOD:
            return self._impl.impl_to_wrapper_expr(target_module, var_name, is_async)
        else:
            binding = "self"
        return f"typing.cast(typing_extensions.Self, {binding}._from_impl({var_name}))"

    def requires_boundary_translation(self) -> bool:
        return True

    def helper_definitions(
        self,
        target_module: str,
        indent: str = "    ",
    ) -> dict[str, str]:
        return self._impl.helper_definitions(target_module, indent)

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        return _impl_ref_dotted(self._impl.impl_ref)


class ListTypeCodegen(TypeCodegen):
    """Codegen for list[T] types."""

    def __init__(self, item_codegen: TypeCodegen):
        self.item_codegen = item_codegen

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        """Return list[WrappedItemType]."""
        item_type_str = self.item_codegen.public_annotation(target_module, is_async)
        return f"list[{item_type_str}]"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        """Generate list comprehension to unwrap items."""
        if not self.item_codegen.requires_boundary_translation():
            return var_name

        item_unwrap = self.item_codegen.wrapper_to_impl_expr("x", target_module)
        return f"[{item_unwrap} for x in {var_name}]"

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        """Generate list comprehension to wrap items."""
        if not self.item_codegen.requires_boundary_translation():
            return var_name

        item_wrap = self.item_codegen.impl_to_wrapper_expr(target_module, "x", is_async)
        expr = f"[{item_wrap} for x in {var_name}]"
        if isinstance(self.item_codegen, BoundTypeVarCodegen):
            ann = self.public_annotation(target_module, is_async)
            return f"typing.cast({ann}, {expr})"
        return expr

    def requires_boundary_translation(self) -> bool:
        return self.item_codegen.requires_boundary_translation()

    def helper_definitions(
        self,
        target_module: str,
        indent: str = "    ",
    ) -> dict[str, str]:
        """Recursively collect helpers from item codegen."""
        return self.item_codegen.helper_definitions(target_module, indent)

    def references_wrapper_class(self) -> bool:
        return self.item_codegen.references_wrapper_class()

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        item_type_str = self.item_codegen.implementation_annotation(target_module, is_async)
        return f"list[{item_type_str}]"


class DictTypeCodegen(TypeCodegen):
    """Codegen for dict[K, V] types."""

    def __init__(self, key_codegen: TypeCodegen, value_codegen: TypeCodegen):
        self.key_codegen = key_codegen
        self.value_codegen = value_codegen

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        """Return dict[WrappedKeyType, WrappedValueType]."""
        key_type_str = self.key_codegen.public_annotation(target_module, is_async)
        value_type_str = self.value_codegen.public_annotation(target_module, is_async)
        return f"dict[{key_type_str}, {value_type_str}]"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        """Generate dict comprehension to unwrap values."""
        if not self.value_codegen.requires_boundary_translation():
            return var_name

        value_unwrap = self.value_codegen.wrapper_to_impl_expr("v", target_module)
        return f"{{k: {value_unwrap} for k, v in {var_name}.items()}}"

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        """Generate dict comprehension to wrap values."""
        if not self.value_codegen.requires_boundary_translation():
            return var_name

        value_wrap = self.value_codegen.impl_to_wrapper_expr(target_module, "v", is_async)
        return f"{{k: {value_wrap} for k, v in {var_name}.items()}}"

    def requires_boundary_translation(self) -> bool:
        return self.value_codegen.requires_boundary_translation()

    def helper_definitions(
        self,
        target_module: str,
        indent: str = "    ",
    ) -> dict[str, str]:
        """Recursively collect helpers from key and value codegens."""
        helpers = {}
        helpers.update(self.key_codegen.helper_definitions(target_module, indent))
        helpers.update(self.value_codegen.helper_definitions(target_module, indent))
        return helpers

    def references_wrapper_class(self) -> bool:
        return self.key_codegen.references_wrapper_class() or self.value_codegen.references_wrapper_class()

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        key_type_str = self.key_codegen.implementation_annotation(target_module, is_async)
        value_type_str = self.value_codegen.implementation_annotation(target_module, is_async)
        return f"dict[{key_type_str}, {value_type_str}]"


class SequenceTypeCodegen(TypeCodegen):
    """Codegen for typing.Sequence[T]."""

    def __init__(self, item_codegen: TypeCodegen):
        self.item_codegen = item_codegen

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        item_type_str = self.item_codegen.public_annotation(target_module, is_async)
        return f"typing.Sequence[{item_type_str}]"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        if not self.item_codegen.requires_boundary_translation():
            return var_name
        item_unwrap = self.item_codegen.wrapper_to_impl_expr("x", target_module)
        return f"[{item_unwrap} for x in {var_name}]"

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        if not self.item_codegen.requires_boundary_translation():
            return var_name
        item_wrap = self.item_codegen.impl_to_wrapper_expr(target_module, "x", is_async)
        expr = f"[{item_wrap} for x in {var_name}]"
        if isinstance(self.item_codegen, BoundTypeVarCodegen):
            ann = self.public_annotation(target_module, is_async)
            return f"typing.cast({ann}, {expr})"
        return expr

    def requires_boundary_translation(self) -> bool:
        return self.item_codegen.requires_boundary_translation()

    def helper_definitions(
        self,
        target_module: str,
        indent: str = "    ",
    ) -> dict[str, str]:
        return self.item_codegen.helper_definitions(target_module, indent)

    def references_wrapper_class(self) -> bool:
        return self.item_codegen.references_wrapper_class()

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        item_type_str = self.item_codegen.implementation_annotation(target_module, is_async)
        return f"typing.Sequence[{item_type_str}]"


class CollectionTypeCodegen(TypeCodegen):
    """Codegen for typing.Collection[T]."""

    def __init__(self, item_codegen: TypeCodegen):
        self.item_codegen = item_codegen

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        item_type_str = self.item_codegen.public_annotation(target_module, is_async)
        return f"typing.Collection[{item_type_str}]"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        if not self.item_codegen.requires_boundary_translation():
            return var_name
        item_unwrap = self.item_codegen.wrapper_to_impl_expr("x", target_module)
        return f"[{item_unwrap} for x in {var_name}]"

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        if not self.item_codegen.requires_boundary_translation():
            return var_name
        item_wrap = self.item_codegen.impl_to_wrapper_expr(target_module, "x", is_async)
        expr = f"[{item_wrap} for x in {var_name}]"
        if isinstance(self.item_codegen, BoundTypeVarCodegen):
            ann = self.public_annotation(target_module, is_async)
            return f"typing.cast({ann}, {expr})"
        return expr

    def requires_boundary_translation(self) -> bool:
        return self.item_codegen.requires_boundary_translation()

    def helper_definitions(
        self,
        target_module: str,
        indent: str = "    ",
    ) -> dict[str, str]:
        return self.item_codegen.helper_definitions(target_module, indent)

    def references_wrapper_class(self) -> bool:
        return self.item_codegen.references_wrapper_class()

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        item_type_str = self.item_codegen.implementation_annotation(target_module, is_async)
        return f"typing.Collection[{item_type_str}]"


class TupleTypeCodegen(TypeCodegen):
    """Codegen for tuple types - both fixed-size tuple[T1, T2] and variable-length tuple[T, ...]."""

    def __init__(self, item_codegens: list[TypeCodegen]):
        """
        Args:
            item_codegens: List of codegens for each tuple element.
                               If all elements are the same type, this can be a single-item list.
        """
        self.item_codegens = item_codegens

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        """Return tuple[WrappedType1, WrappedType2, ...] or tuple[WrappedType, ...]."""
        if len(self.item_codegens) == 1:
            item_type_str = self.item_codegens[0].public_annotation(target_module, is_async)
            return f"tuple[{item_type_str}, ...]"
        else:
            item_type_strs = [t.public_annotation(target_module, is_async) for t in self.item_codegens]
            return f"tuple[{', '.join(item_type_strs)}]"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        """Generate tuple comprehension/constructor to unwrap items."""
        if not self.requires_boundary_translation():
            return var_name

        if len(self.item_codegens) == 1:
            item_unwrap = self.item_codegens[0].wrapper_to_impl_expr("x", target_module)
            return f"tuple({item_unwrap} for x in {var_name})"
        else:
            wrapper_to_impl_exprs = [
                t.wrapper_to_impl_expr(f"{var_name}[{i}]", target_module) for i, t in enumerate(self.item_codegens)
            ]
            return f"({', '.join(wrapper_to_impl_exprs)})"

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        """Generate tuple comprehension/constructor to wrap items."""
        if not self.requires_boundary_translation():
            return var_name

        if len(self.item_codegens) == 1:
            item_wrap = self.item_codegens[0].impl_to_wrapper_expr(target_module, "x", is_async)
            expr = f"tuple({item_wrap} for x in {var_name})"
        else:
            impl_to_wrapper_exprs = [
                t.impl_to_wrapper_expr(target_module, f"{var_name}[{i}]", is_async)
                for i, t in enumerate(self.item_codegens)
            ]
            expr = f"({', '.join(impl_to_wrapper_exprs)})"
        if any(isinstance(t, BoundTypeVarCodegen) for t in self.item_codegens):
            ann = self.public_annotation(target_module, is_async)
            return f"typing.cast({ann}, {expr})"
        return expr

    def requires_boundary_translation(self) -> bool:
        return any(t.requires_boundary_translation() for t in self.item_codegens)

    def helper_definitions(
        self,
        target_module: str,
        indent: str = "    ",
    ) -> dict[str, str]:
        """Recursively collect helpers from all item codegens."""
        helpers = {}
        for codegen in self.item_codegens:
            helpers.update(codegen.helper_definitions(target_module, indent))
        return helpers

    def references_wrapper_class(self) -> bool:
        return any(codegen.references_wrapper_class() for codegen in self.item_codegens)

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        if len(self.item_codegens) == 1:
            item_type_str = self.item_codegens[0].implementation_annotation(target_module, is_async)
            return f"tuple[{item_type_str}, ...]"
        item_type_strs = [t.implementation_annotation(target_module, is_async) for t in self.item_codegens]
        return f"tuple[{', '.join(item_type_strs)}]"


class OptionalTypeCodegen(TypeCodegen):
    """Codegen for Optional[T] (Union[T, None]) types."""

    def __init__(self, inner_codegen: TypeCodegen):
        self.inner_codegen = inner_codegen

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        """Return typing.Union[WrappedInnerType, None]."""
        inner_type_str = self.inner_codegen.public_annotation(target_module, is_async)
        return f"typing.Union[{inner_type_str}, None]"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        """Generate conditional expression to unwrap if not None."""
        if not self.inner_codegen.requires_boundary_translation():
            return var_name

        inner_unwrap = self.inner_codegen.wrapper_to_impl_expr(var_name, target_module)
        return f"{inner_unwrap} if {var_name} is not None else None"

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        """Generate conditional expression to wrap if not None."""
        if not self.inner_codegen.requires_boundary_translation():
            return var_name

        inner_wrap = self.inner_codegen.impl_to_wrapper_expr(target_module, var_name, is_async)
        return f"{inner_wrap} if {var_name} is not None else None"

    def requires_boundary_translation(self) -> bool:
        return self.inner_codegen.requires_boundary_translation()

    def helper_definitions(
        self,
        target_module: str,
        indent: str = "    ",
    ) -> dict[str, str]:
        """Recursively collect helpers from inner codegen."""
        return self.inner_codegen.helper_definitions(target_module, indent)

    def references_wrapper_class(self) -> bool:
        return self.inner_codegen.references_wrapper_class()

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        inner_type_str = self.inner_codegen.implementation_annotation(target_module, is_async)
        return f"typing.Union[{inner_type_str}, None]"


@dataclasses.dataclass(frozen=True)
class _UnionArmRuntimeSpec:
    discriminator_key: tuple[str, ...] | None
    runtime_action_key: tuple[str, ...] | None
    unwrap_guard_expr: str | None
    wrap_guard_expr: str | None
    translated: bool
    unwrap_value_expr: str
    wrap_value_expr: str


def _impl_ref_dotted(impl_ref: ObjectReferenceIR) -> str:
    q = impl_ref.qualname
    if ".<locals>." in q or q.startswith("<locals>."):
        return f"{impl_ref.module}.{q.rpartition('.')[2]}"
    return f"{impl_ref.module}.{q}"


def _identity_runtime_expr(signature_text: str) -> str | None:
    if signature_text in {"None", "int", "str", "bool", "float", "bytes", "complex"}:
        return signature_text
    if signature_text.startswith("typing."):
        return None
    if "[" in signature_text or "]" in signature_text or " " in signature_text or "|" in signature_text:
        return None
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*", signature_text):
        return None
    return signature_text


def _runtime_action_key(codegen: TypeCodegen) -> tuple[str, ...] | None:
    if isinstance(codegen, PlainTypeCodegen):
        return ("identity", codegen._signature_text)

    if isinstance(codegen, WrappedClassTypeCodegen):
        return ("wrapped_impl", codegen.impl_ref.module, codegen.impl_ref.qualname)

    if isinstance(codegen, Synchronicity1WrappedClassTypeCodegen):
        return ("synchronicity1_wrapped_impl", codegen.impl_ref.module, codegen.impl_ref.qualname)

    if isinstance(codegen, ParameterizedWrappedClassTypeCodegen):
        return _runtime_action_key(codegen._inner)

    if isinstance(codegen, BoundTypeVarCodegen):
        return _runtime_action_key(codegen._bound)

    if isinstance(codegen, SelfTypeCodegen):
        return ("wrapped_impl", codegen._impl.impl_ref.module, codegen._impl.impl_ref.qualname)

    if isinstance(codegen, ListTypeCodegen):
        if not codegen.requires_boundary_translation():
            return ("identity", "list")
        item_key = _runtime_action_key(codegen.item_codegen)
        if item_key is None:
            return None
        return ("list", *item_key)

    if isinstance(codegen, DictTypeCodegen):
        if not codegen.requires_boundary_translation():
            return ("identity", "dict")
        key_key = _runtime_action_key(codegen.key_codegen)
        value_key = _runtime_action_key(codegen.value_codegen)
        if key_key is None or value_key is None:
            return None
        return ("dict", "key", *key_key, "value", *value_key)

    if isinstance(codegen, TupleTypeCodegen):
        if not codegen.requires_boundary_translation():
            return ("identity", "tuple")
        if len(codegen.item_codegens) == 1:
            item_key = _runtime_action_key(codegen.item_codegens[0])
            if item_key is None:
                return None
            return ("tuple", "variadic", *item_key)
        parts: list[str] = ["tuple", "fixed"]
        for item_codegen in codegen.item_codegens:
            item_key = _runtime_action_key(item_codegen)
            if item_key is None:
                return None
            parts.extend(["item", *item_key])
        return tuple(parts)

    if isinstance(codegen, OptionalTypeCodegen):
        inner_key = _runtime_action_key(codegen.inner_codegen)
        if inner_key is None:
            return None
        return ("optional", *inner_key)

    if isinstance(codegen, CallableTypeCodegen):
        return ("callable",)

    return None


def _union_arm_runtime_spec(
    codegen: TypeCodegen,
    target_module: str,
    *,
    is_async: bool,
) -> _UnionArmRuntimeSpec:
    if isinstance(codegen, PlainTypeCodegen):
        runtime_expr = _identity_runtime_expr(codegen.public_annotation(target_module, is_async))
        if runtime_expr == "None":
            return _UnionArmRuntimeSpec(
                discriminator_key=("none",),
                runtime_action_key=_runtime_action_key(codegen),
                unwrap_guard_expr="_v is None",
                wrap_guard_expr="_v is None",
                translated=False,
                unwrap_value_expr="_v",
                wrap_value_expr="_v",
            )
        if runtime_expr is not None:
            return _UnionArmRuntimeSpec(
                discriminator_key=("identity", runtime_expr),
                runtime_action_key=_runtime_action_key(codegen),
                unwrap_guard_expr=f"isinstance(_v, {runtime_expr})",
                wrap_guard_expr=f"isinstance(_v, {runtime_expr})",
                translated=False,
                unwrap_value_expr="_v",
                wrap_value_expr="_v",
            )
        return _UnionArmRuntimeSpec(
            discriminator_key=None,
            runtime_action_key=_runtime_action_key(codegen),
            unwrap_guard_expr=None,
            wrap_guard_expr=None,
            translated=False,
            unwrap_value_expr="_v",
            wrap_value_expr="_v",
        )

    if isinstance(codegen, WrappedClassTypeCodegen):
        wrapper_expr = _wrapper_ref_runtime_expr(codegen._wrapper, target_module)
        impl_expr = _impl_ref_dotted(codegen.impl_ref)
        return _UnionArmRuntimeSpec(
            discriminator_key=("impl", codegen.impl_ref.module, codegen.impl_ref.qualname),
            runtime_action_key=_runtime_action_key(codegen),
            unwrap_guard_expr=f"isinstance(_v, {wrapper_expr})",
            wrap_guard_expr=f"isinstance(_v, {impl_expr})",
            translated=True,
            unwrap_value_expr="_v._impl_instance",
            wrap_value_expr=codegen.impl_to_wrapper_expr(target_module, "_v", is_async),
        )

    if isinstance(codegen, ParameterizedWrappedClassTypeCodegen):
        inner_spec = _union_arm_runtime_spec(codegen._inner, target_module, is_async=is_async)
        return dataclasses.replace(
            inner_spec,
            runtime_action_key=_runtime_action_key(codegen),
            wrap_value_expr=codegen.impl_to_wrapper_expr(target_module, "_v", is_async),
        )

    if isinstance(codegen, Synchronicity1WrappedClassTypeCodegen):
        wrapper_expr = _wrapper_ref_runtime_expr(codegen._wrapper, target_module)
        impl_expr = _impl_ref_dotted(codegen.impl_ref)
        return _UnionArmRuntimeSpec(
            discriminator_key=("synchronicity1_impl", codegen.impl_ref.module, codegen.impl_ref.qualname),
            runtime_action_key=_runtime_action_key(codegen),
            unwrap_guard_expr=f"isinstance(_v, {wrapper_expr})",
            wrap_guard_expr=f"isinstance(_v, {impl_expr})",
            translated=True,
            unwrap_value_expr=codegen.wrapper_to_impl_expr("_v"),
            wrap_value_expr=codegen.impl_to_wrapper_expr(target_module, "_v", is_async),
        )

    if isinstance(codegen, BoundTypeVarCodegen):
        spec = _union_arm_runtime_spec(codegen._bound, target_module, is_async=is_async)
        return _UnionArmRuntimeSpec(
            discriminator_key=spec.discriminator_key,
            runtime_action_key=_runtime_action_key(codegen),
            unwrap_guard_expr=spec.unwrap_guard_expr,
            wrap_guard_expr=spec.wrap_guard_expr,
            translated=True,
            unwrap_value_expr=spec.unwrap_value_expr,
            wrap_value_expr=codegen.impl_to_wrapper_expr(target_module, "_v", is_async),
        )

    if isinstance(codegen, SelfTypeCodegen):
        wrapper_expr = _wrapper_ref_runtime_expr(codegen._impl._wrapper, target_module)
        impl_expr = _impl_ref_dotted(codegen._impl.impl_ref)
        return _UnionArmRuntimeSpec(
            discriminator_key=("impl", codegen._impl.impl_ref.module, codegen._impl.impl_ref.qualname),
            runtime_action_key=_runtime_action_key(codegen),
            unwrap_guard_expr=f"isinstance(_v, {wrapper_expr})",
            wrap_guard_expr=f"isinstance(_v, {impl_expr})",
            translated=True,
            unwrap_value_expr="_v._impl_instance",
            wrap_value_expr=codegen.impl_to_wrapper_expr(target_module, "_v", is_async),
        )

    if isinstance(codegen, ListTypeCodegen):
        return _UnionArmRuntimeSpec(
            discriminator_key=("list",),
            runtime_action_key=_runtime_action_key(codegen),
            unwrap_guard_expr="isinstance(_v, list)",
            wrap_guard_expr="isinstance(_v, list)",
            translated=codegen.requires_boundary_translation(),
            unwrap_value_expr=codegen.wrapper_to_impl_expr("_v"),
            wrap_value_expr=codegen.impl_to_wrapper_expr(target_module, "_v", is_async),
        )

    if isinstance(codegen, DictTypeCodegen):
        return _UnionArmRuntimeSpec(
            discriminator_key=("dict",),
            runtime_action_key=_runtime_action_key(codegen),
            unwrap_guard_expr="isinstance(_v, dict)",
            wrap_guard_expr="isinstance(_v, dict)",
            translated=codegen.requires_boundary_translation(),
            unwrap_value_expr=codegen.wrapper_to_impl_expr("_v"),
            wrap_value_expr=codegen.impl_to_wrapper_expr(target_module, "_v", is_async),
        )

    if isinstance(codegen, TupleTypeCodegen):
        return _UnionArmRuntimeSpec(
            discriminator_key=("tuple",),
            runtime_action_key=_runtime_action_key(codegen),
            unwrap_guard_expr="isinstance(_v, tuple)",
            wrap_guard_expr="isinstance(_v, tuple)",
            translated=codegen.requires_boundary_translation(),
            unwrap_value_expr=codegen.wrapper_to_impl_expr("_v"),
            wrap_value_expr=codegen.impl_to_wrapper_expr(target_module, "_v", is_async),
        )

    if isinstance(codegen, CallableTypeCodegen):
        return _UnionArmRuntimeSpec(
            discriminator_key=("callable",),
            runtime_action_key=_runtime_action_key(codegen),
            unwrap_guard_expr="callable(_v)",
            wrap_guard_expr="callable(_v)",
            translated=codegen.requires_boundary_translation(),
            unwrap_value_expr=codegen.wrapper_to_impl_expr("_v", target_module),
            wrap_value_expr=codegen.impl_to_wrapper_expr(target_module, "_v", is_async),
        )

    raise TypeError(f"Union translation does not support runtime discrimination for {type(codegen).__name__}")


class UnionTypeCodegen(TypeCodegen):
    """Codegen for top-level ``Union[...]`` values with runtime-discriminable branches."""

    def __init__(self, item_codegens: list[TypeCodegen], source_label: str | None = None):
        self.item_codegens = item_codegens
        self.source_label = source_label

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        item_types = [t.public_annotation(target_module, is_async) for t in self.item_codegens]
        return f"typing.Union[{', '.join(item_types)}]"

    def _error_prefix(self) -> str:
        if self.source_label:
            return f"{self.source_label}: "
        return ""

    def _arm_type_label(self, codegen: TypeCodegen) -> str:
        return codegen.public_annotation("", is_async=True)

    def _runtime_specs(self, target_module: str, *, is_async: bool) -> list[_UnionArmRuntimeSpec]:
        specs = [_union_arm_runtime_spec(t, target_module, is_async=is_async) for t in self.item_codegens]
        if not any(spec.translated for spec in specs):
            return specs

        unique_specs: list[_UnionArmRuntimeSpec] = []
        seen: dict[tuple[str, ...], _UnionArmRuntimeSpec] = {}
        seen_labels: dict[tuple[str, ...], str] = {}
        for spec, codegen in zip(specs, self.item_codegens, strict=False):
            label = self._arm_type_label(codegen)
            if spec.discriminator_key is None:
                if spec.translated:
                    raise TypeError(
                        self._error_prefix()
                        + "Union translation requires runtime-discriminable translated arms; "
                        + f"unsupported union member {label!r}"
                    )
                unique_specs.append(spec)
                continue
            prev = seen.get(spec.discriminator_key)
            if prev is not None:
                if prev.runtime_action_key is not None and prev.runtime_action_key == spec.runtime_action_key:
                    continue
                if prev.translated or spec.translated:
                    previous_label = seen_labels[spec.discriminator_key]
                    raise TypeError(
                        self._error_prefix()
                        + "Union translation cannot disambiguate multiple arms with the same runtime shape "
                        + f"{previous_label!r} and {label!r}"
                    )
                continue
            seen[spec.discriminator_key] = spec
            seen_labels[spec.discriminator_key] = label
            unique_specs.append(spec)
        return unique_specs

    def _branch_expr(self, target_module: str, var_name: str, *, is_async: bool, for_wrap: bool) -> str:
        specs = self._runtime_specs(target_module, is_async=is_async)
        if not any(spec.translated for spec in specs):
            return var_name

        none_specs = [spec for spec in specs if spec.discriminator_key == ("none",)]
        translated_specs = [spec for spec in specs if spec.translated and spec.discriminator_key != ("none",)]
        identity_known_specs = [
            spec for spec in specs if not spec.translated and spec.discriminator_key not in {None, ("none",)}
        ]
        has_identity_fallback = any(not spec.translated and spec.discriminator_key is None for spec in specs)

        cases: list[tuple[str, str]] = []
        if none_specs:
            none_spec = none_specs[0]
            cases.append(
                (
                    none_spec.wrap_guard_expr if for_wrap else none_spec.unwrap_guard_expr,  # type: ignore[arg-type]
                    none_spec.wrap_value_expr if for_wrap else none_spec.unwrap_value_expr,
                )
            )
        for spec in translated_specs:
            guard = spec.wrap_guard_expr if for_wrap else spec.unwrap_guard_expr
            value_expr = spec.wrap_value_expr if for_wrap else spec.unwrap_value_expr
            assert guard is not None
            cases.append((guard, value_expr))
        for spec in identity_known_specs:
            guard = spec.wrap_guard_expr if for_wrap else spec.unwrap_guard_expr
            value_expr = spec.wrap_value_expr if for_wrap else spec.unwrap_value_expr
            assert guard is not None
            cases.append((guard, value_expr))

        fallback_expr = (
            "_v"
            if has_identity_fallback
            else '(_ for _ in ()).throw(TypeError(f"Unexpected value for union translation: {type(_v)!r}"))'
        )
        expr = fallback_expr
        for guard, value_expr in reversed(cases):
            expr = f"({value_expr} if {guard} else {expr})"
        return f"((lambda _v: {expr})({var_name}))"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        return self._branch_expr(target_module or "", var_name, is_async=True, for_wrap=False)

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        return self._branch_expr(target_module, var_name, is_async=is_async, for_wrap=True)

    def requires_boundary_translation(self) -> bool:
        return any(t.requires_boundary_translation() for t in self.item_codegens)

    def helper_definitions(
        self,
        target_module: str,
        indent: str = "    ",
    ) -> dict[str, str]:
        helpers: dict[str, str] = {}
        for codegen in self.item_codegens:
            helpers.update(codegen.helper_definitions(target_module, indent))
        return helpers

    def references_wrapper_class(self) -> bool:
        return any(codegen.references_wrapper_class() for codegen in self.item_codegens)

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        item_types = [t.implementation_annotation(target_module, is_async) for t in self.item_codegens]
        return f"typing.Union[{', '.join(item_types)}]"


class AsyncGeneratorTypeCodegen(TypeCodegen):
    """Codegen for async generators, including yield and send translation."""

    def __init__(self, yield_codegen: TypeCodegen, send_codegen: TypeCodegen | None):
        self.yield_codegen = yield_codegen
        self.send_codegen = send_codegen
        self._uid = uuid.uuid4().hex[:8]

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        """Return AsyncGenerator[T, S] for async context, Generator[T, S, None] for sync context."""
        yield_type_str = self.yield_codegen.public_annotation(target_module, is_async)

        if self.send_codegen is None:
            send_type_str = None
        else:
            send_type_str = self.send_codegen.public_annotation(target_module, is_async)
        if is_async:
            # Always emit the send type: single-argument AsyncGenerator requires Python 3.13+.
            return f"typing.AsyncGenerator[{yield_type_str}, {send_type_str or 'None'}]"
        return f"typing.Generator[{yield_type_str}, {send_type_str or 'None'}, None]"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        """Generators don't unwrap at the parameter level."""
        return var_name

    def _needs_value_translation(self) -> bool:
        return self.yield_codegen.requires_boundary_translation() or (
            self.send_codegen is not None and self.send_codegen.requires_boundary_translation()
        )

    def _send_to_impl_expr(self, target_module: str) -> str:
        if self.send_codegen is None or not self.send_codegen.requires_boundary_translation():
            return "_sent"
        translated = self.send_codegen.wrapper_to_impl_expr("_sent", target_module)
        return f"(None if _sent is None else {translated})"

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        """Return expression that wraps an async generator.

        When neither yielded nor sent values need translation, delegates directly to the synchronizer.
        Otherwise, calls a generated helper function.
        """
        if not self._needs_value_translation():
            if is_async:
                return f"_synchronizer._run_generator_async({var_name})"
            else:
                return f"_synchronizer._run_generator_sync({var_name})"

        helper_name = self._get_helper_name(target_module)

        if is_async:
            return f"self.{helper_name}({var_name})"
        else:
            return f"self.{helper_name}_sync({var_name})"

    def requires_boundary_translation(self) -> bool:
        """Async generators ALWAYS need translation for synchronizer integration."""
        return True

    def _get_helper_name(self, target_module: str) -> str:
        """Generate a unique helper function name for this async generator wrapper."""
        yield_type_str = self.yield_codegen.public_annotation(target_module)
        sanitized = yield_type_str.replace("[", "_").replace("]", "").replace(".", "_").replace(", ", "_")
        return f"_wrap_async_gen_{sanitized}_{self._uid}"

    def helper_definitions(self, target_module: str, indent: str = "    ") -> dict[str, str]:
        """Generate helper functions for translating async generator values.

        Returns empty dict when values don't need translation (impl_to_wrapper_expr
        delegates directly to the synchronizer in that case).
        """
        if not self._needs_value_translation():
            return {}

        helpers: dict[str, str] = {}

        helpers.update(self.yield_codegen.helper_definitions(target_module, indent))
        if self.send_codegen is not None:
            helpers.update(self.send_codegen.helper_definitions(target_module, indent))

        helper_name = self._get_helper_name(target_module)
        impl_to_wrapper_expr = self.yield_codegen.impl_to_wrapper_expr(target_module, "_item")
        wrapper_to_impl_expr = self._send_to_impl_expr(target_module)

        async_helper = f"""{indent}@staticmethod
{indent}async def {helper_name}(_gen):
{indent}    _wrapped = _synchronizer._run_generator_async(_gen)
{indent}    _sent = None
{indent}    try:
{indent}        while True:
{indent}            try:
{indent}                _item = await _wrapped.asend({wrapper_to_impl_expr})
{indent}                _sent = yield {impl_to_wrapper_expr}
{indent}            except StopAsyncIteration:
{indent}                break
{indent}    finally:
{indent}        await _wrapped.aclose()"""

        sync_helper = f"""{indent}@staticmethod
{indent}def {helper_name}_sync(_gen):
{indent}    _wrapped = _synchronizer._run_generator_sync(_gen)
{indent}    _sent = None
{indent}    try:
{indent}        while True:
{indent}            try:
{indent}                _item = _wrapped.send({wrapper_to_impl_expr})
{indent}                _sent = yield {impl_to_wrapper_expr}
{indent}            except StopIteration:
{indent}                break
{indent}    finally:
{indent}        _wrapped.close()"""

        helpers[helper_name] = async_helper
        helpers[f"{helper_name}_sync"] = sync_helper

        return helpers

    def references_wrapper_class(self) -> bool:
        return self.yield_codegen.references_wrapper_class() or (
            self.send_codegen is not None and self.send_codegen.references_wrapper_class()
        )

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        yield_type_str = self.yield_codegen.implementation_annotation(target_module, is_async)
        send_type_str = (
            None if self.send_codegen is None else self.send_codegen.implementation_annotation(target_module, is_async)
        )
        if is_async:
            # Always emit the send type: single-argument AsyncGenerator requires Python 3.13+.
            return f"typing.AsyncGenerator[{yield_type_str}, {send_type_str or 'None'}]"
        return f"typing.Generator[{yield_type_str}, {send_type_str or 'None'}, None]"


class SyncGeneratorTypeCodegen(TypeCodegen):
    """Codegen for sync Generator types."""

    def __init__(self, yield_codegen: TypeCodegen, send_codegen: TypeCodegen, return_codegen: TypeCodegen):
        self.yield_codegen = yield_codegen
        self.send_codegen = send_codegen
        self.return_codegen = return_codegen
        self._uid = uuid.uuid4().hex[:8]

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        """Return the public three-parameter generator annotation."""
        yield_type_str = self.yield_codegen.public_annotation(target_module, is_async)
        send_type_str = self.send_codegen.public_annotation(target_module, is_async)
        return_type_str = self.return_codegen.public_annotation(target_module, is_async)
        return f"typing.Generator[{yield_type_str}, {send_type_str}, {return_type_str}]"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        """Generators don't unwrap at the parameter level."""
        return var_name

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        """Return expression that wraps a sync generator."""
        if not self.requires_boundary_translation():
            return var_name

        helper_name = self._get_helper_name(target_module)
        return f"self.{helper_name}({var_name})"

    def requires_boundary_translation(self) -> bool:
        """Whether any value crossing the generator boundary needs translation."""
        return any(
            codegen.requires_boundary_translation()
            for codegen in (self.yield_codegen, self.send_codegen, self.return_codegen)
        )

    def _get_helper_name(self, target_module: str) -> str:
        """Generate a unique helper function name for this sync generator wrapper."""
        yield_type_str = self.yield_codegen.public_annotation(target_module)
        sanitized = yield_type_str.replace("[", "_").replace("]", "").replace(".", "_").replace(", ", "_")
        return f"_wrap_gen_{sanitized}_{self._uid}"

    def _send_to_impl_expr(self, target_module: str) -> str:
        if not self.send_codegen.requires_boundary_translation():
            return "_sent"
        translated = self.send_codegen.wrapper_to_impl_expr("_sent", target_module)
        return f"(None if _sent is None else {translated})"

    def helper_definitions(self, target_module: str, indent: str = "    ") -> dict[str, str]:
        """Generate helper function for wrapping sync generators."""
        helpers: dict[str, str] = {}
        for codegen in (self.yield_codegen, self.send_codegen, self.return_codegen):
            helpers.update(codegen.helper_definitions(target_module, indent))

        if not self.requires_boundary_translation():
            return helpers

        helper_name = self._get_helper_name(target_module)

        yield_expr = self.yield_codegen.impl_to_wrapper_expr(target_module, "_item")
        send_expr = self._send_to_impl_expr(target_module)
        return_expr = self.return_codegen.impl_to_wrapper_expr(target_module, "_stop.value")
        helper_code = f"""{indent}@staticmethod
{indent}def {helper_name}(_gen):
{indent}    _sent = None
{indent}    try:
{indent}        while True:
{indent}            try:
{indent}                _item = _gen.send({send_expr})
{indent}                _sent = yield {yield_expr}
{indent}            except StopIteration as _stop:
{indent}                return {return_expr}
{indent}    finally:
{indent}        _gen.close()"""
        helpers[helper_name] = helper_code

        return helpers

    def references_wrapper_class(self) -> bool:
        return any(
            codegen.references_wrapper_class()
            for codegen in (self.yield_codegen, self.send_codegen, self.return_codegen)
        )

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        yield_type_str = self.yield_codegen.implementation_annotation(target_module, is_async)
        send_type_str = self.send_codegen.implementation_annotation(target_module, is_async)
        return_type_str = self.return_codegen.implementation_annotation(target_module, is_async)
        return f"typing.Generator[{yield_type_str}, {send_type_str}, {return_type_str}]"


class SyncIteratorTypeCodegen(TypeCodegen):
    """Codegen for synchronous iterators with translated item values."""

    def __init__(self, item_codegen: TypeCodegen):
        self.item_codegen = item_codegen
        self._uid = uuid.uuid4().hex[:8]

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        item_type_str = self.item_codegen.public_annotation(target_module, is_async)
        return f"typing.Iterator[{item_type_str}]"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        return var_name

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        if not self.requires_boundary_translation():
            return var_name
        return f"self.{self._get_helper_name(target_module)}({var_name})"

    def requires_boundary_translation(self) -> bool:
        return self.item_codegen.requires_boundary_translation()

    def _get_helper_name(self, target_module: str) -> str:
        item_type_str = self.item_codegen.public_annotation(target_module)
        sanitized = item_type_str.replace("[", "_").replace("]", "").replace(".", "_").replace(", ", "_")
        return f"_wrap_iter_{sanitized}_{self._uid}"

    def helper_definitions(self, target_module: str, indent: str = "    ") -> dict[str, str]:
        helpers = self.item_codegen.helper_definitions(target_module, indent)
        if not self.requires_boundary_translation():
            return helpers

        helper_name = self._get_helper_name(target_module)
        item_expr = self.item_codegen.impl_to_wrapper_expr(target_module, "_item")
        helpers[helper_name] = f"""{indent}@staticmethod
{indent}def {helper_name}(_iter):
{indent}    for _item in _iter:
{indent}        yield {item_expr}"""
        return helpers

    def references_wrapper_class(self) -> bool:
        return self.item_codegen.references_wrapper_class()

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        item_type_str = self.item_codegen.implementation_annotation(target_module, is_async)
        return f"typing.Iterator[{item_type_str}]"


class AsyncIteratorTypeCodegen(TypeCodegen):
    """Codegen for AsyncIterator types (not generators).

    AsyncIterator is more general than AsyncGenerator - it only has __aiter__() and __anext__(),
    not asend()/aclose(). This codegen handles iterators that aren't generators.
    """

    def __init__(self, item_codegen: TypeCodegen, runtime_package: str = "synchronicity2"):
        self.item_codegen = item_codegen
        self._runtime_package = runtime_package

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        """Return SyncOrAsyncIterator[T] - works in both sync and async contexts."""
        item_type_str = self.item_codegen.public_annotation(target_module, is_async)
        return f"{self._runtime_package}.types.SyncOrAsyncIterator[{item_type_str}]"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        """Iterators don't unwrap at the parameter level."""
        return var_name

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        """Return expression that creates a SyncOrAsyncIterator wrapping an async iterator."""
        if not self.item_codegen.requires_boundary_translation():
            return f"{self._runtime_package}.types.SyncOrAsyncIterator({var_name}, _synchronizer)"

        item_impl_to_wrapper_expr = self.item_codegen.impl_to_wrapper_expr(target_module, "_item", is_async=True)
        item_impl_to_wrapper_expr = item_impl_to_wrapper_expr.replace("self.", "")

        helper_name = self._get_helper_name(target_module)
        return (
            f"{self._runtime_package}.types.SyncOrAsyncIterator({var_name}, _synchronizer, item_wrapper={helper_name})"
        )

    def requires_boundary_translation(self) -> bool:
        """AsyncIterators ALWAYS need translation to convert from async to sync."""
        return True

    def _get_helper_name(self, target_module: str) -> str:
        """Generate a unique helper function name for this async iterator wrapper."""
        item_type_str = self.item_codegen.public_annotation(target_module)
        sanitized = item_type_str.replace("[", "_").replace("]", "").replace(".", "_").replace(", ", "_")
        return f"_wrap_async_iter_{sanitized}"

    def helper_definitions(self, target_module: str, indent: str = "    ") -> dict[str, str]:
        """Generate helper functions for wrapping iterator items if needed."""
        helpers = {}

        helpers.update(self.item_codegen.helper_definitions(target_module, indent))

        if self.item_codegen.requires_boundary_translation():
            helper_name = self._get_helper_name(target_module)
            impl_to_wrapper_expr = self.item_codegen.impl_to_wrapper_expr(target_module, "_item", is_async=True)
            impl_to_wrapper_expr = impl_to_wrapper_expr.replace("self.", "")

            helper_func = f"""def {helper_name}(_item):
    return {impl_to_wrapper_expr}"""

            helpers[helper_name] = helper_func

        return helpers

    def references_wrapper_class(self) -> bool:
        return self.item_codegen.references_wrapper_class()

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        item_type_str = self.item_codegen.implementation_annotation(target_module, is_async)
        return f"collections.abc.AsyncIterator[{item_type_str}]"


class AsyncIterableTypeCodegen(TypeCodegen):
    """Codegen for AsyncIterable[T] types.

    AsyncIterable objects have an __aiter__() method that returns an AsyncIterator.
    For sync wrappers, we convert to regular Iterable[T].
    For async wrappers, we keep AsyncIterable[T].
    """

    def __init__(self, item_codegen: TypeCodegen, runtime_package: str = "synchronicity2"):
        """Initialize with the codegen for the item type."""
        self.item_codegen = item_codegen
        self._runtime_package = runtime_package

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        """Return SyncOrAsyncIterable[T] - works in both sync and async contexts."""
        item_type_str = self.item_codegen.public_annotation(target_module, is_async)
        return f"{self._runtime_package}.types.SyncOrAsyncIterable[{item_type_str}]"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        """No unwrapping needed for async iterables."""
        return var_name

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        """Return expression that creates a SyncOrAsyncIterable wrapping an async iterable."""
        if not self.item_codegen.requires_boundary_translation():
            return f"{self._runtime_package}.types.SyncOrAsyncIterable({var_name}, _synchronizer)"

        helper_suffix = (
            self.item_codegen.public_annotation(target_module, is_async)
            .replace(".", "_")
            .replace("[", "_")
            .replace("]", "")
            .replace(", ", "_")
            .replace(" ", "")
        )
        helper_name = f"_wrap_async_iterable_item_{helper_suffix}"
        return (
            f"{self._runtime_package}.types.SyncOrAsyncIterable({var_name}, _synchronizer, item_wrapper={helper_name})"
        )

    def requires_boundary_translation(self) -> bool:
        """AsyncIterable always needs translation to provide sync/async versions."""
        return True

    def helper_definitions(self, target_module: str, indent: str = "    ") -> dict[str, str]:
        """Generate helper functions for wrapping iterable items if needed."""
        helpers = {}

        helpers.update(self.item_codegen.helper_definitions(target_module, indent))

        if self.item_codegen.requires_boundary_translation():
            public_annotation = self.item_codegen.public_annotation(target_module, is_async=True)
            helper_suffix = (
                public_annotation.replace(".", "_")
                .replace("[", "_")
                .replace("]", "")
                .replace(", ", "_")
                .replace(" ", "")
            )
            helper_name = f"_wrap_async_iterable_item_{helper_suffix}"

            item_impl_to_wrapper_expr = self.item_codegen.impl_to_wrapper_expr(target_module, "_item", is_async=True)
            item_impl_to_wrapper_expr = item_impl_to_wrapper_expr.replace("self.", "")

            helper_func = f"""def {helper_name}(_item):
    return {item_impl_to_wrapper_expr}"""

            helpers[helper_name] = helper_func

        return helpers

    def references_wrapper_class(self) -> bool:
        return self.item_codegen.references_wrapper_class()

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        item_type_str = self.item_codegen.implementation_annotation(target_module, is_async)
        return f"collections.abc.AsyncIterable[{item_type_str}]"


class AsyncContextManagerTypeCodegen(TypeCodegen):
    """Codegen for AsyncContextManager[T] types.

    Wraps an async context manager into SyncOrAsyncContextManager[T] which supports
    both ``with`` (sync) and ``async with`` (async) usage.
    """

    def __init__(self, value_codegen: TypeCodegen, runtime_package: str = "synchronicity2"):
        self.value_codegen = value_codegen
        self._runtime_package = runtime_package

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        value_type_str = self.value_codegen.public_annotation(target_module, is_async)
        return f"{self._runtime_package}.types.SyncOrAsyncContextManager[{value_type_str}]"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        return var_name

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        if not self.value_codegen.requires_boundary_translation():
            return f"{self._runtime_package}.types.SyncOrAsyncContextManager({var_name}, _synchronizer)"
        helper_name = self._get_helper_name(target_module)
        return (
            f"{self._runtime_package}.types.SyncOrAsyncContextManager({var_name}, "
            f"_synchronizer, value_wrapper=self.{helper_name})"
        )

    def requires_boundary_translation(self) -> bool:
        return True

    def _get_helper_name(self, target_module: str) -> str:
        value_type_str = self.value_codegen.public_annotation(target_module)
        sanitized = value_type_str.replace("[", "_").replace("]", "").replace(".", "_").replace(", ", "_")
        return f"_wrap_async_cm_{sanitized}"

    def helper_definitions(self, target_module: str, indent: str = "    ") -> dict[str, str]:
        helpers = {}
        helpers.update(self.value_codegen.helper_definitions(target_module, indent))

        if self.value_codegen.requires_boundary_translation():
            helper_name = self._get_helper_name(target_module)
            impl_to_wrapper_expr = self.value_codegen.impl_to_wrapper_expr(target_module, "_item", is_async=True)
            impl_to_wrapper_expr = impl_to_wrapper_expr.replace("self.", "")

            helper_func = f"""{indent}@staticmethod
{indent}def {helper_name}(_item):
{indent}    return {impl_to_wrapper_expr}"""
            helpers[helper_name] = helper_func

        return helpers

    def references_wrapper_class(self) -> bool:
        return self.value_codegen.references_wrapper_class()

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        value_type_str = self.value_codegen.implementation_annotation(target_module, is_async)
        return f"typing.AsyncContextManager[{value_type_str}]"


class CoroutineTypeCodegen(TypeCodegen):
    """Codegen for Coroutine[YieldType, SendType, ReturnType] types.

    When a function returns Coroutine[Any, Any, T], the wrapper functions should
    have return type T, since:
    - The async wrapper is declared as `async def` which implicitly makes it awaitable
    - The sync wrapper uses the synchronizer to unwrap and return T
    """

    def __init__(self, return_codegen: TypeCodegen):
        """Initialize with the codegen for the return type (third type arg)."""
        self.return_codegen = return_codegen

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        """Return the unwrapped return type."""
        return self.return_codegen.public_annotation(target_module, is_async)

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        """No unwrapping needed - the coroutine itself is passed through."""
        return var_name

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        """No wrapping needed - the coroutine itself is passed through."""
        return var_name

    def requires_boundary_translation(self) -> bool:
        """Coroutines need translation - they must be awaited/run through synchronizer."""
        return True

    def requires_await(self) -> bool:
        """Signal that this type needs to be awaited or run through synchronizer."""
        return True

    def references_wrapper_class(self) -> bool:
        return self.return_codegen.references_wrapper_class()

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        return_type_str = self.return_codegen.implementation_annotation(target_module, is_async)
        return f"typing.Coroutine[typing.Any, typing.Any, {return_type_str}]"


class AwaitableTypeCodegen(TypeCodegen):
    """Codegen for Awaitable[T] types.

    When a function returns Awaitable[T], the wrapper functions should
    have return type T, since:
    - The async wrapper is declared as `async def` which implicitly makes it awaitable
    - The sync wrapper uses the synchronizer to unwrap and return T
    """

    def __init__(self, return_codegen: TypeCodegen):
        """Initialize with the codegen for the return type."""
        self.return_codegen = return_codegen

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        """Return the unwrapped return type."""
        return self.return_codegen.public_annotation(target_module, is_async)

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        """No unwrapping needed - the awaitable itself is passed through."""
        return var_name

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        """No wrapping needed - the awaitable itself is passed through."""
        return var_name

    def requires_boundary_translation(self) -> bool:
        """Awaitables need translation - they must be awaited/run through synchronizer."""
        return True

    def requires_await(self) -> bool:
        """Signal that this type needs to be awaited or run through synchronizer."""
        return True

    def references_wrapper_class(self) -> bool:
        return self.return_codegen.references_wrapper_class()

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        return_type_str = self.return_codegen.implementation_annotation(target_module, is_async)
        return f"typing.Awaitable[{return_type_str}]"


class CallableTypeCodegen(TypeCodegen):
    """Codegen for typing.Callable[[...], T] and typing.Callable[..., T]."""

    def __init__(
        self,
        param_codegens: tuple[TypeCodegen, ...] | None,
        return_codegen: TypeCodegen,
        *,
        param_signature_text: str | None = None,
    ):
        self.param_codegens = param_codegens
        self.return_codegen = return_codegen
        self.param_signature_text = param_signature_text

    def public_annotation(self, target_module: str, is_async: bool = True) -> str:
        if self.param_codegens is None:
            params_str = self.param_signature_text or "..."
        else:
            param_types = (t.public_annotation(target_module, is_async) for t in self.param_codegens)
            params_str = f"[{', '.join(param_types)}]"
        return_type_str = self.return_codegen.public_annotation(target_module, is_async)
        return f"typing.Callable[{params_str}, {return_type_str}]"

    def implementation_annotation(self, target_module: str, is_async: bool = True) -> str:
        if self.param_codegens is None:
            params_str = self.param_signature_text or "..."
        else:
            param_types = (t.implementation_annotation(target_module, is_async) for t in self.param_codegens)
            params_str = f"[{', '.join(param_types)}]"
        return_type_str = self.return_codegen.implementation_annotation(target_module, is_async)
        return f"typing.Callable[{params_str}, {return_type_str}]"

    def _translated_callback_args_expr(self, target_module: str, *, wrapper_to_impl: bool) -> str:
        if self.param_codegens is None:
            return "*_callback_args, **_callback_kwargs"

        translated_args: list[str] = []
        for index, codegen in enumerate(self.param_codegens):
            arg_expr = f"_callback_args[{index}]"
            if not codegen.requires_boundary_translation():
                translated_args.append(arg_expr)
            elif wrapper_to_impl:
                if isinstance(codegen, CallableTypeCodegen):
                    translated_args.append(codegen.wrapper_to_impl_expr(arg_expr, target_module))
                else:
                    translated_args.append(codegen.wrapper_to_impl_expr(arg_expr))
            else:
                translated_args.append(codegen.impl_to_wrapper_expr(target_module, arg_expr, is_async=False))

        translated_tuple = f"({', '.join(translated_args)}"
        if len(translated_args) == 1:
            translated_tuple += ","
        translated_tuple += ")"
        rest_tuple = f"tuple(_callback_args[{len(self.param_codegens)}:])"
        return f"*({translated_tuple} + {rest_tuple}), **_callback_kwargs"

    def wrapper_to_impl_expr(self, var_name: str, target_module: str | None = None) -> str:
        return f"typing.cast(typing.Any, {var_name})"

    def impl_to_wrapper_expr(self, target_module: str, var_name: str, is_async: bool = True) -> str:
        callback_args = self._translated_callback_args_expr(target_module, wrapper_to_impl=True)
        call_expr = f"_impl_callable({callback_args})"
        if self.return_codegen.requires_boundary_translation():
            call_expr = self.return_codegen.impl_to_wrapper_expr(target_module, call_expr, is_async)
        return f"(lambda _impl_callable: (lambda *_callback_args, **_callback_kwargs: {call_expr}))({var_name})"

    def requires_boundary_translation(self) -> bool:
        params_need_translation = (
            False
            if self.param_codegens is None
            else any(t.requires_boundary_translation() for t in self.param_codegens)
        )
        return params_need_translation or self.return_codegen.requires_boundary_translation()

    def references_wrapper_class(self) -> bool:
        param_has_wrapper_ref = (
            False if self.param_codegens is None else any(t.references_wrapper_class() for t in self.param_codegens)
        )
        return param_has_wrapper_ref or self.return_codegen.references_wrapper_class()


def _codegen_for_typevar(
    annotation_ir: TypeVarRefIR,
    runtime_package: str,
    context: TypeCodegenContext | None,
) -> TypeCodegen:
    if context is None or context.type_parameters_by_name is None:
        return PlainTypeCodegen(annotation_ir.name)
    spec = context.type_parameters_by_name.get(annotation_ir.name)
    if spec is None or spec.is_paramspec:
        return PlainTypeCodegen(annotation_ir.name)
    if spec.bound_annotation_ir is not None:
        bound_codegen = codegen_for_annotation(spec.bound_annotation_ir, runtime_package, context=context)
        return BoundTypeVarCodegen(annotation_ir.name, bound_codegen)
    return PlainTypeCodegen(annotation_ir.name)


def codegen_for_annotation(
    annotation_ir: AnnotationIR,
    runtime_package: str,
    *,
    context: TypeCodegenContext | None = None,
) -> TypeCodegen:
    """Build source-generation behavior for a parsed annotation IR node."""
    if isinstance(annotation_ir, PlainAnnotationIR):
        return PlainTypeCodegen(annotation_ir.signature_text)
    if isinstance(annotation_ir, WrappedClassRefIR):
        return WrappedClassTypeCodegen(annotation_ir.impl, annotation_ir.wrapper)
    if isinstance(annotation_ir, Synchronicity1WrappedClassRefIR):
        return Synchronicity1WrappedClassTypeCodegen(annotation_ir.impl, annotation_ir.wrapper)
    if isinstance(annotation_ir, TypeVarRefIR):
        return _codegen_for_typevar(annotation_ir, runtime_package, context)
    if isinstance(annotation_ir, SelfAnnotationIR):
        return SelfTypeCodegen(annotation_ir.owner_impl, annotation_ir.wrapper)
    if isinstance(annotation_ir, ListAnnotationIR):
        return ListTypeCodegen(codegen_for_annotation(annotation_ir.item_ir, runtime_package, context=context))
    if isinstance(annotation_ir, DictAnnotationIR):
        return DictTypeCodegen(
            codegen_for_annotation(annotation_ir.key_ir, runtime_package, context=context),
            codegen_for_annotation(annotation_ir.value_ir, runtime_package, context=context),
        )
    if isinstance(annotation_ir, SequenceAnnotationIR):
        return SequenceTypeCodegen(codegen_for_annotation(annotation_ir.item_ir, runtime_package, context=context))
    if isinstance(annotation_ir, CollectionAnnotationIR):
        return CollectionTypeCodegen(codegen_for_annotation(annotation_ir.item_ir, runtime_package, context=context))
    if isinstance(annotation_ir, TupleAnnotationIR):
        if annotation_ir.variadic:
            (single,) = annotation_ir.element_irs
            return TupleTypeCodegen([codegen_for_annotation(single, runtime_package, context=context)])
        return TupleTypeCodegen(
            [codegen_for_annotation(element, runtime_package, context=context) for element in annotation_ir.element_irs]
        )
    if isinstance(annotation_ir, OptionalAnnotationIR):
        return OptionalTypeCodegen(codegen_for_annotation(annotation_ir.inner_ir, runtime_package, context=context))
    if isinstance(annotation_ir, UnionAnnotationIR):
        return UnionTypeCodegen(
            [codegen_for_annotation(arm, runtime_package, context=context) for arm in annotation_ir.arm_irs],
            source_label=annotation_ir.source_label,
        )
    if isinstance(annotation_ir, AsyncGeneratorAnnotationIR):
        return AsyncGeneratorTypeCodegen(
            codegen_for_annotation(annotation_ir.yield_annotation_ir, runtime_package, context=context),
            (
                None
                if annotation_ir.send_annotation_ir is None
                else codegen_for_annotation(annotation_ir.send_annotation_ir, runtime_package, context=context)
            ),
        )
    if isinstance(annotation_ir, SyncGeneratorAnnotationIR):
        return SyncGeneratorTypeCodegen(
            codegen_for_annotation(annotation_ir.yield_annotation_ir, runtime_package, context=context),
            codegen_for_annotation(annotation_ir.send_annotation_ir, runtime_package, context=context),
            codegen_for_annotation(annotation_ir.return_annotation_ir, runtime_package, context=context),
        )
    if isinstance(annotation_ir, SyncIteratorAnnotationIR):
        return SyncIteratorTypeCodegen(codegen_for_annotation(annotation_ir.item_ir, runtime_package, context=context))
    if isinstance(annotation_ir, AsyncIteratorAnnotationIR):
        return AsyncIteratorTypeCodegen(
            codegen_for_annotation(annotation_ir.item_ir, runtime_package, context=context),
            runtime_package,
        )
    if isinstance(annotation_ir, AsyncIterableAnnotationIR):
        return AsyncIterableTypeCodegen(
            codegen_for_annotation(annotation_ir.item_ir, runtime_package, context=context),
            runtime_package,
        )
    if isinstance(annotation_ir, CoroutineAnnotationIR):
        return CoroutineTypeCodegen(
            codegen_for_annotation(annotation_ir.return_annotation_ir, runtime_package, context=context)
        )
    if isinstance(annotation_ir, AwaitableAnnotationIR):
        return AwaitableTypeCodegen(codegen_for_annotation(annotation_ir.inner_ir, runtime_package, context=context))
    if isinstance(annotation_ir, CallableAnnotationIR):
        parameter_codegen = (
            None
            if annotation_ir.parameter_irs is None
            else tuple(
                codegen_for_annotation(parameter_ir, runtime_package, context=context)
                for parameter_ir in annotation_ir.parameter_irs
            )
        )
        return CallableTypeCodegen(
            parameter_codegen,
            codegen_for_annotation(annotation_ir.return_annotation_ir, runtime_package, context=context),
            param_signature_text=annotation_ir.params_signature_text,
        )
    if isinstance(annotation_ir, ParameterizedWrappedClassRefIR):
        type_argument_codegen = [
            codegen_for_annotation(argument_ir, runtime_package, context=context)
            for argument_ir in annotation_ir.type_argument_irs
        ]
        inner_codegen = codegen_for_annotation(annotation_ir.wrapped_class_ir, runtime_package, context=context)
        assert isinstance(inner_codegen, (WrappedClassTypeCodegen, Synchronicity1WrappedClassTypeCodegen))
        return ParameterizedWrappedClassTypeCodegen(inner_codegen, type_argument_codegen)
    if isinstance(annotation_ir, AsyncContextManagerAnnotationIR):
        return AsyncContextManagerTypeCodegen(
            codegen_for_annotation(annotation_ir.value_ir, runtime_package, context=context),
            runtime_package,
        )
    raise TypeError(f"Unhandled annotation IR: {type(annotation_ir)!r}")
