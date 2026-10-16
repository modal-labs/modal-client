# Copyright Modal Labs 2026
"""Unit tests for type codegens.

Tests each codegen type for:
1. Type signature formatting (public_annotation)
2. Unwrap expressions (wrapper → impl)
3. Wrap expressions (impl → wrapper)
4. Actual execution of wrap/unwrap code
"""

import pytest
import sys
import types
import typing

from synchronicity import Synchronizer as Synchronicity1Synchronizer

from synchronicity2.codegen.emission.type_codegen import (
    AsyncGeneratorTypeCodegen,
    AwaitableTypeCodegen,
    CoroutineTypeCodegen,
    DictTypeCodegen,
    ListTypeCodegen,
    OptionalTypeCodegen,
    ParameterizedWrappedClassTypeCodegen,
    PlainTypeCodegen,
    SyncGeneratorTypeCodegen,
    Synchronicity1WrappedClassTypeCodegen,
    SyncIteratorTypeCodegen,
    TupleTypeCodegen,
    UnionTypeCodegen,
    WrappedClassTypeCodegen,
    codegen_for_annotation,
)
from synchronicity2.codegen.ir.annotations import Synchronicity1WrappedClassRefIR
from synchronicity2.codegen.ir.references import ObjectReferenceIR
from synchronicity2.codegen.parsing.annotations import parse_annotation

_WRAPPER_LOCATION_ATTR = "__synchronicity_wrapper_location__"


def _make_wrapped_codegen(wrapped_class):
    """Helper to create a WrappedClassTypeCodegen from a class with wrapper location set."""
    impl_ref = ObjectReferenceIR(module=wrapped_class.__module__, qualname=wrapped_class.__qualname__)
    wrapper_ref = ObjectReferenceIR("test_module", "TestClass")
    return WrappedClassTypeCodegen(impl_ref, wrapper_ref)


def _codegen_from_annotation(annotation, **kwargs):
    """Annotation → AnnotationIR → runtime codegen (same path as codegen)."""
    ir = parse_annotation(annotation, **kwargs)
    return codegen_for_annotation(ir, "synchronicity2")


@pytest.fixture
def wrapped_class():
    class TestClass:
        def __init__(self, value: int):
            self.value = value

    setattr(TestClass, _WRAPPER_LOCATION_ATTR, ("test_module", "TestClass"))
    return TestClass


class TestPlainTypeCodegen:
    """Test PlainTypeCodegen for primitive types."""

    def test_public_annotation_int(self):
        codegen = PlainTypeCodegen("int")
        assert codegen.public_annotation("test_module") == "int"

    def test_public_annotation_str(self):
        codegen = PlainTypeCodegen("str")
        assert codegen.public_annotation("test_module") == "str"

    def test_wrapper_to_impl_expr_returns_same(self):
        codegen = PlainTypeCodegen("int")
        assert codegen.wrapper_to_impl_expr("value") == "value"

    def test_impl_to_wrapper_expr_returns_same(self):
        codegen = PlainTypeCodegen("str")
        assert codegen.impl_to_wrapper_expr("test_module", "value") == "value"

    def test_requires_boundary_translation_false(self):
        codegen = PlainTypeCodegen("int")
        assert codegen.requires_boundary_translation() is False

    def test_execution_passthrough(self):
        """Test that identity codegen passes values through."""
        codegen = PlainTypeCodegen("int")

        # Unwrap should be identity
        value = 42
        unwrap_code = f"result = {codegen.wrapper_to_impl_expr('value')}"
        exec(unwrap_code, {"value": value}, locals_dict := {})
        assert locals_dict["result"] == 42

        # Wrap should be identity
        wrap_code = f"result = {codegen.impl_to_wrapper_expr('test_module', 'value')}"
        exec(wrap_code, {"value": value}, locals_dict := {})
        assert locals_dict["result"] == 42


class TestWrappedClassTypeCodegen:
    """Test WrappedClassTypeCodegen for wrapped classes."""

    def test_public_annotation_local_reference(self, wrapped_class):
        codegen = _make_wrapped_codegen(wrapped_class)
        # When target_module matches, should return just class name
        result = codegen.public_annotation("test_module")
        assert result == "TestClass"

    def test_public_annotation_cross_module_reference(self, wrapped_class):
        codegen = _make_wrapped_codegen(wrapped_class)
        # When target_module differs, should return fully qualified name
        result = codegen.public_annotation("other_module")
        assert result == "test_module.TestClass"

    def test_wrapper_to_impl_expr(self, wrapped_class):
        codegen = _make_wrapped_codegen(wrapped_class)
        assert codegen.wrapper_to_impl_expr("obj") == "obj._impl_instance"

    def test_impl_to_wrapper_expr_local(self, wrapped_class):
        codegen = _make_wrapped_codegen(wrapped_class)
        result = codegen.impl_to_wrapper_expr("test_module", "impl")
        assert result == "TestClass._from_impl(impl)"

    def test_impl_to_wrapper_expr_cross_module(self, wrapped_class):
        codegen = _make_wrapped_codegen(wrapped_class)
        result = codegen.impl_to_wrapper_expr("other_module", "impl")
        assert result == "test_module.TestClass._from_impl(impl)"

    def test_requires_boundary_translation_true(self, wrapped_class):
        codegen = _make_wrapped_codegen(wrapped_class)
        assert codegen.requires_boundary_translation() is True


class TestSynchronicity1WrappedClassTypeCodegen:
    def test_annotation_creates_distinct_ir_and_codegen(self):
        class LegacyImpl:
            pass

        synchronizer = Synchronicity1Synchronizer()
        synchronizer.wrap(LegacyImpl, name="Legacy", target_module="legacy_api")
        try:
            ir = parse_annotation(
                LegacyImpl,
                synchronicity1_synchronizer=synchronizer,
            )
            codegen = codegen_for_annotation(ir, "synchronicity2")

            assert isinstance(ir, Synchronicity1WrappedClassRefIR)
            assert isinstance(codegen, Synchronicity1WrappedClassTypeCodegen)
            assert codegen.public_annotation("generated_api") == "legacy_api.Legacy"
            assert "_synchronicity1._translate_in(value)" in codegen.wrapper_to_impl_expr("value")
            assert "_synchronicity1._translate_out(result)" in codegen.impl_to_wrapper_expr("generated_api", "result")
        finally:
            synchronizer._close_loop()


class TestListTypeCodegen:
    """Test ListTypeCodegen for list[T] types."""

    def test_public_annotation_primitives(self):
        item_codegen = PlainTypeCodegen("int")
        codegen = ListTypeCodegen(item_codegen)
        assert codegen.public_annotation("test_module") == "list[int]"

    def test_public_annotation_wrapped_class(self, wrapped_class):
        item_codegen = _make_wrapped_codegen(wrapped_class)
        codegen = ListTypeCodegen(item_codegen)
        assert codegen.public_annotation("test_module") == "list[TestClass]"

    def test_wrapper_to_impl_expr_primitives(self):
        """Primitives don't need unwrapping."""
        item_codegen = PlainTypeCodegen("int")
        codegen = ListTypeCodegen(item_codegen)
        assert codegen.wrapper_to_impl_expr("values") == "values"

    def test_wrapper_to_impl_expr_wrapped_class(self, wrapped_class):
        item_codegen = _make_wrapped_codegen(wrapped_class)
        codegen = ListTypeCodegen(item_codegen)
        result = codegen.wrapper_to_impl_expr("items")
        assert result == "[x._impl_instance for x in items]"

    def test_impl_to_wrapper_expr_wrapped_class(self, wrapped_class):
        item_codegen = _make_wrapped_codegen(wrapped_class)
        codegen = ListTypeCodegen(item_codegen)
        result = codegen.impl_to_wrapper_expr("test_module", "impl_items")
        assert result == "[TestClass._from_impl(x) for x in impl_items]"

    def test_requires_boundary_translation(self, wrapped_class):
        # List of primitives doesn't need translation
        codegen1 = ListTypeCodegen(PlainTypeCodegen("int"))
        assert codegen1.requires_boundary_translation() is False

        # List of wrapped classes needs translation
        codegen2 = ListTypeCodegen(_make_wrapped_codegen(wrapped_class))
        assert codegen2.requires_boundary_translation() is True

    def test_execution_list_comprehension(self):
        """Test that list comprehension unwrap code executes correctly."""

        # Create mock objects with _impl_instance
        class MockWrapper:
            def __init__(self, value):
                self._impl_instance = value

        items = [MockWrapper(1), MockWrapper(2), MockWrapper(3)]

        # Execute the unwrap expression
        wrapper_to_impl_expr = "[x._impl_instance for x in items]"
        result = eval(wrapper_to_impl_expr, {"items": items})

        assert result == [1, 2, 3]


class TestDictTypeCodegen:
    """Test DictTypeCodegen for dict[K, V] types."""

    def test_public_annotation(self, wrapped_class):
        key_codegen = PlainTypeCodegen("str")
        value_codegen = _make_wrapped_codegen(wrapped_class)
        codegen = DictTypeCodegen(key_codegen, value_codegen)
        assert codegen.public_annotation("test_module") == "dict[str, TestClass]"

    def test_wrapper_to_impl_expr_primitives(self):
        """Dict with primitive values doesn't need unwrapping."""
        key_codegen = PlainTypeCodegen("str")
        value_codegen = PlainTypeCodegen("int")
        codegen = DictTypeCodegen(key_codegen, value_codegen)
        assert codegen.wrapper_to_impl_expr("mapping") == "mapping"

    def test_wrapper_to_impl_expr_wrapped_values(self, wrapped_class):
        key_codegen = PlainTypeCodegen("str")
        value_codegen = _make_wrapped_codegen(wrapped_class)
        codegen = DictTypeCodegen(key_codegen, value_codegen)
        result = codegen.wrapper_to_impl_expr("mapping")
        assert result == "{k: v._impl_instance for k, v in mapping.items()}"

    def test_impl_to_wrapper_expr_wrapped_values(self, wrapped_class):
        key_codegen = PlainTypeCodegen("str")
        value_codegen = _make_wrapped_codegen(wrapped_class)
        codegen = DictTypeCodegen(key_codegen, value_codegen)
        result = codegen.impl_to_wrapper_expr("test_module", "impl_mapping")
        assert result == "{k: TestClass._from_impl(v) for k, v in impl_mapping.items()}"

    def test_requires_boundary_translation(self, wrapped_class):
        # Dict with primitive values doesn't need translation
        codegen1 = DictTypeCodegen(PlainTypeCodegen("str"), PlainTypeCodegen("int"))
        assert codegen1.requires_boundary_translation() is False

        # Dict with wrapped values needs translation
        codegen2 = DictTypeCodegen(PlainTypeCodegen("str"), _make_wrapped_codegen(wrapped_class))
        assert codegen2.requires_boundary_translation() is True

    def test_execution_dict_comprehension(self):
        """Test that dict comprehension unwrap code executes correctly."""

        class MockWrapper:
            def __init__(self, value):
                self._impl_instance = value

        mapping = {"a": MockWrapper(1), "b": MockWrapper(2)}

        # Execute the unwrap expression
        wrapper_to_impl_expr = "{k: v._impl_instance for k, v in mapping.items()}"
        result = eval(wrapper_to_impl_expr, {"mapping": mapping})

        assert result == {"a": 1, "b": 2}


class TestTupleTypeCodegen:
    """Test TupleTypeCodegen for tuple types."""

    def test_public_annotation_variable_length(self):
        """Test tuple[T, ...] formatting."""
        item_codegen = PlainTypeCodegen("int")
        codegen = TupleTypeCodegen([item_codegen])
        assert codegen.public_annotation("test_module") == "tuple[int, ...]"

    def test_public_annotation_fixed_size(self, wrapped_class):
        """Test tuple[T1, T2] formatting."""
        codegens = [PlainTypeCodegen("int"), _make_wrapped_codegen(wrapped_class)]
        codegen = TupleTypeCodegen(codegens)
        assert codegen.public_annotation("test_module") == "tuple[int, TestClass]"

    def test_wrapper_to_impl_expr_variable_length(self, wrapped_class):
        item_codegen = _make_wrapped_codegen(wrapped_class)
        codegen = TupleTypeCodegen([item_codegen])
        result = codegen.wrapper_to_impl_expr("items")
        assert result == "tuple(x._impl_instance for x in items)"

    def test_wrapper_to_impl_expr_fixed_size(self, wrapped_class):
        """Test fixed-size tuple unwrapping by index."""
        codegens = [PlainTypeCodegen("int"), _make_wrapped_codegen(wrapped_class)]
        codegen = TupleTypeCodegen(codegens)
        result = codegen.wrapper_to_impl_expr("items")
        assert result == "(items[0], items[1]._impl_instance)"

    def test_impl_to_wrapper_expr_fixed_size(self, wrapped_class):
        codegens = [PlainTypeCodegen("int"), _make_wrapped_codegen(wrapped_class)]
        codegen = TupleTypeCodegen(codegens)
        result = codegen.impl_to_wrapper_expr("test_module", "impl_items")
        assert result == "(impl_items[0], TestClass._from_impl(impl_items[1]))"

    def test_execution_fixed_size_tuple(self):
        """Test that fixed-size tuple unwrap code executes correctly."""

        class MockWrapper:
            def __init__(self, value):
                self._impl_instance = value

        items = (42, MockWrapper(100))

        # Execute the unwrap expression
        wrapper_to_impl_expr = "(items[0], items[1]._impl_instance)"
        result = eval(wrapper_to_impl_expr, {"items": items})

        assert result == (42, 100)


class TestOptionalTypeCodegen:
    """Test OptionalTypeCodegen for Optional[T] types."""

    def test_public_annotation(self, wrapped_class):
        inner_codegen = _make_wrapped_codegen(wrapped_class)
        codegen = OptionalTypeCodegen(inner_codegen)
        assert codegen.public_annotation("test_module") == "typing.Union[TestClass, None]"

    def test_wrapper_to_impl_expr(self, wrapped_class):
        inner_codegen = _make_wrapped_codegen(wrapped_class)
        codegen = OptionalTypeCodegen(inner_codegen)
        result = codegen.wrapper_to_impl_expr("obj")
        assert result == "obj._impl_instance if obj is not None else None"

    def test_impl_to_wrapper_expr(self, wrapped_class):
        inner_codegen = _make_wrapped_codegen(wrapped_class)
        codegen = OptionalTypeCodegen(inner_codegen)
        result = codegen.impl_to_wrapper_expr("test_module", "impl")
        assert result == "TestClass._from_impl(impl) if impl is not None else None"

    def test_requires_boundary_translation(self, wrapped_class):
        # Optional primitive doesn't need translation
        codegen1 = OptionalTypeCodegen(PlainTypeCodegen("int"))
        assert codegen1.requires_boundary_translation() is False

        # Optional wrapped class needs translation
        codegen2 = OptionalTypeCodegen(_make_wrapped_codegen(wrapped_class))
        assert codegen2.requires_boundary_translation() is True

    def test_execution_unwrap_none(self):
        """Test unwrap with None value."""
        wrapper_to_impl_expr = "value._impl_instance if value is not None else None"
        result = eval(wrapper_to_impl_expr, {"value": None})
        assert result is None

    def test_execution_unwrap_not_none(self):
        """Test unwrap with non-None value."""

        class MockWrapper:
            def __init__(self, value):
                self._impl_instance = value

        obj = MockWrapper(42)
        wrapper_to_impl_expr = "value._impl_instance if value is not None else None"
        result = eval(wrapper_to_impl_expr, {"value": obj})
        assert result == 42


class TestUnionTypeCodegen:
    """Test UnionTypeCodegen for general Union[T1, T2] types."""

    def test_public_annotation(self, wrapped_class):
        codegen = UnionTypeCodegen([PlainTypeCodegen("int"), _make_wrapped_codegen(wrapped_class)])
        assert codegen.public_annotation("test_module") == "typing.Union[int, TestClass]"

    def test_requires_boundary_translation(self, wrapped_class):
        codegen = UnionTypeCodegen([PlainTypeCodegen("int"), _make_wrapped_codegen(wrapped_class)])
        assert codegen.requires_boundary_translation() is True

        identity_only = UnionTypeCodegen([PlainTypeCodegen("int"), PlainTypeCodegen("str")])
        assert identity_only.requires_boundary_translation() is False

    def test_impl_to_wrapper_expr_mixed_union(self, wrapped_class):
        codegen = UnionTypeCodegen([PlainTypeCodegen("int"), _make_wrapped_codegen(wrapped_class)])
        result = codegen.impl_to_wrapper_expr("test_module", "value")
        assert "TestClass._from_impl(_v)" in result
        assert "isinstance(_v, test.unit.codegens.test_type_codegens.TestUnionTypeCodegen" not in result
        assert "Unexpected value for union translation" in result

    def test_wrapper_to_impl_expr_mixed_union(self, wrapped_class):
        codegen = UnionTypeCodegen([PlainTypeCodegen("int"), _make_wrapped_codegen(wrapped_class)])
        result = codegen.wrapper_to_impl_expr("value")
        assert "isinstance(_v, test_module.TestClass)" in result
        assert "_v._impl_instance" in result
        assert 'getattr(_v, "_impl_instance")' not in result

    def test_execution_impl_to_wrapper_expr_returns_wrapper(self, wrapped_class):
        codegen = UnionTypeCodegen([PlainTypeCodegen("int"), _make_wrapped_codegen(wrapped_class)])

        class TestClass:
            @staticmethod
            def _from_impl(impl):
                return ("wrapped", impl.value)

        impl = wrapped_class(9)
        expr = codegen.impl_to_wrapper_expr("test_module", "value")
        module = sys.modules[wrapped_class.__module__]
        setattr(module, wrapped_class.__name__, wrapped_class)
        result = eval(expr, {"value": impl, "TestClass": TestClass, "test": __import__("test")})
        assert result == ("wrapped", 9)

    def test_execution_wrapper_to_impl_expr_returns_impl(self, wrapped_class):
        codegen = UnionTypeCodegen([PlainTypeCodegen("int"), _make_wrapped_codegen(wrapped_class)])

        class Wrapper:
            def __init__(self, impl):
                self._impl_instance = impl

        test_module = types.SimpleNamespace(TestClass=Wrapper)
        impl = wrapped_class(9)
        wrapper = Wrapper(impl)
        expr = codegen.wrapper_to_impl_expr("value")
        result = eval(expr, {"value": wrapper, "test_module": test_module})
        assert result is impl

    def test_rejects_ambiguous_translated_union(self, wrapped_class):
        codegen = UnionTypeCodegen(
            [
                ListTypeCodegen(_make_wrapped_codegen(wrapped_class)),
                ListTypeCodegen(PlainTypeCodegen("int")),
            ],
            source_label="pkg.mod.func return",
        )
        with pytest.raises(TypeError, match=r"pkg\.mod\.func return: .*list\[test_module\.TestClass\].*list\[int\]"):
            codegen.impl_to_wrapper_expr("test_module", "value")

    def test_allows_same_wrapped_generic_base(self, wrapped_class):
        impl_ref = ObjectReferenceIR(module=wrapped_class.__module__, qualname=wrapped_class.__qualname__)
        wrapper_ref = ObjectReferenceIR("test_module", "TestClass")
        inner = WrappedClassTypeCodegen(impl_ref, wrapper_ref)
        generic_str = ParameterizedWrappedClassTypeCodegen(inner, [PlainTypeCodegen("str")])
        generic_list_int = ParameterizedWrappedClassTypeCodegen(
            inner,
            [ListTypeCodegen(PlainTypeCodegen("int"))],
        )

        codegen = UnionTypeCodegen([generic_str, generic_list_int], source_label="pkg.mod.func return")

        impl_to_wrapper_expr = codegen.impl_to_wrapper_expr("test_module", "value")
        wrapper_to_impl_expr = codegen.wrapper_to_impl_expr("value")

        assert "TestClass._from_impl(_v)" in impl_to_wrapper_expr
        assert impl_to_wrapper_expr.count("TestClass._from_impl(_v)") == 1
        assert wrapper_to_impl_expr.count("isinstance(_v, test_module.TestClass)") == 1
        assert "_v._impl_instance" in wrapper_to_impl_expr


class TestGeneratorCodegen:
    """Test GeneratorCodegen for Generator/AsyncGenerator types."""

    def test_public_annotation_async_generator(self, wrapped_class):
        from synchronicity2.codegen.emission.type_codegen import AsyncGeneratorTypeCodegen

        yield_codegen = _make_wrapped_codegen(wrapped_class)
        codegen = AsyncGeneratorTypeCodegen(yield_codegen, PlainTypeCodegen("None"))
        assert codegen.public_annotation("test_module", is_async=True) == "typing.AsyncGenerator[TestClass, None]"
        # Sync context should preserve send type (even if it's just None)
        assert codegen.public_annotation("test_module", is_async=False) == "typing.Generator[TestClass, None, None]"

    def test_public_annotation_async_iterator_no_send(self):
        """AsyncIterator doesn't have send type."""
        from synchronicity2.codegen.emission.type_codegen import AsyncGeneratorTypeCodegen

        yield_codegen = PlainTypeCodegen("int")
        codegen = AsyncGeneratorTypeCodegen(yield_codegen, None)
        assert codegen.public_annotation("test_module", is_async=True) == "typing.AsyncGenerator[int, None]"
        assert codegen.public_annotation("test_module", is_async=False) == "typing.Generator[int, None, None]"

    def test_public_annotation_sync_generator(self):
        from synchronicity2.codegen.emission.type_codegen import SyncGeneratorTypeCodegen

        yield_codegen = PlainTypeCodegen("str")
        codegen = SyncGeneratorTypeCodegen(yield_codegen, PlainTypeCodegen("None"), PlainTypeCodegen("None"))
        assert codegen.public_annotation("test_module") == "typing.Generator[str, None, None]"

    def test_two_way_generator_with_send_type(self):
        """Test that two-way generators preserve send type in both contexts."""
        from synchronicity2.codegen.emission.type_codegen import AsyncGeneratorTypeCodegen

        yield_codegen = PlainTypeCodegen("str")
        # Two-way generator: yields str, accepts str via send
        codegen = AsyncGeneratorTypeCodegen(yield_codegen, PlainTypeCodegen("str"))

        # Async context: AsyncGenerator[str, str]
        assert codegen.public_annotation("test_module", is_async=True) == "typing.AsyncGenerator[str, str]"

        # Sync context: Generator[str, str, None]
        # Should preserve send type to support two-way generators
        assert codegen.public_annotation("test_module", is_async=False) == "typing.Generator[str, str, None]"

    def test_two_way_generator_with_int_send_type(self):
        """Test two-way generator with int send type."""
        from synchronicity2.codegen.emission.type_codegen import AsyncGeneratorTypeCodegen

        yield_codegen = PlainTypeCodegen("int")
        codegen = AsyncGeneratorTypeCodegen(yield_codegen, PlainTypeCodegen("int"))

        # Both contexts should preserve send type
        assert codegen.public_annotation("test_module", is_async=True) == "typing.AsyncGenerator[int, int]"
        assert codegen.public_annotation("test_module", is_async=False) == "typing.Generator[int, int, None]"

    def test_one_way_generator_send_none(self):
        """Test one-way generator (send type is None, no send support needed)."""
        from synchronicity2.codegen.emission.type_codegen import AsyncGeneratorTypeCodegen

        yield_codegen = PlainTypeCodegen("str")
        # One-way generator: yields str, doesn't use send
        codegen = AsyncGeneratorTypeCodegen(yield_codegen, PlainTypeCodegen("None"))

        # Both should have None as send type
        assert codegen.public_annotation("test_module", is_async=True) == "typing.AsyncGenerator[str, None]"
        assert codegen.public_annotation("test_module", is_async=False) == "typing.Generator[str, None, None]"

    def test_requires_boundary_translation(self, wrapped_class):
        from synchronicity2.codegen.emission.type_codegen import AsyncGeneratorTypeCodegen, SyncGeneratorTypeCodegen

        # Async generators ALWAYS need translation (for synchronizer integration)
        codegen1 = AsyncGeneratorTypeCodegen(PlainTypeCodegen("int"), PlainTypeCodegen("None"))
        assert codegen1.requires_boundary_translation() is True

        # Generator of wrapped classes also needs translation
        codegen2 = AsyncGeneratorTypeCodegen(_make_wrapped_codegen(wrapped_class), PlainTypeCodegen("None"))
        assert codegen2.requires_boundary_translation() is True

        # Sync generators only need translation if yield type needs translation
        codegen3 = SyncGeneratorTypeCodegen(PlainTypeCodegen("int"), PlainTypeCodegen("None"), PlainTypeCodegen("None"))
        assert codegen3.requires_boundary_translation() is False

        codegen4 = SyncGeneratorTypeCodegen(
            _make_wrapped_codegen(wrapped_class), PlainTypeCodegen("None"), PlainTypeCodegen("None")
        )
        assert codegen4.requires_boundary_translation() is True


class TestCodegenFromAnnotation:
    """``parse_annotation`` + ``codegen_for_annotation`` (codegen path)."""

    def test_create_primitive(self):
        codegen = _codegen_from_annotation(int)
        assert isinstance(codegen, PlainTypeCodegen)

    def test_create_wrapped_class(self, wrapped_class):
        codegen = _codegen_from_annotation(wrapped_class)
        assert isinstance(codegen, WrappedClassTypeCodegen)

    def test_create_list(self):
        from typing import List

        codegen = _codegen_from_annotation(List[int])
        assert isinstance(codegen, ListTypeCodegen)
        assert isinstance(codegen.item_codegen, PlainTypeCodegen)

    def test_create_dict(self):
        from typing import Dict

        codegen = _codegen_from_annotation(Dict[str, int])
        assert isinstance(codegen, DictTypeCodegen)

    def test_create_tuple_fixed(self, wrapped_class):
        from typing import Tuple

        codegen = _codegen_from_annotation(Tuple[int, wrapped_class])
        assert isinstance(codegen, TupleTypeCodegen)
        assert len(codegen.item_codegens) == 2

    def test_create_optional(self, wrapped_class):
        from typing import Optional

        codegen = _codegen_from_annotation(Optional[wrapped_class])
        assert isinstance(codegen, OptionalTypeCodegen)
        assert isinstance(codegen.inner_codegen, WrappedClassTypeCodegen)

    def test_create_union(self, wrapped_class):
        codegen = _codegen_from_annotation(int | wrapped_class)
        assert isinstance(codegen, UnionTypeCodegen)
        assert len(codegen.item_codegens) == 2

    def test_create_async_generator(self):
        from typing import AsyncGenerator

        codegen = _codegen_from_annotation(AsyncGenerator[str, None])
        assert isinstance(codegen, AsyncGeneratorTypeCodegen)

    def test_create_async_generator_parses_wrapped_send_type(self, wrapped_class):
        codegen = _codegen_from_annotation(typing.AsyncGenerator[str, wrapped_class])

        assert isinstance(codegen, AsyncGeneratorTypeCodegen)
        assert isinstance(codegen.send_codegen, WrappedClassTypeCodegen)
        assert codegen.public_annotation("test_module") == "typing.AsyncGenerator[str, TestClass]"
        assert codegen.implementation_annotation("test_module") == (
            f"typing.AsyncGenerator[str, {wrapped_class.__module__}.{wrapped_class.__name__}]"
        )

    def test_create_sync_generator_preserves_all_type_arguments(self, wrapped_class):
        codegen = _codegen_from_annotation(typing.Generator[str, wrapped_class, wrapped_class])

        assert isinstance(codegen, SyncGeneratorTypeCodegen)
        assert isinstance(codegen.send_codegen, WrappedClassTypeCodegen)
        assert isinstance(codegen.return_codegen, WrappedClassTypeCodegen)
        assert codegen.public_annotation("test_module") == "typing.Generator[str, TestClass, TestClass]"

    def test_create_sync_iterator_is_distinct_from_generator(self, wrapped_class):
        codegen = _codegen_from_annotation(typing.Iterator[wrapped_class])

        assert isinstance(codegen, SyncIteratorTypeCodegen)
        assert isinstance(codegen.item_codegen, WrappedClassTypeCodegen)
        assert codegen.public_annotation("test_module") == "typing.Iterator[TestClass]"

    def test_nested_list_of_optional_wrapped(self, wrapped_class):
        """Test nested type: list[Optional[WrappedClass]]."""
        from typing import List, Optional

        codegen = _codegen_from_annotation(List[Optional[wrapped_class]])
        assert isinstance(codegen, ListTypeCodegen)
        assert isinstance(codegen.item_codegen, OptionalTypeCodegen)
        assert isinstance(codegen.item_codegen.inner_codegen, WrappedClassTypeCodegen)

        # Check type signature
        result = codegen.public_annotation("test_module")
        assert result == "list[typing.Union[TestClass, None]]"

        # Check needs translation
        assert codegen.requires_boundary_translation() is True

    def test_create_coroutine_with_args(self):
        """Test Coroutine[Any, Any, str] creates CoroutineTypeCodegen."""
        from typing import Any, Coroutine

        codegen = _codegen_from_annotation(Coroutine[Any, Any, str])
        assert isinstance(codegen, CoroutineTypeCodegen)

        # Check that return type is unwrapped to str
        result = codegen.public_annotation("test_module")
        assert result == "str"

    def test_create_coroutine_bare(self):
        """Test bare Coroutine (no type args) creates CoroutineTypeCodegen with identity return."""
        from typing import Coroutine

        codegen = _codegen_from_annotation(Coroutine)
        assert isinstance(codegen, CoroutineTypeCodegen)
        assert isinstance(codegen.return_codegen, PlainTypeCodegen)

    def test_create_awaitable_with_args(self):
        """Test Awaitable[str] creates AwaitableTypeCodegen."""
        from typing import Awaitable

        codegen = _codegen_from_annotation(Awaitable[str])
        assert isinstance(codegen, AwaitableTypeCodegen)

        # Check that return type is unwrapped to str
        result = codegen.public_annotation("test_module")
        assert result == "str"

    def test_create_awaitable_bare(self):
        """Test bare Awaitable (no type args) creates AwaitableTypeCodegen with identity return."""
        from typing import Awaitable

        codegen = _codegen_from_annotation(Awaitable)
        assert isinstance(codegen, AwaitableTypeCodegen)
        assert isinstance(codegen.return_codegen, PlainTypeCodegen)


class TestComplexNestedTypes:
    """Test complex nested type transformations."""

    def test_dict_of_list_of_wrapped(self, wrapped_class):
        """Test dict[str, list[WrappedClass]]."""
        from typing import Dict, List

        codegen = _codegen_from_annotation(Dict[str, List[wrapped_class]])

        # Check type signature
        result = codegen.public_annotation("test_module")
        assert result == "dict[str, list[TestClass]]"

        # Check unwrap expression
        unwrap = codegen.wrapper_to_impl_expr("data")
        assert "[x._impl_instance for x in v]" in unwrap
        assert "for k, v in data.items()" in unwrap

        # Check needs translation
        assert codegen.requires_boundary_translation() is True

    def test_tuple_of_mixed_types(self, wrapped_class):
        """Test tuple[int, WrappedClass, str]."""
        from typing import Tuple

        codegen = _codegen_from_annotation(Tuple[int, wrapped_class, str])

        # Check type signature
        result = codegen.public_annotation("test_module")
        assert result == "tuple[int, TestClass, str]"

        # Check unwrap - should unwrap only the wrapped class at index 1
        unwrap = codegen.wrapper_to_impl_expr("items")
        assert "items[0]" in unwrap
        assert "items[1]._impl_instance" in unwrap
        assert "items[2]" in unwrap

    def test_execution_nested_unwrap(self):
        """Test executing nested unwrap code."""

        class MockWrapper:
            def __init__(self, value):
                self._impl_instance = value

        # Test dict[str, list[Wrapped]]
        data = {"a": [MockWrapper(1), MockWrapper(2)], "b": [MockWrapper(3)]}

        # Execute nested unwrap
        wrapper_to_impl_expr = "{k: [x._impl_instance for x in v] for k, v in data.items()}"
        result = eval(wrapper_to_impl_expr, {"data": data})

        assert result == {"a": [1, 2], "b": [3]}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
