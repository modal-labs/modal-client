# Copyright Modal Labs 2026
"""Unit tests for IR → emitted source (module-level functions).

IR literals below are explicit dataclasses (same shapes as the parse layer). ``IMPL`` is this
module so emitted implementation qualname references match assertions.
"""

from __future__ import annotations

import dataclasses

from synchronicity2.codegen.emission.module_codegen import emit_wrapped_function
from synchronicity2.codegen.ir.annotations import (
    AsyncGeneratorAnnotationIR,
    AwaitableAnnotationIR,
    CoroutineAnnotationIR,
    DictAnnotationIR,
    ListAnnotationIR,
    OptionalAnnotationIR,
    PlainAnnotationIR,
    WrappedClassRefIR,
)
from synchronicity2.codegen.ir.declarations import ParameterIR, SignatureIR, WrappedFunctionIR
from synchronicity2.codegen.ir.references import ObjectReferenceIR

IMPL = __name__
TARGET = "test_module"

# --- Module-level function IR (qualnames are synthetic; shapes match parse output.) ---

IR_FN_ASYNC_GEN = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_async_gen"),
    needs_async_wrapper=True,
    is_async_gen=True,
    parameters=(
        ParameterIR(
            name="items",
            kind=1,
            annotation_ir=ListAnnotationIR(item_ir=PlainAnnotationIR(signature_text="str")),
            default_expr=None,
        ),
    ),
    return_annotation_ir=AsyncGeneratorAnnotationIR(
        yield_annotation_ir=PlainAnnotationIR(signature_text="str"),
        send_annotation_ir=PlainAnnotationIR(signature_text="None"),
    ),
)
IR_FN_BARE_ITERATOR = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_declared_bare_iterator"),
    needs_async_wrapper=True,
    is_async_gen=True,
    parameters=(),
    return_annotation_ir=AsyncGeneratorAnnotationIR(
        yield_annotation_ir=PlainAnnotationIR(signature_text="typing.Any"),
        send_annotation_ir=PlainAnnotationIR(signature_text="None"),
    ),
)
IR_FN_COMPLEX_TYPES = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_complex_types"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(
        ParameterIR(
            name="items",
            kind=1,
            annotation_ir=ListAnnotationIR(item_ir=PlainAnnotationIR(signature_text="str")),
            default_expr=None,
        ),
        ParameterIR(
            name="config",
            kind=1,
            annotation_ir=DictAnnotationIR(
                key_ir=PlainAnnotationIR(signature_text="str"), value_ir=PlainAnnotationIR(signature_text="int")
            ),
            default_expr=None,
        ),
        ParameterIR(
            name="optional_param",
            kind=1,
            annotation_ir=OptionalAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="str")),
            default_expr="None",
        ),
    ),
    return_annotation_ir=AwaitableAnnotationIR(
        inner_ir=DictAnnotationIR(
            key_ir=PlainAnnotationIR(signature_text="str"),
            value_ir=ListAnnotationIR(item_ir=PlainAnnotationIR(signature_text="int")),
        )
    ),
)
IR_FN_CREATE_AWAITABLE = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_create_awaitable"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="x", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
    ),
    return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="str")),
)
IR_FN_CREATE_AWAITABLE_BARE = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_create_awaitable_bare"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="x", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
    ),
    return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="typing.Any")),
)
IR_FN_CREATE_COROUTINE = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_create_coroutine"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="x", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
    ),
    return_annotation_ir=CoroutineAnnotationIR(return_annotation_ir=PlainAnnotationIR(signature_text="str")),
)
IR_FN_CREATE_PEOPLE = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_create_people"),
    needs_async_wrapper=False,
    is_async_gen=False,
    parameters=(
        ParameterIR(
            name="names",
            kind=1,
            annotation_ir=ListAnnotationIR(item_ir=PlainAnnotationIR(signature_text="str")),
            default_expr=None,
        ),
    ),
    return_annotation_ir=ListAnnotationIR(
        item_ir=WrappedClassRefIR(impl=ObjectReferenceIR(IMPL, "Person"), wrapper=ObjectReferenceIR(TARGET, "Person"))
    ),
)
IR_FN_CREATE_PERSON = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_create_person"),
    needs_async_wrapper=False,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="name", kind=1, annotation_ir=PlainAnnotationIR(signature_text="str"), default_expr=None),
    ),
    return_annotation_ir=WrappedClassRefIR(
        impl=ObjectReferenceIR(IMPL, "Person"), wrapper=ObjectReferenceIR(TARGET, "Person")
    ),
)
IR_FN_KEYWORD_ONLY = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_keyword_only"),
    needs_async_wrapper=False,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="_sentinel", kind=1, annotation_ir=None, default_expr="None"),
        ParameterIR(
            name="required_kwonly",
            kind=3,
            annotation_ir=PlainAnnotationIR(signature_text="int"),
            default_expr=None,
        ),
        ParameterIR(
            name="optional_kwonly",
            kind=3,
            annotation_ir=PlainAnnotationIR(signature_text="str"),
            default_expr="'x'",
        ),
    ),
    return_annotation_ir=PlainAnnotationIR(signature_text="None"),
)
IR_FN_GENERIC_TYPES = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_generic_types"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(
        ParameterIR(
            name="items",
            kind=1,
            annotation_ir=ListAnnotationIR(item_ir=PlainAnnotationIR(signature_text="str")),
            default_expr=None,
        ),
        ParameterIR(
            name="mapping",
            kind=1,
            annotation_ir=DictAnnotationIR(
                key_ir=PlainAnnotationIR(signature_text="str"), value_ir=PlainAnnotationIR(signature_text="int")
            ),
            default_expr=None,
        ),
        ParameterIR(
            name="optional_set",
            kind=1,
            annotation_ir=PlainAnnotationIR(signature_text="typing.UnionType[set[int], None]"),
            default_expr="None",
        ),
    ),
    return_annotation_ir=AwaitableAnnotationIR(
        inner_ir=ListAnnotationIR(
            item_ir=DictAnnotationIR(
                key_ir=PlainAnnotationIR(signature_text="str"), value_ir=PlainAnnotationIR(signature_text="int")
            )
        )
    ),
)
IR_FN_GREET = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_greet"),
    needs_async_wrapper=False,
    is_async_gen=False,
    parameters=(
        ParameterIR(
            name="person",
            kind=1,
            annotation_ir=WrappedClassRefIR(
                impl=ObjectReferenceIR(IMPL, "Person"), wrapper=ObjectReferenceIR(TARGET, "Person")
            ),
            default_expr=None,
        ),
    ),
    return_annotation_ir=PlainAnnotationIR(signature_text="str"),
)
IR_FN_NO_ANNOTATION = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_no_annotation"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="x", kind=1, annotation_ir=None, default_expr=None),
        ParameterIR(name="y", kind=1, annotation_ir=None, default_expr="42"),
    ),
    return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="typing.Any")),
)
IR_FN_NO_TYPES = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_no_types"),
    needs_async_wrapper=False,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="x", kind=1, annotation_ir=None, default_expr=None),
        ParameterIR(name="y", kind=1, annotation_ir=None, default_expr=None),
    ),
    return_annotation_ir=PlainAnnotationIR(signature_text=""),
)
IR_FN_POSONLY = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_with_posonly"),
    needs_async_wrapper=False,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="a", kind=0, annotation_ir=None, default_expr=None),
        ParameterIR(name="b", kind=0, annotation_ir=None, default_expr=None),
        ParameterIR(name="c", kind=1, annotation_ir=None, default_expr=None),
        ParameterIR(name="d", kind=1, annotation_ir=None, default_expr="10"),
    ),
    return_annotation_ir=PlainAnnotationIR(signature_text="int"),
)
IR_FN_SIMPLE_TYPES = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_simple_types"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="x", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
    ),
    return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="str")),
)
IR_FN_STREAM_BATCHES = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_stream_person_batches"),
    needs_async_wrapper=True,
    is_async_gen=True,
    parameters=(
        ParameterIR(
            name="batch_size", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None
        ),
    ),
    return_annotation_ir=AsyncGeneratorAnnotationIR(
        yield_annotation_ir=ListAnnotationIR(
            item_ir=WrappedClassRefIR(
                impl=ObjectReferenceIR(IMPL, "Person"), wrapper=ObjectReferenceIR(TARGET, "Person")
            )
        ),
        send_annotation_ir=None,
    ),
)
IR_FN_STREAM_PEOPLE = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_stream_people"),
    needs_async_wrapper=True,
    is_async_gen=True,
    parameters=(
        ParameterIR(name="count", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
    ),
    return_annotation_ir=AsyncGeneratorAnnotationIR(
        yield_annotation_ir=WrappedClassRefIR(
            impl=ObjectReferenceIR(IMPL, "Person"), wrapper=ObjectReferenceIR(TARGET, "Person")
        ),
        send_annotation_ir=None,
    ),
)
IR_FN_SYNC_ADD = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_sync_add"),
    needs_async_wrapper=False,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="a", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
        ParameterIR(name="b", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
    ),
    return_annotation_ir=PlainAnnotationIR(signature_text="int"),
)
IR_FN_VARARGS = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_with_varargs"),
    needs_async_wrapper=False,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="posonly", kind=1, annotation_ir=None, default_expr=None),
        ParameterIR(name="a", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
        ParameterIR(name="b", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr="10"),
        ParameterIR(name="extra", kind=2, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
        ParameterIR(name="c", kind=3, annotation_ir=None, default_expr=None),
        ParameterIR(name="extrakwargs", kind=4, annotation_ir=None, default_expr=None),
    ),
    return_annotation_ir=PlainAnnotationIR(signature_text="str"),
)
IR_FN_WITH_DEFAULTS = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_with_defaults"),
    needs_async_wrapper=False,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="a", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
        ParameterIR(name="b", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr="10"),
        ParameterIR(name="c", kind=1, annotation_ir=PlainAnnotationIR(signature_text="str"), default_expr="'hello'"),
    ),
    return_annotation_ir=PlainAnnotationIR(signature_text="str"),
)
IR_FN_WITH_MANY_DEFAULTS = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_with_many_defaults"),
    needs_async_wrapper=False,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="required", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
        ParameterIR(name="text", kind=1, annotation_ir=PlainAnnotationIR(signature_text="str"), default_expr="'hello'"),
        ParameterIR(
            name="enabled", kind=1, annotation_ir=PlainAnnotationIR(signature_text="bool"), default_expr="True"
        ),
        ParameterIR(
            name="payload", kind=1, annotation_ir=PlainAnnotationIR(signature_text="bytes"), default_expr="b'data'"
        ),
        ParameterIR(
            name="coords",
            kind=1,
            annotation_ir=PlainAnnotationIR(signature_text="tuple[int, int]"),
            default_expr="(1, 2)",
        ),
        ParameterIR(
            name="tags",
            kind=1,
            annotation_ir=PlainAnnotationIR(signature_text="list[str]"),
            default_expr="['a', 'b']",
        ),
        ParameterIR(
            name="mapping",
            kind=1,
            annotation_ir=PlainAnnotationIR(signature_text="dict[str, int]"),
            default_expr="{'a': 1}",
        ),
        ParameterIR(
            name="items",
            kind=1,
            annotation_ir=PlainAnnotationIR(signature_text="set[int]"),
            default_expr="{1, 2}",
        ),
        ParameterIR(
            name="frozen",
            kind=1,
            annotation_ir=PlainAnnotationIR(signature_text="frozenset[int]"),
            default_expr="frozenset({1, 2})",
        ),
        ParameterIR(
            name="window",
            kind=1,
            annotation_ir=PlainAnnotationIR(signature_text="slice"),
            default_expr="slice(1, 2, 3)",
        ),
        ParameterIR(
            name="optional",
            kind=3,
            annotation_ir=PlainAnnotationIR(signature_text="typing.Any"),
            default_expr="None",
        ),
    ),
    return_annotation_ir=PlainAnnotationIR(signature_text="str"),
)
IR_FN_OVERLOADS_WITH_TRANSLATION = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_overloaded"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(ParameterIR(name="value", kind=1, annotation_ir=None, default_expr=None),),
    return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="typing.Any")),
    overloads=(
        SignatureIR(
            parameters=(
                ParameterIR(
                    name="value",
                    kind=1,
                    annotation_ir=PlainAnnotationIR(signature_text="int"),
                    default_expr=None,
                ),
            ),
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="int")),
        ),
        SignatureIR(
            parameters=(
                ParameterIR(
                    name="value",
                    kind=1,
                    annotation_ir=WrappedClassRefIR(
                        impl=ObjectReferenceIR(IMPL, "Person"),
                        wrapper=ObjectReferenceIR(TARGET, "Person"),
                    ),
                    default_expr=None,
                ),
            ),
            return_annotation_ir=AwaitableAnnotationIR(
                inner_ir=WrappedClassRefIR(
                    impl=ObjectReferenceIR(IMPL, "Person"),
                    wrapper=ObjectReferenceIR(TARGET, "Person"),
                )
            ),
        ),
    ),
)


def _fn_short(ir: WrappedFunctionIR) -> str:
    return ir.impl_ref.qualname.rpartition(".")[2]


def test_emit_async_function_basic_template():
    ir = IR_FN_SIMPLE_TYPES
    code = emit_wrapped_function(ir, TARGET)
    compile(code, "<string>", "exec")
    name = _fn_short(ir)
    assert f"class _{name}_FunctionWithAio(FunctionWithAio):" in code
    assert f"_run_function_async({IMPL}.{name}(x))" in code
    assert f"@function_with_aio(_{name}_FunctionWithAio)" in code
    assert f"def {name}" in code
    assert "async def aio(self, x: int) -> str:" in code
    assert "def __init__(self, sync_impl:" not in code
    assert "_run_function_sync" in code
    assert "_run_function_async" in code
    assert "x: int" in code
    assert "-> str" in code


def test_emit_async_function_complex_types():
    code = emit_wrapped_function(IR_FN_COMPLEX_TYPES, TARGET)
    compile(code, "<string>", "exec")
    assert "items: list" in code
    assert "config: dict" in code
    assert (
        "optional_param: typing.Union[str, None]" in code
        or "optional_param: str | None" in code
        or "optional_param: typing.Optional" in code
    )
    assert "-> dict" in code
    assert "= None" in code


def test_emit_async_function_no_annotations():
    ir = IR_FN_NO_ANNOTATION
    code = emit_wrapped_function(ir, TARGET)
    compile(code, "<string>", "exec")
    n = _fn_short(ir)
    assert f"class _{n}_FunctionWithAio(FunctionWithAio):" in code
    assert "async def aio(self, x, y = 42) -> typing.Any:" in code
    assert f"def {n}" in code


def test_emit_async_function_template_line_order():
    ir = IR_FN_SIMPLE_TYPES
    code = emit_wrapped_function(ir, TARGET)
    lines = code.split("\n")
    async_line = next(i for i, line in enumerate(lines) if "async def aio(" in line)
    assert async_line is not None


def test_emit_async_function_docstring_skips_with_aio_class_level_copy():
    ir = dataclasses.replace(IR_FN_CREATE_AWAITABLE, docstring="Awaitable docstring.")
    code = emit_wrapped_function(ir, TARGET)
    assert code.count('"""Awaitable docstring."""') == 3
    assert ('class _fn_create_awaitable_FunctionWithAio(FunctionWithAio):\n    """Awaitable docstring."""') not in code


def test_emit_async_function_generic_types():
    code = emit_wrapped_function(IR_FN_GENERIC_TYPES, TARGET)
    compile(code, "<string>", "exec")
    assert "items: list[str]" in code
    assert "mapping: dict[str, int]" in code
    assert "optional_set:" in code and "set[int]" in code
    assert "-> list[dict[str, int]]" in code
    assert "= None" in code
    assert "async def aio(" in code
    assert "@function_with_aio" in code
    assert f"{IMPL}.fn_generic_types(" in code


def test_emit_async_generator_function():
    """When yield type needs no translation, helpers are skipped and we delegate directly."""
    code = emit_wrapped_function(IR_FN_ASYNC_GEN, TARGET)
    compile(code, "<string>", "exec")
    assert "_run_generator_sync" in code
    assert "_run_generator_async" in code
    assert "_run_function_sync" not in code
    # No helper functions needed when yield type doesn't need translation
    assert "_wrap_async_gen" not in code
    # Direct delegation to synchronizer
    assert "_run_generator_async(gen)" in code
    assert "yield from _synchronizer._run_generator_sync(gen)" in code
    assert "_sent = yield _item" in code
    assert f"gen = {IMPL}.fn_async_gen(" in code
    assert "await _wrapped.asend(_sent)" in code
    assert "typing.Generator[str" in code
    assert "typing.AsyncGenerator[str" in code
    assert "items: list" in code


def test_emit_async_generator_template_pattern():
    ir = IR_FN_ASYNC_GEN
    code = emit_wrapped_function(ir, TARGET)
    name = _fn_short(ir)
    assert f"class _{name}_FunctionWithAio(FunctionWithAio):" in code
    assert f"@function_with_aio(_{name}_FunctionWithAio)" in code
    assert "async def aio(self, items: list[str]) -> typing.AsyncGenerator[str, None]:" in code
    assert f"def {name}" in code
    assert f"gen = {IMPL}.fn_async_gen(" in code


def test_emit_function_overloads_translate_each_overload():
    code = emit_wrapped_function(IR_FN_OVERLOADS_WITH_TRANSLATION, TARGET)
    compile(code, "<string>", "exec")
    assert "class _fn_overloaded_FunctionWithAio(FunctionWithAio):" in code
    assert "def __call__(self, value: int) -> int: ..." in code
    assert 'def __call__(self, value: "Person") -> "Person": ...' in code
    assert "async def aio(self, value: int) -> int: ..." in code
    assert 'async def aio(self, value: "Person") -> "Person": ...' in code
    assert "def __call__(self, value) -> typing.Any:" in code
    assert "return self._sync_impl(value)" in code
    assert "async def aio(self, value) -> typing.Any:" in code
    assert "@function_with_aio(_fn_overloaded_FunctionWithAio)" in code
    assert "_run_function_async(test.unit.emission.test_functions.fn_overloaded(value))" in code
    assert "def fn_overloaded(value) -> typing.Any:" in code


def test_emit_async_generator_wrapped_yield_type_quoting():
    code = emit_wrapped_function(IR_FN_STREAM_PEOPLE, TARGET)
    compile(code, "<string>", "exec")
    assert ' -> "typing.Generator[Person, None, None]"' in code
    assert ' -> "typing.AsyncGenerator[Person, None]"' in code
    assert 'Generator["Person"' not in code
    assert 'AsyncGenerator["Person"' not in code


def test_emit_async_generator_nested_wrapped_yield_quoting():
    code = emit_wrapped_function(IR_FN_STREAM_BATCHES, TARGET)
    compile(code, "<string>", "exec")
    assert ' -> "typing.Generator[list[Person], None, None]"' in code
    assert ' -> "typing.AsyncGenerator[list[Person], None]"' in code


def test_emit_declared_bare_iterator():
    code = emit_wrapped_function(IR_FN_BARE_ITERATOR, TARGET)
    assert "class _fn_declared_bare_iterator_FunctionWithAio(FunctionWithAio):" in code
    assert "async def aio(self) -> typing.AsyncGenerator[typing.Any, None]:" in code
    assert "@function_with_aio" in code
    assert "def fn_declared_bare_iterator() -> typing.Generator[typing.Any, None, None]" in code


def test_emit_sync_function_basic():
    code = emit_wrapped_function(IR_FN_SYNC_ADD, TARGET)
    compile(code, "<string>", "exec")
    assert "await impl_function" not in code
    assert "_run_function_sync" not in code
    assert "_run_function_async" not in code
    assert "return test.unit.emission.test_functions.fn_sync_add(a, b)" in code
    assert "class _simple_add" not in code
    assert "@wrapped_function" not in code
    assert "async def aio" not in code


def test_emit_sync_function_wrapped_arg():
    code = emit_wrapped_function(IR_FN_GREET, TARGET)
    compile(code, "<string>", "exec")
    assert "person_impl = person._impl_instance" in code
    assert "return test.unit.emission.test_functions.fn_greet(person_impl)" in code
    assert "_run_function_sync" not in code
    assert "class _greet" not in code
    assert "@wrapped_function" not in code


def test_emit_sync_function_wrapped_return():
    code = emit_wrapped_function(IR_FN_CREATE_PERSON, TARGET)
    compile(code, "<string>", "exec")
    assert "result = test.unit.emission.test_functions.fn_create_person(name)" in code
    assert "return Person._from_impl(result)" in code
    assert "_run_function_sync" not in code
    assert "class _create_person" not in code


def test_emit_sync_function_list_wrapped_return():
    code = emit_wrapped_function(IR_FN_CREATE_PEOPLE, TARGET)
    compile(code, "<string>", "exec")
    assert "result = test.unit.emission.test_functions.fn_create_people(names)" in code
    assert "[Person._from_impl(x) for x in result]" in code
    assert "_run_function_sync" not in code
    assert "class _create_people" not in code


def test_emit_sync_function_no_annotations():
    code = emit_wrapped_function(IR_FN_NO_TYPES, TARGET)
    compile(code, "<string>", "exec")
    assert "return test.unit.emission.test_functions.fn_no_types(x, y)" in code
    assert "_run_function_sync" not in code


def test_emit_sync_function_default_args():
    code = emit_wrapped_function(IR_FN_WITH_DEFAULTS, TARGET)
    compile(code, "<string>", "exec")
    assert "b: int = 10" in code
    assert "c: str = 'hello'" in code
    assert "return test.unit.emission.test_functions.fn_with_defaults(a, b, c)" in code


def test_emit_sync_function_various_builtin_default_values():
    code = emit_wrapped_function(IR_FN_WITH_MANY_DEFAULTS, TARGET)
    compile(code, "<string>", "exec")
    assert "text: str = 'hello'" in code
    assert "enabled: bool = True" in code
    assert "payload: bytes = b'data'" in code
    assert "coords: tuple[int, int] = (1, 2)" in code
    assert "tags: list[str] = ['a', 'b']" in code
    assert "mapping: dict[str, int] = {'a': 1}" in code
    assert "items: set[int] = {1, 2}" in code
    assert "frozen: frozenset[int] = frozenset({1, 2})" in code
    assert "window: slice = slice(1, 2, 3)" in code
    assert "optional: typing.Any = None" in code


def test_emit_function_varargs():
    code = emit_wrapped_function(IR_FN_VARARGS, TARGET)
    assert "def fn_with_varargs(posonly, a: int, b: int = 10, *extra: int, c, **extrakwargs)" in code
    assert "return test.unit.emission.test_functions.fn_with_varargs(posonly, a, b, *extra, c=c, **extrakwargs)" in code
    compile(code, "<string>", "exec")


def test_emit_function_positional_only():
    code = emit_wrapped_function(IR_FN_POSONLY, TARGET)
    assert "def fn_with_posonly(a, b, /, c, d = 10)" in code
    assert "return test.unit.emission.test_functions.fn_with_posonly(a, b, c, d)" in code
    compile(code, "<string>", "exec")


def test_emit_function_keyword_only_separator():
    code = emit_wrapped_function(IR_FN_KEYWORD_ONLY, TARGET)
    assert "def fn_keyword_only(_sentinel = None, *, required_kwonly: int, optional_kwonly: str = 'x')" in code
    assert (
        "return test.unit.emission.test_functions.fn_keyword_only("
        "_sentinel, required_kwonly=required_kwonly, optional_kwonly=optional_kwonly)" in code
    )
    compile(code, "<string>", "exec")


def test_emit_sync_function_returning_coroutine():
    code = emit_wrapped_function(IR_FN_CREATE_COROUTINE, TARGET)
    assert "@function_with_aio(_fn_create_coroutine_FunctionWithAio)" in code
    assert "class _fn_create_coroutine_FunctionWithAio(FunctionWithAio):" in code
    assert "async def aio(self, x: int) -> str:" in code
    assert "def fn_create_coroutine(x: int) -> str" in code
    assert "_run_function_sync" in code
    assert "_run_function_async" in code


def test_emit_sync_function_returning_awaitable():
    code = emit_wrapped_function(IR_FN_CREATE_AWAITABLE, TARGET)
    assert "@function_with_aio(_fn_create_awaitable_FunctionWithAio)" in code
    assert "class _fn_create_awaitable_FunctionWithAio(FunctionWithAio):" in code
    assert "async def aio(self, x: int) -> str:" in code
    assert "def fn_create_awaitable(x: int) -> str" in code
    assert "_run_function_sync" in code
    assert "_run_function_async" in code


def test_emit_sync_function_returning_bare_awaitable():
    code = emit_wrapped_function(IR_FN_CREATE_AWAITABLE_BARE, TARGET)
    assert "@function_with_aio(_fn_create_awaitable_bare_FunctionWithAio)" in code
    assert "class _fn_create_awaitable_bare_FunctionWithAio(FunctionWithAio):" in code
    assert "async def aio(self, x: int) -> typing.Any:" in code
    assert "def fn_create_awaitable_bare(x: int) -> typing.Any" in code
    assert "_run_function_sync" in code
    assert "_run_function_async" in code
