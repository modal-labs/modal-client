# Copyright Modal Labs 2026
"""IR → code tests for unwrap/wrap and translation-shaped emission.

IR literals live in this module. ``IMPL`` is this module name for emitted references.
"""

from __future__ import annotations

from synchronicity2.codegen.emission.module_codegen import emit_wrapped_class, emit_wrapped_function
from synchronicity2.codegen.ir.annotations import (
    AsyncGeneratorAnnotationIR,
    AwaitableAnnotationIR,
    ListAnnotationIR,
    OptionalAnnotationIR,
    PlainAnnotationIR,
    TupleAnnotationIR,
    WrappedClassRefIR,
)
from synchronicity2.codegen.ir.declarations import (
    MethodBindingKind,
    ParameterIR,
    WrappedClassIR,
    WrappedFunctionIR,
    WrappedMethodIR,
)
from synchronicity2.codegen.ir.references import ObjectReferenceIR

IMPL = __name__
TARGET = "test_module"

IR_CLASS_HELPER = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "HelperTestClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "HelperTestClass"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="__init__",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(
                    name="value", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None
                ),
            ),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=PlainAnnotationIR(signature_text=""),
        ),
    ),
)
IR_CLASS_HELPER_SUBCLASS = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "HelperTestSubclass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "HelperTestSubclass"),
    wrapped_bases=((ObjectReferenceIR(IMPL, "HelperTestClass"), ObjectReferenceIR(TARGET, "HelperTestClass")),),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="__init__",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(
                    name="value", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None
                ),
            ),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=PlainAnnotationIR(signature_text=""),
        ),
    ),
)
IR_FN_NODE_GENERATOR = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_node_generator"),
    needs_async_wrapper=True,
    is_async_gen=True,
    parameters=(),
    return_annotation_ir=AsyncGeneratorAnnotationIR(
        yield_annotation_ir=WrappedClassRefIR(
            impl=ObjectReferenceIR(IMPL, "GenNode"), wrapper=ObjectReferenceIR(TARGET, "GenNode")
        ),
        send_annotation_ir=PlainAnnotationIR(signature_text="None"),
    ),
)
IR_FN_RETURNS_STRING = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_returns_string"),
    needs_async_wrapper=False,
    is_async_gen=False,
    parameters=(),
    return_annotation_ir=PlainAnnotationIR(signature_text="str"),
)
IR_FN_SIMPLE_GEN = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_simple_gen"),
    needs_async_wrapper=True,
    is_async_gen=True,
    parameters=(),
    return_annotation_ir=AsyncGeneratorAnnotationIR(
        yield_annotation_ir=PlainAnnotationIR(signature_text="str"),
        send_annotation_ir=PlainAnnotationIR(signature_text="None"),
    ),
)
IR_FN_TUPLE_GENERATORS = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "fn_tuple_generators"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(),
    return_annotation_ir=AwaitableAnnotationIR(
        inner_ir=TupleAnnotationIR(
            element_irs=(
                AsyncGeneratorAnnotationIR(
                    yield_annotation_ir=PlainAnnotationIR(signature_text="str"),
                    send_annotation_ir=PlainAnnotationIR(signature_text="None"),
                ),
                AsyncGeneratorAnnotationIR(
                    yield_annotation_ir=PlainAnnotationIR(signature_text="int"),
                    send_annotation_ir=PlainAnnotationIR(signature_text="None"),
                ),
            ),
            variadic=False,
        )
    ),
)
IR_TR_CONNECT_NODES = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "tr_connect_nodes"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(
        ParameterIR(
            name="parent",
            kind=1,
            annotation_ir=WrappedClassRefIR(
                impl=ObjectReferenceIR(IMPL, "TestNode"), wrapper=ObjectReferenceIR(TARGET, "TestNode")
            ),
            default_expr=None,
        ),
        ParameterIR(
            name="child",
            kind=1,
            annotation_ir=WrappedClassRefIR(
                impl=ObjectReferenceIR(IMPL, "TestNode"), wrapper=ObjectReferenceIR(TARGET, "TestNode")
            ),
            default_expr=None,
        ),
    ),
    return_annotation_ir=AwaitableAnnotationIR(
        inner_ir=TupleAnnotationIR(
            element_irs=(
                WrappedClassRefIR(
                    impl=ObjectReferenceIR(IMPL, "TestNode"), wrapper=ObjectReferenceIR(TARGET, "TestNode")
                ),
                WrappedClassRefIR(
                    impl=ObjectReferenceIR(IMPL, "TestNode"), wrapper=ObjectReferenceIR(TARGET, "TestNode")
                ),
            ),
            variadic=False,
        )
    ),
)
IR_TR_CREATE_NODE = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "tr_create_node"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(
        ParameterIR(name="value", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
    ),
    return_annotation_ir=AwaitableAnnotationIR(
        inner_ir=WrappedClassRefIR(
            impl=ObjectReferenceIR(IMPL, "TestNode"), wrapper=ObjectReferenceIR(TARGET, "TestNode")
        )
    ),
)
IR_TR_GET_NODE_LIST = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "tr_get_node_list"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(
        ParameterIR(
            name="nodes",
            kind=1,
            annotation_ir=ListAnnotationIR(
                item_ir=WrappedClassRefIR(
                    impl=ObjectReferenceIR(IMPL, "TestNode"), wrapper=ObjectReferenceIR(TARGET, "TestNode")
                )
            ),
            default_expr=None,
        ),
    ),
    return_annotation_ir=AwaitableAnnotationIR(
        inner_ir=ListAnnotationIR(
            item_ir=WrappedClassRefIR(
                impl=ObjectReferenceIR(IMPL, "TestNode"), wrapper=ObjectReferenceIR(TARGET, "TestNode")
            )
        )
    ),
)
IR_TR_GET_OPTIONAL_NODE = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "tr_get_optional_node"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(
        ParameterIR(
            name="node",
            kind=1,
            annotation_ir=OptionalAnnotationIR(
                inner_ir=WrappedClassRefIR(
                    impl=ObjectReferenceIR(IMPL, "TestNode"), wrapper=ObjectReferenceIR(TARGET, "TestNode")
                )
            ),
            default_expr=None,
        ),
    ),
    return_annotation_ir=AwaitableAnnotationIR(
        inner_ir=OptionalAnnotationIR(
            inner_ir=WrappedClassRefIR(
                impl=ObjectReferenceIR(IMPL, "TestNode"), wrapper=ObjectReferenceIR(TARGET, "TestNode")
            )
        )
    ),
)
IR_TR_PROCESS_LIST = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "tr_process_list"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(
        ParameterIR(
            name="nodes",
            kind=1,
            annotation_ir=ListAnnotationIR(
                item_ir=WrappedClassRefIR(
                    impl=ObjectReferenceIR(IMPL, "CollectionTestNode"),
                    wrapper=ObjectReferenceIR(TARGET, "CollectionTestNode"),
                )
            ),
            default_expr=None,
        ),
    ),
    return_annotation_ir=AwaitableAnnotationIR(
        inner_ir=ListAnnotationIR(
            item_ir=WrappedClassRefIR(
                impl=ObjectReferenceIR(IMPL, "CollectionTestNode"),
                wrapper=ObjectReferenceIR(TARGET, "CollectionTestNode"),
            )
        )
    ),
)
IR_TR_PROCESS_NODE = WrappedFunctionIR(
    impl_ref=ObjectReferenceIR(IMPL, "tr_process_node"),
    needs_async_wrapper=True,
    is_async_gen=False,
    parameters=(
        ParameterIR(
            name="node",
            kind=1,
            annotation_ir=WrappedClassRefIR(
                impl=ObjectReferenceIR(IMPL, "UnwrapTestNode"), wrapper=ObjectReferenceIR(TARGET, "UnwrapTestNode")
            ),
            default_expr=None,
        ),
    ),
    return_annotation_ir=AwaitableAnnotationIR(
        inner_ir=WrappedClassRefIR(
            impl=ObjectReferenceIR(IMPL, "UnwrapTestNode"), wrapper=ObjectReferenceIR(TARGET, "UnwrapTestNode")
        )
    ),
)
IR_TR_TESTNODE_CLASS = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "TestNode"),
    wrapper_ref=ObjectReferenceIR(TARGET, "TestNode"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="__init__",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(
                    name="value", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None
                ),
            ),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=PlainAnnotationIR(signature_text=""),
        ),
        WrappedMethodIR(
            method_name="create_child",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(
                    name="child_value", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None
                ),
            ),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="'TestNode'")),
        ),
    ),
)


def test_emit_translation_function_and_class_signatures():
    create_node_code = emit_wrapped_function(IR_TR_CREATE_NODE, TARGET)
    connect_nodes_code = emit_wrapped_function(IR_TR_CONNECT_NODES, TARGET)
    get_node_list_code = emit_wrapped_function(IR_TR_GET_NODE_LIST, TARGET)
    get_optional_node_code = emit_wrapped_function(IR_TR_GET_OPTIONAL_NODE, TARGET)
    class_code = emit_wrapped_class(IR_TR_TESTNODE_CLASS, TARGET)

    assert "_instance_cache: weakref.WeakValueDictionary" in class_code
    assert "def _from_impl(cls, impl_instance: typing.Any)" in class_code
    assert "_wrapped_from_impl(cls, impl_instance, cls._instance_cache, _synchronizer)" in class_code

    assert (
        'def tr_create_node(value: int) -> "TestNode":' in create_node_code
        or "def tr_create_node(value: int) -> 'TestNode':" in create_node_code
    )
    assert 'def tr_get_node_list(nodes: "list[TestNode]") -> "list[TestNode]":' in get_node_list_code
    assert (
        'def tr_get_optional_node(node: "typing.Union[TestNode, None]") '
        '-> "typing.Union[TestNode, None]":' in get_optional_node_code
    )
    assert (
        'def tr_connect_nodes(parent: "TestNode", child: "TestNode") -> "tuple[TestNode, TestNode]":'
        in connect_nodes_code
        or "def tr_connect_nodes(parent: 'TestNode', child: 'TestNode') -> 'tuple[TestNode, TestNode]':"
        in connect_nodes_code
    )

    assert "parent_impl = parent._impl_instance" in connect_nodes_code
    assert "child_impl = child._impl_instance" in connect_nodes_code
    assert "TestNode._from_impl(" in connect_nodes_code


def test_emit_wrapper_helpers():
    compiled_code = emit_wrapped_class(IR_CLASS_HELPER, TARGET)
    assert "_instance_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()" in compiled_code
    assert "def _from_impl(cls, impl_instance: typing.Any)" in compiled_code
    assert "_wrapped_from_impl(cls, impl_instance, cls._instance_cache, _synchronizer)" in compiled_code


def test_emit_subclass_wrapper_helpers_define_own_cache():
    compiled_code = emit_wrapped_class(IR_CLASS_HELPER_SUBCLASS, TARGET)
    assert "class HelperTestSubclass(HelperTestClass):" in compiled_code
    assert "def _from_impl(cls, impl_instance: typing.Any)" in compiled_code
    assert '-> "HelperTestSubclass":' in compiled_code or "-> 'HelperTestSubclass':" in compiled_code
    assert "_wrapped_from_impl(cls, impl_instance, cls._instance_cache, _synchronizer)" in compiled_code
    assert "WeakValueDictionary = weakref.WeakValueDictionary()" in compiled_code


def test_emit_unwrap_in_function_bodies():
    compiled_code = emit_wrapped_function(IR_TR_PROCESS_NODE, TARGET)
    assert "node_impl = node._impl_instance" in compiled_code
    assert "return UnwrapTestNode._from_impl(result)" in compiled_code


def test_emit_collection_translation():
    compiled_code = emit_wrapped_function(IR_TR_PROCESS_LIST, TARGET)
    assert "[x._impl_instance for x in nodes]" in compiled_code
    assert "[CollectionTestNode._from_impl(x) for x in " in compiled_code


def test_emit_primitives_no_impl_suffix():
    compiled_code = emit_wrapped_function(IR_FN_RETURNS_STRING, TARGET)
    assert "str._from_impl" not in compiled_code


def test_emit_async_generator_wrapping_helpers():
    """When yield type needs no translation, helpers are skipped; direct synchronizer delegation."""
    compiled_code = emit_wrapped_function(IR_FN_SIMPLE_GEN, TARGET)
    # No helper functions needed for identity yield types
    assert "_wrap_async_gen" not in compiled_code
    # Direct delegation to synchronizer
    assert "_synchronizer._run_generator_async(gen)" in compiled_code
    assert "yield from _synchronizer._run_generator_sync(gen)" in compiled_code
    assert "await _wrapped.asend(_sent)" in compiled_code
    assert "await _wrapped.aclose()" in compiled_code
    assert "async def aio(self)" in compiled_code
    assert "-> typing.Generator[str, None, None]:" in compiled_code
    assert "-> typing.AsyncGenerator[str, None]:" in compiled_code


def test_emit_tuple_of_generators():
    """Tuple of generators with identity yield types uses direct synchronizer delegation."""
    compiled_code = emit_wrapped_function(IR_FN_TUPLE_GENERATORS, TARGET)
    # No helper functions needed for identity yield types
    assert "_wrap_async_gen" not in compiled_code
    # Direct delegation in tuple construction
    assert "_synchronizer._run_generator_async(result[0])" in compiled_code
    assert "_synchronizer._run_generator_async(result[1])" in compiled_code
    assert "_synchronizer._run_generator_sync(result[0])" in compiled_code
    assert "_synchronizer._run_generator_sync(result[1])" in compiled_code
    assert (
        "def fn_tuple_generators() -> "
        "tuple[typing.Generator[str, None, None], typing.Generator[int, None, None]]:" in compiled_code
    )
    assert (
        "async def aio(self) -> "
        "tuple[typing.AsyncGenerator[str, None], typing.AsyncGenerator[int, None]]:" in compiled_code
    )


def test_emit_generator_with_wrapped_yield_type():
    compiled_code = emit_wrapped_function(IR_FN_NODE_GENERATOR, TARGET)
    assert "_wrap_async_gen" in compiled_code
    assert "GenNode._from_impl(_item)" in compiled_code
    assert "async def _wrap_async_gen" in compiled_code
    assert "def _wrap_async_gen" in compiled_code
