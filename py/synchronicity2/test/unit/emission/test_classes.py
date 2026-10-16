# Copyright Modal Labs 2026
"""Unit tests for IR → emitted source (wrapper classes).

IR literals are explicit dataclasses in this module (same shapes as the parse layer).
``IMPL`` is this module so emitted implementation references match assertions.
"""

from __future__ import annotations

import dataclasses
import weakref

from synchronicity2.codegen.emission.module_codegen import emit_wrapped_class
from synchronicity2.codegen.ir.annotations import (
    AsyncContextManagerAnnotationIR,
    AsyncGeneratorAnnotationIR,
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
    TypeVarRefIR,
    UnionAnnotationIR,
    WrappedClassRefIR,
)
from synchronicity2.codegen.ir.declarations import (
    MethodBindingKind,
    ParameterIR,
    SignatureIR,
    WrappedClassIR,
    WrappedClassPropertyIR,
    WrappedMethodIR,
)
from synchronicity2.codegen.ir.references import ObjectReferenceIR

IMPL = __name__
TARGET = "test_module"

IR_CLASS_AITER_ASYNC_GEN = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitAsyncWithGenerator"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitAsyncWithGenerator"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="__aiter__",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=True,
            is_async=True,
            return_annotation_ir=AsyncGeneratorAnnotationIR(
                yield_annotation_ir=PlainAnnotationIR(signature_text="float"),
                send_annotation_ir=PlainAnnotationIR(signature_text="None"),
            ),
        ),
    ),
)

IR_CLASS_AITER_ASYNC_ITER_TYPE = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitAsyncIterType"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitAsyncIterType"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="__aiter__",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=AsyncIteratorAnnotationIR(item_ir=PlainAnnotationIR(signature_text="bool")),
        ),
    ),
)

IR_CLASS_AITER_ASYNC_NO_ANN = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitAsyncWithoutAnnotation"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitAsyncWithoutAnnotation"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="__aiter__",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="typing.Any")),
        ),
    ),
)

IR_CLASS_AITER_ASYNC_WITH_ANN = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitAsyncWithAnnotation"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitAsyncWithAnnotation"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="__aiter__",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(
                inner_ir=AsyncIteratorAnnotationIR(item_ir=PlainAnnotationIR(signature_text="int"))
            ),
        ),
    ),
)

IR_CLASS_AITER_SYNC_NO_ANN = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitSyncWithoutAnnotation"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitSyncWithoutAnnotation"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="__aiter__",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=PlainAnnotationIR(signature_text=""),
        ),
    ),
)

IR_CLASS_AITER_SYNC_WITH_ANN = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitSyncWithAnnotation"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitSyncWithAnnotation"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="__aiter__",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=AsyncIteratorAnnotationIR(item_ir=PlainAnnotationIR(signature_text="str")),
        ),
    ),
)

IR_CLASS_ASYNC_GEN = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitAsyncGeneratorClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitAsyncGeneratorClass"),
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
                    name="items",
                    kind=1,
                    annotation_ir=ListAnnotationIR(item_ir=PlainAnnotationIR(signature_text="str")),
                    default_expr=None,
                ),
            ),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=PlainAnnotationIR(signature_text=""),
        ),
        WrappedMethodIR(
            method_name="stream_items",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=True,
            is_async=True,
            return_annotation_ir=AsyncGeneratorAnnotationIR(
                yield_annotation_ir=PlainAnnotationIR(signature_text="str"),
                send_annotation_ir=PlainAnnotationIR(signature_text="None"),
            ),
        ),
        WrappedMethodIR(
            method_name="stream_with_filter",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(
                    name="prefix", kind=1, annotation_ir=PlainAnnotationIR(signature_text="str"), default_expr=None
                ),
            ),
            is_async_gen=True,
            is_async=True,
            return_annotation_ir=AsyncGeneratorAnnotationIR(
                yield_annotation_ir=PlainAnnotationIR(signature_text="str"),
                send_annotation_ir=PlainAnnotationIR(signature_text="None"),
            ),
        ),
    ),
)

IR_CLASS_ASYNC_ITERABLE = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitAsyncIterableClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitAsyncIterableClass"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="__aiter__",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=AsyncIteratorAnnotationIR(item_ir=PlainAnnotationIR(signature_text="str")),
        ),
    ),
)

IR_CLASS_ASYNC_ITERATOR = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitAsyncIteratorClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitAsyncIteratorClass"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="__aiter__",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=SelfAnnotationIR(
                owner_impl=ObjectReferenceIR(IMPL, "EmitAsyncIteratorClass"),
                wrapper=ObjectReferenceIR(TARGET, "EmitAsyncIteratorClass"),
            ),
        ),
        WrappedMethodIR(
            method_name="__anext__",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="int")),
        ),
    ),
)
IR_CLASS_METHOD_WITH_DEFAULTS = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitMethodDefaults"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitMethodDefaults"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="configure",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(
                    name="name",
                    kind=1,
                    annotation_ir=PlainAnnotationIR(signature_text="str"),
                    default_expr="'hello'",
                ),
                ParameterIR(
                    name="enabled",
                    kind=1,
                    annotation_ir=PlainAnnotationIR(signature_text="bool"),
                    default_expr="True",
                ),
                ParameterIR(
                    name="payload",
                    kind=1,
                    annotation_ir=PlainAnnotationIR(signature_text="bytes"),
                    default_expr="b'data'",
                ),
            ),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=PlainAnnotationIR(signature_text="str"),
        ),
    ),
)

IR_CLASS_AWAITABLE_METHOD = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitAwaitableMethodClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitAwaitableMethodClass"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="create_awaitable",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(name="x", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
            ),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="str")),
        ),
    ),
)
IR_CLASS_OVERLOADED_METHOD = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitOverloadedMethodClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitOverloadedMethodClass"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="resolve",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(ParameterIR(name="value", kind=1, annotation_ir=None, default_expr=None),),
            is_async_gen=False,
            is_async=True,
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
                                impl=ObjectReferenceIR(IMPL, "Node"),
                                wrapper=ObjectReferenceIR(TARGET, "Node"),
                            ),
                            default_expr=None,
                        ),
                    ),
                    return_annotation_ir=AwaitableAnnotationIR(
                        inner_ir=WrappedClassRefIR(
                            impl=ObjectReferenceIR(IMPL, "Node"),
                            wrapper=ObjectReferenceIR(TARGET, "Node"),
                        )
                    ),
                ),
            ),
        ),
    ),
)

IR_CLASS_SYNC_OVERLOADED_METHOD = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitSyncOverloadedMethodClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitSyncOverloadedMethodClass"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="decorate",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(ParameterIR(name="value", kind=1, annotation_ir=None, default_expr=None),),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=PlainAnnotationIR(signature_text="typing.Any"),
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
                    return_annotation_ir=PlainAnnotationIR(signature_text="int"),
                ),
                SignatureIR(
                    parameters=(
                        ParameterIR(
                            name="value",
                            kind=1,
                            annotation_ir=WrappedClassRefIR(
                                impl=ObjectReferenceIR(IMPL, "Node"),
                                wrapper=ObjectReferenceIR(TARGET, "Node"),
                            ),
                            default_expr=None,
                        ),
                    ),
                    return_annotation_ir=WrappedClassRefIR(
                        impl=ObjectReferenceIR(IMPL, "Node"),
                        wrapper=ObjectReferenceIR(TARGET, "Node"),
                    ),
                ),
            ),
        ),
    ),
)

IR_CLASS_COMPLEX = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitComplexClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitComplexClass"),
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
                    name="data",
                    kind=1,
                    annotation_ir=ListAnnotationIR(item_ir=PlainAnnotationIR(signature_text="str")),
                    default_expr=None,
                ),
            ),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=PlainAnnotationIR(signature_text=""),
        ),
        WrappedMethodIR(
            method_name="get_data_length",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="int")),
        ),
        WrappedMethodIR(
            method_name="process_data",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(
                    name="config",
                    kind=1,
                    annotation_ir=DictAnnotationIR(
                        key_ir=PlainAnnotationIR(signature_text="str"), value_ir=PlainAnnotationIR(signature_text="int")
                    ),
                    default_expr=None,
                ),
                ParameterIR(
                    name="optional_filter",
                    kind=1,
                    annotation_ir=OptionalAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="str")),
                    default_expr="None",
                ),
            ),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(
                inner_ir=ListAnnotationIR(item_ir=PlainAnnotationIR(signature_text="str"))
            ),
        ),
    ),
)

IR_CLASS_CONTAINER = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitContainer"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitContainer"),
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
                    name="node",
                    kind=1,
                    annotation_ir=WrappedClassRefIR(
                        impl=ObjectReferenceIR(IMPL, "EmitNode"), wrapper=ObjectReferenceIR(TARGET, "Node")
                    ),
                    default_expr=None,
                ),
                ParameterIR(
                    name="name", kind=1, annotation_ir=PlainAnnotationIR(signature_text="str"), default_expr=None
                ),
                ParameterIR(
                    name="count", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr="5"
                ),
            ),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=PlainAnnotationIR(signature_text=""),
        ),
    ),
)

IR_CLASS_COROUTINE_METHOD = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitCoroutineMethodClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitCoroutineMethodClass"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="create_coroutine",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(name="x", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
            ),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=CoroutineAnnotationIR(return_annotation_ir=PlainAnnotationIR(signature_text="str")),
        ),
    ),
)

IR_CLASS_EMPTY = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitEmptyClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitEmptyClass"),
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
            method_name="sync_method",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=PlainAnnotationIR(signature_text="int"),
        ),
    ),
)

IR_CLASS_MIXED = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitMixedClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitMixedClass"),
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
                    name="data",
                    kind=1,
                    annotation_ir=ListAnnotationIR(item_ir=PlainAnnotationIR(signature_text="str")),
                    default_expr=None,
                ),
            ),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=PlainAnnotationIR(signature_text=""),
        ),
        WrappedMethodIR(
            method_name="process_generator",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=True,
            is_async=True,
            return_annotation_ir=AsyncGeneratorAnnotationIR(
                yield_annotation_ir=PlainAnnotationIR(signature_text="str"),
                send_annotation_ir=PlainAnnotationIR(signature_text="None"),
            ),
        ),
        WrappedMethodIR(
            method_name="process_sync",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(
                    name="item", kind=1, annotation_ir=PlainAnnotationIR(signature_text="str"), default_expr=None
                ),
            ),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="str")),
        ),
        WrappedMethodIR(
            method_name="sync_method",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(
                    name="item", kind=1, annotation_ir=PlainAnnotationIR(signature_text="str"), default_expr=None
                ),
            ),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=PlainAnnotationIR(signature_text="str"),
        ),
    ),
)

IR_CLASS_NO_INIT = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitNoInit"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitNoInit"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="method",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="int")),
        ),
    ),
)

IR_CLASS_SELF = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitSelfMethodClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitSelfMethodClass"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="accept",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(
                    name="s",
                    kind=1,
                    annotation_ir=SelfAnnotationIR(
                        owner_impl=ObjectReferenceIR(IMPL, "EmitSelfMethodClass"),
                        wrapper=ObjectReferenceIR(TARGET, "EmitSelfMethodClass"),
                    ),
                    default_expr=None,
                ),
            ),
            is_async_gen=False,
            is_async=False,
            return_annotation_ir=SelfAnnotationIR(
                owner_impl=ObjectReferenceIR(IMPL, "EmitSelfMethodClass"),
                wrapper=ObjectReferenceIR(TARGET, "EmitSelfMethodClass"),
            ),
        ),
    ),
)

IR_CLASS_CLASSMETHOD_SELF = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitSelfClassmethodClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitSelfClassmethodClass"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="create",
            method_type=MethodBindingKind.CLASSMETHOD,
            parameters=(),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(
                inner_ir=SelfAnnotationIR(
                    owner_impl=ObjectReferenceIR(IMPL, "EmitSelfClassmethodClass"),
                    wrapper=ObjectReferenceIR(TARGET, "EmitSelfClassmethodClass"),
                )
            ),
        ),
    ),
)

IR_CLASS_SIMPLE = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitSimpleClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitSimpleClass"),
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
            method_name="add_to_value",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(
                    name="amount", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None
                ),
            ),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="int")),
        ),
        WrappedMethodIR(
            method_name="get_value",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="int")),
        ),
        WrappedMethodIR(
            method_name="set_value",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(
                    name="new_value", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None
                ),
            ),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="None")),
        ),
    ),
)

IR_CLASS_VARARGS = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitVarArgsClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitVarArgsClass"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="method_with_posonly",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(name="x", kind=0, annotation_ir=None, default_expr=None),
                ParameterIR(name="y", kind=0, annotation_ir=None, default_expr=None),
                ParameterIR(name="z", kind=1, annotation_ir=None, default_expr=None),
                ParameterIR(name="w", kind=1, annotation_ir=None, default_expr="10"),
            ),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="int")),
        ),
        WrappedMethodIR(
            method_name="method_with_varargs",
            method_type=MethodBindingKind.INSTANCE,
            parameters=(
                ParameterIR(name="a", kind=1, annotation_ir=PlainAnnotationIR(signature_text="int"), default_expr=None),
                ParameterIR(
                    name="args", kind=2, annotation_ir=PlainAnnotationIR(signature_text="str"), default_expr=None
                ),
                ParameterIR(name="b", kind=3, annotation_ir=None, default_expr=None),
                ParameterIR(
                    name="kwargs", kind=4, annotation_ir=PlainAnnotationIR(signature_text="float"), default_expr=None
                ),
            ),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="str")),
        ),
    ),
)

IR_CLASS_TRANSLATED_STATICMETHOD_VARARGS = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitTranslatedStaticVarArgsClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitTranslatedStaticVarArgsClass"),
    wrapped_bases=(),
    generic_type_parameters=None,
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="collect",
            method_type=MethodBindingKind.STATICMETHOD,
            parameters=(
                ParameterIR(
                    name="args",
                    kind=2,
                    annotation_ir=WrappedClassRefIR(
                        impl=ObjectReferenceIR(IMPL, "Node"),
                        wrapper=ObjectReferenceIR(TARGET, "Node"),
                    ),
                    default_expr=None,
                ),
                ParameterIR(
                    name="kwargs",
                    kind=4,
                    annotation_ir=WrappedClassRefIR(
                        impl=ObjectReferenceIR(IMPL, "Node"),
                        wrapper=ObjectReferenceIR(TARGET, "Node"),
                    ),
                    default_expr=None,
                ),
            ),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="list[str]")),
        ),
    ),
)

IR_CLASS_TRANSLATED_STATICMETHOD_SUBSCRIPTED_VARARGS = WrappedClassIR(
    impl_ref=ObjectReferenceIR(IMPL, "EmitTranslatedSubscriptedStaticVarArgsClass"),
    wrapper_ref=ObjectReferenceIR(TARGET, "EmitTranslatedSubscriptedStaticVarArgsClass"),
    wrapped_bases=(),
    generic_type_parameters=("T",),
    attributes=(),
    properties=(),
    methods=(
        WrappedMethodIR(
            method_name="collect",
            method_type=MethodBindingKind.STATICMETHOD,
            parameters=(
                ParameterIR(
                    name="args",
                    kind=2,
                    annotation_ir=ParameterizedWrappedClassRefIR(
                        wrapped_class_ir=WrappedClassRefIR(
                            impl=ObjectReferenceIR(IMPL, "Node"),
                            wrapper=ObjectReferenceIR(TARGET, "Node"),
                        ),
                        type_argument_irs=(PlainAnnotationIR(signature_text="T"),),
                    ),
                    default_expr=None,
                ),
            ),
            is_async_gen=False,
            is_async=True,
            return_annotation_ir=AwaitableAnnotationIR(inner_ir=PlainAnnotationIR(signature_text="list[str]")),
        ),
    ),
)


def _impl_short(ir: WrappedClassIR) -> str:
    return ir.impl_ref.qualname.rpartition(".")[2]


def test_emit_class_basic():
    ir = IR_CLASS_SIMPLE
    code = emit_wrapped_class(ir, TARGET)
    compile(code, "<string>", "exec")
    assert f"# Proxy type for the underlying implementation type {IMPL}.{_impl_short(ir)}." in code
    assert f"class {_impl_short(ir)}:" in code
    assert "_impl_instance" in code
    assert "@method_with_aio(" in code
    assert "def get_value(self" in code
    assert "def set_value(self" in code
    assert "def add_to_value(self" in code
    assert f"{IMPL}.EmitSimpleClass.add_to_value(self._impl_instance, amount)" in code


def test_emit_class_method_descriptors():
    code = emit_wrapped_class(IR_CLASS_SIMPLE, TARGET)
    assert "class _EmitSimpleClass_add_to_value_MethodWithAio(MethodWithAio):" in code
    assert "@method_with_aio(_EmitSimpleClass_add_to_value_MethodWithAio)" in code
    assert "async def aio(self, amount: int) -> int:" in code
    assert "def add_to_value(self, amount: int) -> int" in code
    assert "return self._sync_impl(amount)" in code
    assert "self._with_aio_from_impl = _from_impl" not in code
    assert "def _from_impl(self, impl_instance: typing.Any) -> typing.Any:" not in code
    assert "impl_method =" not in code


def test_emit_class_method_docstring_skips_with_aio_class_level_copy():
    method_ir = dataclasses.replace(IR_CLASS_AWAITABLE_METHOD.methods[0], docstring="Method docstring.")
    ir = dataclasses.replace(IR_CLASS_AWAITABLE_METHOD, methods=(method_ir,))
    code = emit_wrapped_class(ir, TARGET)
    assert code.count('"""Method docstring."""') == 3
    assert (
        'class _EmitAwaitableMethodClass_create_awaitable_MethodWithAio(MethodWithAio):\n    """Method docstring."""'
    ) not in code


def test_emit_class_complex_types():
    code = emit_wrapped_class(IR_CLASS_COMPLEX, TARGET)
    compile(code, "<string>", "exec")
    assert "config: dict[str, int]" in code
    assert (
        "optional_filter: typing.Union[str, None]" in code
        or "optional_filter: str | None" in code
        or "optional_filter: typing.Optional[str]" in code
    )
    assert "-> list[str]" in code


def test_emit_class_quotes_annotations_shadowed_by_class_namespace():
    ir = WrappedClassIR(
        impl_ref=ObjectReferenceIR(IMPL, "EmitShadowedBuiltinAnnotation"),
        wrapper_ref=ObjectReferenceIR(TARGET, "EmitShadowedBuiltinAnnotation"),
        wrapped_bases=(),
        generic_type_parameters=None,
        attributes=(),
        properties=(),
        methods=(
            WrappedMethodIR(
                method_name="list",
                method_type=MethodBindingKind.INSTANCE,
                parameters=(),
                is_async_gen=False,
                is_async=False,
                return_annotation_ir=ListAnnotationIR(item_ir=PlainAnnotationIR(signature_text="str")),
            ),
            WrappedMethodIR(
                method_name="ls",
                method_type=MethodBindingKind.INSTANCE,
                parameters=(),
                is_async_gen=False,
                is_async=False,
                return_annotation_ir=ListAnnotationIR(item_ir=PlainAnnotationIR(signature_text="str")),
            ),
        ),
    )

    code = emit_wrapped_class(ir, TARGET)

    assert 'def ls(self) -> "list[str]":' in code
    exec(code, {"weakref": weakref})


def test_emit_class_async_generators():
    code = emit_wrapped_class(IR_CLASS_ASYNC_GEN, TARGET)
    compile(code, "<string>", "exec")
    assert "_run_generator_sync" in code
    assert "_run_generator_async" in code
    assert "_sent = yield _item" in code
    assert "await _wrapped.asend(_sent)" in code
    assert "impl_method =" not in code
    # No helper functions needed when yield type doesn't need translation
    assert "@staticmethod" not in code
    assert "_wrap_async_gen" not in code


def test_emit_class_mixed_methods():
    code = emit_wrapped_class(IR_CLASS_MIXED, TARGET)
    compile(code, "<string>", "exec")
    assert "@method_with_aio(" in code
    assert "def process_sync(self" in code
    assert "def process_generator(self" in code


def test_emit_class_type_annotations_preserved():
    code = emit_wrapped_class(IR_CLASS_SIMPLE, TARGET)
    assert "new_value: int" in code
    assert "amount: int" in code
    assert "-> int" in code
    assert "-> None" in code


def test_emit_class_impl_instance_access():
    ir = IR_CLASS_SIMPLE
    code = emit_wrapped_class(ir, TARGET)
    short = _impl_short(ir)
    assert f"self._impl_instance = {IMPL}.{short}(" in code
    assert f"class {short}:" in code
    assert "_synchronizer._run_function_async" in code


def test_emit_class_multiple_wrapped_registry_entries():
    for ir in (IR_CLASS_SIMPLE, IR_CLASS_COMPLEX):
        code = emit_wrapped_class(ir, TARGET)
        compile(code, "<string>", "exec")
        assert f"class {_impl_short(ir)}:" in code
        assert "_impl_instance" in code


def test_emit_class_no_async_methods():
    ir = IR_CLASS_EMPTY
    code = emit_wrapped_class(ir, TARGET)
    compile(code, "<string>", "exec")
    assert f"class {_impl_short(ir)}:" in code
    assert "_impl_instance" in code


def test_emit_class_method_with_varargs():
    code = emit_wrapped_class(IR_CLASS_VARARGS, TARGET)
    assert "*args: str" in code
    assert "**kwargs: float" in code
    assert "a: int, *args: str, b, **kwargs: float" in code
    assert "x, y, /" in code
    assert "*args" in code
    assert "b=b" in code
    assert "**kwargs" in code
    assert f"{IMPL}.EmitVarArgsClass.method_with_varargs(self._impl_instance, a, *args, b=b, **kwargs)" in code


def test_emit_staticmethod_translated_varargs_and_kwargs():
    code = emit_wrapped_class(IR_CLASS_TRANSLATED_STATICMETHOD_VARARGS, TARGET)

    assert 'def collect(*args: "Node", **kwargs: "Node") -> list[str]:' in code
    assert "args_impl = tuple(_item._impl_instance for _item in args)" in code
    assert "kwargs_impl = {_key: _value._impl_instance for _key, _value in kwargs.items()}" in code
    assert f"{IMPL}.EmitTranslatedStaticVarArgsClass.collect(*args_impl, **kwargs_impl)" in code


def test_emit_staticmethod_translated_subscripted_varargs_quotes_forward_ref():
    code = emit_wrapped_class(IR_CLASS_TRANSLATED_STATICMETHOD_SUBSCRIPTED_VARARGS, TARGET)

    assert 'def __call__(self, *args: "Node[T]") -> list[str]:' in code
    assert 'def aio(self, *args: "Node[T]") -> list[str]:' in code
    assert 'def collect(*args: "Node[T]") -> list[str]:' in code
    assert "args_impl = tuple(_item._impl_instance for _item in args)" in code


def test_emit_staticmethod_with_method_local_typevar_without_type_checking_stub():
    ir = WrappedClassIR(
        impl_ref=ObjectReferenceIR(IMPL, "EmitMethodLocalTypeVarClass"),
        wrapper_ref=ObjectReferenceIR(TARGET, "EmitMethodLocalTypeVarClass"),
        wrapped_bases=(),
        generic_type_parameters=None,
        attributes=(),
        properties=(),
        class_properties=(),
        manual_attributes=(),
        methods=(
            WrappedMethodIR(
                method_name="echo",
                method_type=MethodBindingKind.STATICMETHOD,
                parameters=(
                    ParameterIR(
                        name="value",
                        kind=1,
                        annotation_ir=TypeVarRefIR(name="T"),
                        default_expr=None,
                    ),
                ),
                is_async_gen=False,
                is_async=True,
                return_annotation_ir=AwaitableAnnotationIR(inner_ir=TypeVarRefIR(name="T")),
            ),
        ),
    )

    code = emit_wrapped_class(ir, TARGET)

    assert "@staticmethod_with_aio(_EmitMethodLocalTypeVarClass_echo_MethodWithAio)" in code
    assert "if typing.TYPE_CHECKING" not in code


def test_emit_class_method_various_builtin_default_values():
    code = emit_wrapped_class(IR_CLASS_METHOD_WITH_DEFAULTS, TARGET)
    compile(code, "<string>", "exec")
    assert "def configure(self, name: str = 'hello', enabled: bool = True, payload: bytes = b'data') -> str:" in code


def test_emit_class_constructor_with_wrapped_param():
    code = emit_wrapped_class(IR_CLASS_CONTAINER, TARGET)
    compile(code, "<string>", "exec")
    assert 'def __init__(self, node: "Node", name: str, count: int = 5):' in code
    assert "node_impl = node._impl_instance" in code
    assert f"self._impl_instance = {IMPL}.EmitContainer(node_impl, name, count)" in code
    assert "name_impl" not in code
    assert "count_impl" not in code


def test_emit_class_sync_method_returning_coroutine():
    code = emit_wrapped_class(IR_CLASS_COROUTINE_METHOD, TARGET)
    assert "class _EmitCoroutineMethodClass_create_coroutine_MethodWithAio(MethodWithAio):" in code
    assert "@method_with_aio(_EmitCoroutineMethodClass_create_coroutine_MethodWithAio)" in code
    assert "def create_coroutine(self, x: int) -> str:" in code
    assert "async def aio(self, x: int) -> str:" in code
    assert "_run_function_sync" in code
    assert "_run_function_async" in code


def test_emit_class_sync_method_returning_awaitable():
    code = emit_wrapped_class(IR_CLASS_AWAITABLE_METHOD, TARGET)
    assert "class _EmitAwaitableMethodClass_create_awaitable_MethodWithAio(MethodWithAio):" in code
    assert "@method_with_aio(_EmitAwaitableMethodClass_create_awaitable_MethodWithAio)" in code
    assert "def create_awaitable(self, x: int) -> str:" in code
    assert "async def aio(self, x: int) -> str:" in code
    assert "_run_function_sync" in code
    assert "_run_function_async" in code


def test_emit_class_method_overloads_translate_each_overload():
    code = emit_wrapped_class(IR_CLASS_OVERLOADED_METHOD, TARGET)
    compile(code, "<string>", "exec")
    assert "class _EmitOverloadedMethodClass_resolve_MethodWithAio(MethodWithAio):" in code
    assert "def __call__(self, value: int) -> int: ..." in code
    assert 'def __call__(self, value: "Node") -> "Node": ...' in code
    assert "async def aio(self, value: int) -> int: ..." in code
    assert 'async def aio(self, value: "Node") -> "Node": ...' in code
    assert "def __call__(self, value) -> typing.Any:" in code
    assert "return self._sync_impl(value)" in code
    assert "async def aio(self, value) -> typing.Any:" in code
    assert "@method_with_aio(_EmitOverloadedMethodClass_resolve_MethodWithAio)" in code
    assert (
        "_run_function_async("
        "test.unit.emission.test_classes.EmitOverloadedMethodClass.resolve("
        "self._wrapper_instance._impl_instance, value))" in code
    )
    assert "def resolve(self, value) -> typing.Any:" in code


def test_emit_sync_method_overloads():
    code = emit_wrapped_class(IR_CLASS_SYNC_OVERLOADED_METHOD, TARGET)
    compile(code, "<string>", "exec")
    assert "@typing.overload" in code
    assert "def decorate(self, value: int) -> int: ..." in code
    assert 'def decorate(self, value: "Node") -> "Node": ...' in code
    assert "def decorate(self, value) -> typing.Any:" in code


def test_emit_class_aiter_typed_iter():
    code = emit_wrapped_class(IR_CLASS_ASYNC_ITERABLE, TARGET)
    assert "def __iter__(self) -> synchronicity2.types.SyncOrAsyncIterator[str]:" in code
    assert "def __aiter__(self) -> synchronicity2.types.SyncOrAsyncIterator[str]:" in code


def test_emit_class_anext_typed_next():
    code = emit_wrapped_class(IR_CLASS_ASYNC_ITERATOR, TARGET)
    assert "def __next__(self) -> int:" in code
    assert "async def __anext__(self) -> int:" in code
    assert "def __iter__(self) -> typing_extensions.Self:" in code
    assert "def __aiter__(self) -> typing_extensions.Self:" in code


def test_emit_class_preserves_typing_self():
    code = emit_wrapped_class(IR_CLASS_SELF, TARGET)
    assert "def accept(self, s: typing_extensions.Self) -> typing_extensions.Self:" in code
    assert "typing.cast(typing_extensions.Self, self._from_impl(result))" in code
    assert "s_impl = s._impl_instance" in code


def test_emit_classmethod_with_aio_threads_owner_self_type_into_helper():
    code = emit_wrapped_class(IR_CLASS_CLASSMETHOD_SELF, TARGET)
    assert (
        "_EmitSelfClassmethodClass_create_SelfType = typing.TypeVar("
        '"_EmitSelfClassmethodClass_create_SelfType", bound="EmitSelfClassmethodClass")'
    ) in code
    assert (
        "class _EmitSelfClassmethodClass_create_MethodWithAio(MethodWithAio, "
        "typing.Generic[_EmitSelfClassmethodClass_create_SelfType]):"
    ) in code
    assert "def __call__(self) -> _EmitSelfClassmethodClass_create_SelfType:" in code
    assert "async def aio(self) -> _EmitSelfClassmethodClass_create_SelfType:" in code


def test_emit_class_aiter_signature_variations():
    code = emit_wrapped_class(IR_CLASS_AITER_SYNC_WITH_ANN, TARGET)
    assert "def __iter__(self) -> synchronicity2.types.SyncOrAsyncIterator[str]:" in code
    assert "def __aiter__(self) -> synchronicity2.types.SyncOrAsyncIterator[str]:" in code

    code = emit_wrapped_class(IR_CLASS_AITER_SYNC_NO_ANN, TARGET)
    assert "def __iter__(self):" in code
    assert "def __aiter__(self):" in code
    assert " -> :" not in code

    code = emit_wrapped_class(IR_CLASS_AITER_ASYNC_WITH_ANN, TARGET)
    assert "def __iter__(self) -> synchronicity2.types.SyncOrAsyncIterator[int]:" in code
    assert "def __aiter__(self) -> synchronicity2.types.SyncOrAsyncIterator[int]:" in code

    code = emit_wrapped_class(IR_CLASS_AITER_ASYNC_NO_ANN, TARGET)
    assert "def __iter__(self) -> typing.Any:" in code
    assert "def __aiter__(self) -> typing.Any:" in code

    code = emit_wrapped_class(IR_CLASS_AITER_ASYNC_GEN, TARGET)
    assert "def __iter__(self) -> typing.Generator[float, None, None]:" in code
    assert "def __aiter__(self) -> typing.AsyncGenerator[float, None]:" in code

    code = emit_wrapped_class(IR_CLASS_AITER_ASYNC_ITER_TYPE, TARGET)
    assert "def __iter__(self) -> synchronicity2.types.SyncOrAsyncIterator[bool]:" in code
    assert "def __aiter__(self) -> synchronicity2.types.SyncOrAsyncIterator[bool]:" in code


def test_emit_classproperty_translation():
    ir = WrappedClassIR(
        impl_ref=ObjectReferenceIR(IMPL, "EmitClassPropertyService"),
        wrapper_ref=ObjectReferenceIR(TARGET, "EmitClassPropertyService"),
        wrapped_bases=(),
        generic_type_parameters=None,
        attributes=(),
        properties=(),
        methods=(),
        class_properties=(
            WrappedClassPropertyIR(
                name="manager",
                return_annotation_ir=WrappedClassRefIR(
                    impl=ObjectReferenceIR(IMPL, "EmitClassPropertyManager"),
                    wrapper=ObjectReferenceIR(TARGET, "EmitClassPropertyManager"),
                ),
            ),
        ),
    )

    code = emit_wrapped_class(ir, TARGET)

    assert '    @classproperty\n    def manager(cls) -> "EmitClassPropertyManager":' in code
    assert "_impl_val = test.unit.emission.test_classes.EmitClassPropertyService.manager" in code
    assert "return EmitClassPropertyManager._from_impl(_impl_val)" in code


def test_emit_classmethod_and_staticmethod_async_context_manager_helper_binding():
    ir = WrappedClassIR(
        impl_ref=ObjectReferenceIR(IMPL, "EmitContextFactories"),
        wrapper_ref=ObjectReferenceIR(TARGET, "EmitContextFactories"),
        wrapped_bases=(),
        generic_type_parameters=None,
        attributes=(),
        properties=(),
        methods=(
            WrappedMethodIR(
                method_name="connect_class",
                method_type=MethodBindingKind.CLASSMETHOD,
                parameters=(),
                is_async_gen=False,
                is_async=False,
                return_annotation_ir=AsyncContextManagerAnnotationIR(
                    value_ir=WrappedClassRefIR(
                        impl=ObjectReferenceIR(IMPL, "EmitConnection"),
                        wrapper=ObjectReferenceIR(TARGET, "EmitConnection"),
                    )
                ),
            ),
            WrappedMethodIR(
                method_name="connect_static",
                method_type=MethodBindingKind.STATICMETHOD,
                parameters=(),
                is_async_gen=False,
                is_async=False,
                return_annotation_ir=AsyncContextManagerAnnotationIR(
                    value_ir=WrappedClassRefIR(
                        impl=ObjectReferenceIR(IMPL, "EmitConnection"),
                        wrapper=ObjectReferenceIR(TARGET, "EmitConnection"),
                    )
                ),
            ),
        ),
    )

    code = emit_wrapped_class(ir, TARGET)

    assert "value_wrapper=cls._wrap_async_cm_EmitConnection" in code
    assert "value_wrapper=EmitContextFactories._wrap_async_cm_EmitConnection" in code


def test_emit_sequence_and_callable_ellipsis_annotations():
    ir = WrappedClassIR(
        impl_ref=ObjectReferenceIR(IMPL, "EmitSequenceCallableService"),
        wrapper_ref=ObjectReferenceIR(TARGET, "EmitSequenceCallableService"),
        wrapped_bases=(),
        generic_type_parameters=None,
        attributes=(
            (
                "deps",
                CallableAnnotationIR(
                    parameter_irs=None,
                    return_annotation_ir=SequenceAnnotationIR(
                        item_ir=WrappedClassRefIR(
                            impl=ObjectReferenceIR(IMPL, "Node"),
                            wrapper=ObjectReferenceIR(TARGET, "Node"),
                        )
                    ),
                ),
            ),
        ),
        properties=(),
        methods=(
            WrappedMethodIR(
                method_name="clone_all",
                method_type=MethodBindingKind.INSTANCE,
                parameters=(
                    ParameterIR(
                        name="nodes",
                        kind=1,
                        annotation_ir=SequenceAnnotationIR(
                            item_ir=WrappedClassRefIR(
                                impl=ObjectReferenceIR(IMPL, "Node"),
                                wrapper=ObjectReferenceIR(TARGET, "Node"),
                            )
                        ),
                        default_expr=None,
                    ),
                ),
                is_async_gen=False,
                is_async=True,
                return_annotation_ir=AwaitableAnnotationIR(
                    inner_ir=SequenceAnnotationIR(
                        item_ir=WrappedClassRefIR(
                            impl=ObjectReferenceIR(IMPL, "Node"),
                            wrapper=ObjectReferenceIR(TARGET, "Node"),
                        )
                    )
                ),
            ),
            WrappedMethodIR(
                method_name="clone_collection",
                method_type=MethodBindingKind.INSTANCE,
                parameters=(
                    ParameterIR(
                        name="nodes",
                        kind=1,
                        annotation_ir=CollectionAnnotationIR(
                            item_ir=WrappedClassRefIR(
                                impl=ObjectReferenceIR(IMPL, "Node"),
                                wrapper=ObjectReferenceIR(TARGET, "Node"),
                            )
                        ),
                        default_expr=None,
                    ),
                ),
                is_async_gen=False,
                is_async=True,
                return_annotation_ir=AwaitableAnnotationIR(
                    inner_ir=CollectionAnnotationIR(
                        item_ir=WrappedClassRefIR(
                            impl=ObjectReferenceIR(IMPL, "Node"),
                            wrapper=ObjectReferenceIR(TARGET, "Node"),
                        )
                    )
                ),
            ),
        ),
    )

    code = emit_wrapped_class(ir, TARGET)

    assert 'def deps(self) -> "typing.Callable[..., typing.Sequence[Node]]":' in code
    assert 'def clone_all(self, nodes: "typing.Sequence[Node]") -> "typing.Sequence[Node]":' in code
    assert 'def clone_collection(self, nodes: "typing.Collection[Node]") -> "typing.Collection[Node]":' in code
    assert "nodes_impl = [x._impl_instance for x in nodes]" in code


def test_emit_class_without_explicit_init():
    ir = IR_CLASS_NO_INIT
    code = emit_wrapped_class(ir, TARGET)
    assert "def __init__(self):" in code
    assert "def __init__(self, *args, **kwargs):" not in code
    assert f"{IMPL}.{_impl_short(ir)}()" in code


def test_emit_getattr_without_union_fallback_uses_plain_return():
    ir = WrappedClassIR(
        impl_ref=ObjectReferenceIR(IMPL, "EmitDynamicOwner"),
        wrapper_ref=ObjectReferenceIR(TARGET, "EmitDynamicOwner"),
        wrapped_bases=(),
        generic_type_parameters=None,
        attributes=(),
        properties=(),
        methods=(
            WrappedMethodIR(
                method_name="__getattr__",
                method_type=MethodBindingKind.INSTANCE,
                parameters=(
                    ParameterIR(
                        name="name",
                        kind=1,
                        annotation_ir=PlainAnnotationIR(signature_text="str"),
                        default_expr=None,
                    ),
                ),
                is_async_gen=False,
                is_async=False,
                return_annotation_ir=PlainAnnotationIR(signature_text="typing.Any"),
            ),
        ),
    )

    code = emit_wrapped_class(ir, TARGET)

    assert "return result" in code
    assert "return _wrap_maybe_from_impl(result, _synchronizer)" not in code


def test_emit_getattr_with_union_any_fallback_uses_runtime_fallback_wrap():
    ir = WrappedClassIR(
        impl_ref=ObjectReferenceIR(IMPL, "EmitExplicitDynamicOwner"),
        wrapper_ref=ObjectReferenceIR(TARGET, "EmitExplicitDynamicOwner"),
        wrapped_bases=(),
        generic_type_parameters=None,
        attributes=(),
        properties=(),
        methods=(
            WrappedMethodIR(
                method_name="__getattr__",
                method_type=MethodBindingKind.INSTANCE,
                parameters=(
                    ParameterIR(
                        name="name",
                        kind=1,
                        annotation_ir=PlainAnnotationIR(signature_text="str"),
                        default_expr=None,
                    ),
                ),
                is_async_gen=False,
                is_async=False,
                return_annotation_ir=UnionAnnotationIR(
                    arm_irs=(
                        WrappedClassRefIR(
                            impl=ObjectReferenceIR(IMPL, "EmitPayload"),
                            wrapper=ObjectReferenceIR(TARGET, "EmitPayload"),
                        ),
                        PlainAnnotationIR(signature_text="typing.Any"),
                    )
                ),
            ),
        ),
    )

    code = emit_wrapped_class(ir, TARGET)

    expected_branch = (
        "EmitPayload._from_impl(_v) if isinstance(_v, test.unit.emission.test_classes.EmitPayload) else _v"
    )
    assert expected_branch in code
