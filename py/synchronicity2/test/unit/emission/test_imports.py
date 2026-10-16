# Copyright Modal Labs 2026
from synchronicity2.codegen.emission.imports import annotation_import_modules
from synchronicity2.codegen.ir.annotations import (
    AsyncGeneratorAnnotationIR,
    CallableAnnotationIR,
    ParameterizedWrappedClassRefIR,
    PlainAnnotationIR,
    SelfAnnotationIR,
    Synchronicity1WrappedClassRefIR,
    WrappedClassRefIR,
)
from synchronicity2.codegen.ir.references import ObjectReferenceIR


def test_annotation_import_modules_collects_metadata_across_nested_ir_tree() -> None:
    annotation_ir = CallableAnnotationIR(
        parameter_irs=(
            AsyncGeneratorAnnotationIR(
                yield_annotation_ir=PlainAnnotationIR("YieldType", import_modules=("yield_module",)),
                send_annotation_ir=PlainAnnotationIR("SendType", import_modules=("send_module",)),
            ),
            SelfAnnotationIR(
                owner_impl=ObjectReferenceIR("self_impl", "Service"),
                wrapper=ObjectReferenceIR("self_wrapper", "Service"),
            ),
        ),
        return_annotation_ir=ParameterizedWrappedClassRefIR(
            wrapped_class_ir=Synchronicity1WrappedClassRefIR(
                impl=ObjectReferenceIR("legacy_impl", "Container"),
                wrapper=ObjectReferenceIR("legacy_wrapper", "Container"),
            ),
            type_argument_irs=(
                WrappedClassRefIR(
                    impl=ObjectReferenceIR("current_impl", "Item"),
                    wrapper=ObjectReferenceIR("current_wrapper", "Item"),
                ),
            ),
        ),
        params_signature_import_modules=("params_module",),
    )

    assert annotation_import_modules(annotation_ir) == frozenset(
        {
            "current_wrapper",
            "legacy_impl",
            "legacy_wrapper",
            "params_module",
            "send_module",
            "typing_extensions",
            "yield_module",
        }
    )
