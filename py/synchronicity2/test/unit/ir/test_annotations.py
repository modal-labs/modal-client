# Copyright Modal Labs 2026
import pytest

from synchronicity2.codegen.ir.annotations import (
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
    walk_annotation_irs,
)
from synchronicity2.codegen.ir.references import ObjectReferenceIR

PLAIN_IR = PlainAnnotationIR("str")
WRAPPED_IR = WrappedClassRefIR(
    impl=ObjectReferenceIR("implementation", "Service"),
    wrapper=ObjectReferenceIR("generated", "Service"),
)


@pytest.mark.parametrize(
    ("annotation_ir", "expected"),
    [
        (PLAIN_IR, ()),
        (WRAPPED_IR, ()),
        (
            Synchronicity1WrappedClassRefIR(
                impl=ObjectReferenceIR("legacy_implementation", "Service"),
                wrapper=ObjectReferenceIR("legacy_generated", "Service"),
            ),
            (),
        ),
        (TypeVarRefIR("T"), ()),
        (
            SelfAnnotationIR(
                owner_impl=ObjectReferenceIR("implementation", "Service"),
                wrapper=ObjectReferenceIR("generated", "Service"),
            ),
            (),
        ),
        (ListAnnotationIR(PLAIN_IR), (PLAIN_IR,)),
        (DictAnnotationIR(PLAIN_IR, WRAPPED_IR), (PLAIN_IR, WRAPPED_IR)),
        (SequenceAnnotationIR(PLAIN_IR), (PLAIN_IR,)),
        (CollectionAnnotationIR(PLAIN_IR), (PLAIN_IR,)),
        (TupleAnnotationIR((PLAIN_IR, WRAPPED_IR), variadic=False), (PLAIN_IR, WRAPPED_IR)),
        (OptionalAnnotationIR(PLAIN_IR), (PLAIN_IR,)),
        (UnionAnnotationIR((PLAIN_IR, WRAPPED_IR)), (PLAIN_IR, WRAPPED_IR)),
        (AsyncGeneratorAnnotationIR(PLAIN_IR, WRAPPED_IR), (PLAIN_IR, WRAPPED_IR)),
        (AsyncGeneratorAnnotationIR(PLAIN_IR, None), (PLAIN_IR,)),
        (SyncGeneratorAnnotationIR(PLAIN_IR, WRAPPED_IR, PLAIN_IR), (PLAIN_IR, WRAPPED_IR, PLAIN_IR)),
        (SyncIteratorAnnotationIR(PLAIN_IR), (PLAIN_IR,)),
        (AsyncIteratorAnnotationIR(PLAIN_IR), (PLAIN_IR,)),
        (AsyncIterableAnnotationIR(PLAIN_IR), (PLAIN_IR,)),
        (CoroutineAnnotationIR(PLAIN_IR), (PLAIN_IR,)),
        (AwaitableAnnotationIR(PLAIN_IR), (PLAIN_IR,)),
        (AsyncContextManagerAnnotationIR(PLAIN_IR), (PLAIN_IR,)),
        (CallableAnnotationIR((PLAIN_IR,), WRAPPED_IR), (PLAIN_IR, WRAPPED_IR)),
        (CallableAnnotationIR(None, WRAPPED_IR), (WRAPPED_IR,)),
        (
            ParameterizedWrappedClassRefIR(WRAPPED_IR, (PLAIN_IR,)),
            (WRAPPED_IR, PLAIN_IR),
        ),
    ],
)
def test_referenced_irs_returns_direct_annotation_children(
    annotation_ir: AnnotationIR, expected: tuple[AnnotationIR, ...]
) -> None:
    assert annotation_ir.referenced_irs() == expected


def test_walk_annotation_irs_visits_root_and_descendants_depth_first() -> None:
    parameterized_ir = ParameterizedWrappedClassRefIR(WRAPPED_IR, (PLAIN_IR,))
    dict_ir = DictAnnotationIR(PLAIN_IR, parameterized_ir)
    callable_ir = CallableAnnotationIR((dict_ir,), WRAPPED_IR)

    assert tuple(walk_annotation_irs(callable_ir)) == (
        callable_ir,
        dict_ir,
        PLAIN_IR,
        parameterized_ir,
        WRAPPED_IR,
        PLAIN_IR,
        WRAPPED_IR,
    )
