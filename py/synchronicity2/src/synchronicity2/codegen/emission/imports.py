# Copyright Modal Labs 2026
"""Collect imports required to emit wrapper modules from data-only IR."""

from __future__ import annotations

from ..ir.annotations import AnnotationIR
from ..ir.declarations import (
    ModuleIR,
    ParameterIR,
    SignatureIR,
    WrappedClassIR,
    WrappedClassPropertyIR,
    WrappedFunctionIR,
    WrappedMethodIR,
    WrappedPropertyIR,
)


def annotation_import_modules(annotation_ir: AnnotationIR) -> frozenset[str]:
    """Return modules referenced by one annotation tree."""

    return annotation_ir.required_import_modules()


def module_import_modules(module_ir: ModuleIR) -> frozenset[str]:
    """Return modules referenced by annotations and wrapper inheritance in a module IR."""

    modules = {module_ir.synchronizer_module}
    for type_parameter_ir in module_ir.typevar_specs:
        modules.update(type_parameter_ir.import_modules)
    for class_ir in module_ir.wrapped_classes:
        modules.update(_wrapped_class_import_modules(class_ir))
    for function_ir in module_ir.wrapped_functions:
        modules.update(_wrapped_function_import_modules(function_ir))
    return frozenset(modules)


def _parameter_import_modules(parameter_ir: ParameterIR) -> frozenset[str]:
    if parameter_ir.annotation_ir is None:
        return frozenset()
    return annotation_import_modules(parameter_ir.annotation_ir)


def _signature_import_modules(signature_ir: SignatureIR) -> frozenset[str]:
    modules = set(annotation_import_modules(signature_ir.return_annotation_ir))
    for parameter_ir in signature_ir.parameters:
        modules.update(_parameter_import_modules(parameter_ir))
    return frozenset(modules)


def _wrapped_function_import_modules(function_ir: WrappedFunctionIR) -> frozenset[str]:
    signatures = function_ir.overloads or (SignatureIR(function_ir.parameters, function_ir.return_annotation_ir),)
    return _merge_signature_import_modules(*signatures)


def _wrapped_method_import_modules(method_ir: WrappedMethodIR) -> frozenset[str]:
    signatures = method_ir.overloads or (SignatureIR(method_ir.parameters, method_ir.return_annotation_ir),)
    return _merge_signature_import_modules(*signatures)


def _wrapped_property_import_modules(property_ir: WrappedPropertyIR) -> frozenset[str]:
    return _merge_annotation_import_modules(property_ir.return_annotation_ir, property_ir.setter_annotation_ir)


def _wrapped_class_property_import_modules(property_ir: WrappedClassPropertyIR) -> frozenset[str]:
    return _merge_annotation_import_modules(property_ir.return_annotation_ir)


def _wrapped_class_import_modules(class_ir: WrappedClassIR) -> frozenset[str]:
    modules = {wrapper_ref.module for _impl_ref, wrapper_ref in class_ir.wrapped_bases}
    for _attribute_name, annotation_ir in class_ir.attributes:
        if annotation_ir is not None:
            modules.update(annotation_import_modules(annotation_ir))
    for property_ir in class_ir.properties:
        modules.update(_wrapped_property_import_modules(property_ir))
    for class_property_ir in class_ir.class_properties:
        modules.update(_wrapped_class_property_import_modules(class_property_ir))
    for method_ir in class_ir.methods:
        modules.update(_wrapped_method_import_modules(method_ir))
    return frozenset(modules)


def _merge_annotation_import_modules(*annotation_irs: AnnotationIR | None) -> frozenset[str]:
    modules: set[str] = set()
    for annotation_ir in annotation_irs:
        if annotation_ir is not None:
            modules.update(annotation_import_modules(annotation_ir))
    return frozenset(modules)


def _merge_signature_import_modules(*signature_irs: SignatureIR) -> frozenset[str]:
    modules: set[str] = set()
    for signature_ir in signature_irs:
        modules.update(_signature_import_modules(signature_ir))
    return frozenset(modules)
