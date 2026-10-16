# Copyright Modal Labs 2026
"""Parse registered implementation declarations into data-only IR."""

from __future__ import annotations

import collections.abc
import inspect
import sys
import types
import typing

from synchronicity2.module import (
    _IMPL_WRAPPER_LOCATION_ATTR,
    Module,
    RegistrationInfo,
    _direct_wrapper_location,
)

from ..ir.annotations import (
    AnnotationIR,
    AsyncContextManagerAnnotationIR,
    AsyncIteratorAnnotationIR,
    AwaitableAnnotationIR,
    CoroutineAnnotationIR,
)
from ..ir.declarations import (
    ManualClassAttributeAccessKind,
    ManualClassAttributeIR,
    ManualReexportIR,
    MethodBindingKind,
    ModuleIR,
    SignatureIR,
    WrappedClassIR,
    WrappedClassPropertyIR,
    WrappedFunctionIR,
    WrappedMethodIR,
    WrappedPropertyIR,
)
from ..ir.references import ObjectReferenceIR
from .annotations import _synchronicity1_wrapper_ref, parse_annotation
from .signatures import (
    _extract_typevars_from_function,
    _normalize_async_annotation,
    _safe_get_annotations,
    is_async_generator,
    parse_parameters_to_ir,
)
from .type_parameters import type_parameter_irs_from_collected

if typing.TYPE_CHECKING:
    from synchronicity import Synchronizer as Synchronicity1Synchronizer


def _get_wrapper_location(t: type) -> tuple[str, str] | None:
    return _direct_wrapper_location(t)


def _wrapper_ref_for_impl_type(
    impl_type: type,
    synchronicity1_synchronizer: Synchronicity1Synchronizer | None,
) -> ObjectReferenceIR | None:
    loc = _get_wrapper_location(impl_type)
    if loc is not None:
        return ObjectReferenceIR(*loc)
    synchronicity1_ref = _synchronicity1_wrapper_ref(impl_type, synchronicity1_synchronizer)
    if synchronicity1_ref is not None:
        return synchronicity1_ref
    return None


def _is_manual_wrapper(obj: object, *, manual_wrapper_ids: frozenset[int]) -> bool:
    return id(obj) in manual_wrapper_ids


def _manual_class_attribute_access_kind(obj: object) -> ManualClassAttributeAccessKind:
    if isinstance(obj, (classmethod, staticmethod, property)) or _is_classproperty_descriptor(obj):
        return ManualClassAttributeAccessKind.RAW_CLASS_DICT
    if getattr(obj, "_synchronicity_raw_class_dict", False):
        return ManualClassAttributeAccessKind.RAW_CLASS_DICT
    return ManualClassAttributeAccessKind.ATTRIBUTE


def _is_classproperty_descriptor(obj: object) -> bool:
    descriptor_type = type(obj)
    return descriptor_type.__name__ == "classproperty" and isinstance(getattr(obj, "fget", None), classmethod)


def _is_dunder_name(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _should_scan_class_member_name(name: str, *, include_underscored_methods: bool) -> bool:
    if name in _TYPE_SCANNED_DUNDER_METHODS:
        return True
    if _is_dunder_name(name):
        return False
    if name.startswith("_"):
        return include_underscored_methods
    return True


def _should_include_private_staticmethod(name: str, *, is_manual: bool) -> bool:
    return is_manual or not name.startswith("_")


def _manual_wrapper_impl_ref(module: Module, obj: object) -> ObjectReferenceIR:
    ref = module._manual_wrapper_ref(obj)
    if ref is None:
        raise TypeError(f"Manual wrapper object {obj!r} is missing manual wrapper reference metadata")
    return ObjectReferenceIR(module=ref.module, qualname=ref.qualname)


def _check_annotation_for_cross_refs(
    annotation,
    current_module: str,
    cross_module_refs: dict,
) -> None:
    if isinstance(annotation, type):
        wrapper_location = _get_wrapper_location(annotation)
        if wrapper_location is not None:
            target_module, wrapper_name = wrapper_location
            if target_module != current_module:
                if target_module not in cross_module_refs:
                    cross_module_refs[target_module] = set()
                cross_module_refs[target_module].add(wrapper_name)

    args = typing.get_args(annotation)
    if args:
        for arg in args:
            _check_annotation_for_cross_refs(arg, current_module, cross_module_refs)


def _check_impl_type_for_cross_ref(
    impl_type: type,
    current_module: str,
    cross_module_refs: dict[str, set[str]],
) -> None:
    wrapper_location = _get_wrapper_location(impl_type)
    if wrapper_location is None:
        return
    target_module, wrapper_name = wrapper_location
    if target_module != current_module:
        if target_module not in cross_module_refs:
            cross_module_refs[target_module] = set()
        cross_module_refs[target_module].add(wrapper_name)


def _iter_overload_functions(f: types.FunctionType) -> tuple[types.FunctionType, ...]:
    return tuple(typing.cast(types.FunctionType, overload) for overload in typing.get_overloads(f))


_FORWARDED_DUNDER_METHODS = frozenset(
    {
        "__get__",
        "__getattr__",
        "__call__",
        "__contains__",
        "__delitem__",
        "__getitem__",
        "__setitem__",
    }
)
_TYPE_SCANNED_DUNDER_METHODS = _FORWARDED_DUNDER_METHODS | frozenset(
    {
        "__aenter__",
        "__aexit__",
        "__aiter__",
        "__anext__",
    }
)


def cross_module_imports_for_module(
    module_name: str,
    module_items: dict,
    *,
    manual_wrapper_ids: frozenset[int] = frozenset(),
    forbidden_wrapper_modules: frozenset[str] | None = None,
) -> dict[str, set[str]]:
    cross_module_refs: dict[str, set[str]] = {}

    for obj, _registration in module_items.items():
        if _is_manual_wrapper(obj, manual_wrapper_ids=manual_wrapper_ids):
            continue
        if isinstance(obj, types.FunctionType):
            for f in (obj, *_iter_overload_functions(obj)):
                annotations = _safe_get_annotations(f, forbidden_wrapper_modules=forbidden_wrapper_modules)
                for annotation in annotations.values():
                    _check_annotation_for_cross_refs(annotation, module_name, cross_module_refs)
        elif isinstance(obj, type):
            include_underscored_methods = _registration.include_underscored_methods
            for base in getattr(obj, "__bases__", ()):
                _check_impl_type_for_cross_ref(base, module_name, cross_module_refs)
            for method_name, attr in tuple(obj.__dict__.items()):
                is_manual = _is_manual_wrapper(attr, manual_wrapper_ids=manual_wrapper_ids)
                if (
                    not _should_scan_class_member_name(
                        method_name,
                        include_underscored_methods=include_underscored_methods,
                    )
                    and not is_manual
                ):
                    continue
                if is_manual and not isinstance(attr, (classmethod, staticmethod)) and not inspect.isfunction(attr):
                    continue
                if isinstance(attr, staticmethod) and not _should_include_private_staticmethod(
                    method_name,
                    is_manual=is_manual,
                ):
                    continue
                if isinstance(attr, classmethod | staticmethod):
                    method = typing.cast(types.FunctionType, attr.__func__)
                elif isinstance(attr, property) or _is_classproperty_descriptor(attr):
                    if method_name.startswith("_") and include_underscored_methods and not is_manual:
                        continue
                    getter = attr.fget
                    setter = attr.fset if isinstance(attr, property) else None
                    if getter is not None:
                        getter_func = getter.__func__ if isinstance(getter, classmethod) else getter
                        annotations = _safe_get_annotations(
                            getter_func, forbidden_wrapper_modules=forbidden_wrapper_modules
                        )
                        for annotation in annotations.values():
                            _check_annotation_for_cross_refs(annotation, module_name, cross_module_refs)
                    if setter is not None:
                        annotations = _safe_get_annotations(setter, forbidden_wrapper_modules=forbidden_wrapper_modules)
                        for annotation in annotations.values():
                            _check_annotation_for_cross_refs(annotation, module_name, cross_module_refs)
                    continue
                elif inspect.isfunction(attr):
                    method = attr
                else:
                    continue
                for f in (method, *_iter_overload_functions(method)):
                    annotations = _safe_get_annotations(f, forbidden_wrapper_modules=forbidden_wrapper_modules)
                    for annotation in annotations.values():
                        _check_annotation_for_cross_refs(annotation, module_name, cross_module_refs)

    return cross_module_refs


def build_module_ir(
    module: Module,
    *,
    synchronizer_module: str,
    forbidden_wrapper_modules: frozenset[str] | None = None,
    synchronicity1_synchronizer: Synchronicity1Synchronizer | None = None,
) -> ModuleIR:
    """Step 1–2 for a single output module: layout, cross-refs, and collected type variables."""
    module_items = module._module_items()
    impl_modules = {o.__module__ for o in module_items.keys()}
    manual_wrapper_ids = module._manual_wrapper_ids
    cross = cross_module_imports_for_module(
        module.target_module,
        module_items,
        manual_wrapper_ids=manual_wrapper_ids,
        forbidden_wrapper_modules=forbidden_wrapper_modules,
    )

    classes: list[type] = []
    class_registrations: dict[type, RegistrationInfo] = {}
    functions: list[tuple[types.FunctionType, str]] = []
    manual_reexports: list[ManualReexportIR] = []
    for o, registration in module_items.items():
        if _is_manual_wrapper(o, manual_wrapper_ids=manual_wrapper_ids):
            manual_reexports.append(
                ManualReexportIR(
                    impl_ref=_manual_wrapper_impl_ref(module, o),
                    export_name=registration.name,
                )
            )
            continue
        if isinstance(o, type):
            classes.append(o)
            class_registrations[o] = registration
        elif isinstance(o, types.FunctionType):
            functions.append((o, registration.name))

    for export_name, ref in module._manual_export_refs.items():
        manual_reexports.append(
            ManualReexportIR(
                impl_ref=ObjectReferenceIR(ref.module, ref.qualname),
                export_name=export_name,
            )
        )

    module_typevars: dict[str, typing.TypeVar | typing.ParamSpec] = {}

    for func, _export_name in functions:
        for overload_func in (func, *_iter_overload_functions(func)):
            annotations = _safe_get_annotations(overload_func, forbidden_wrapper_modules=forbidden_wrapper_modules)
            module_typevars.update(_extract_typevars_from_function(overload_func, annotations))

    for cls in classes:
        bases_to_check = getattr(cls, "__orig_bases__", cls.__bases__)
        for base in bases_to_check:
            origin = typing.get_origin(base)
            if origin is not None and hasattr(origin, "__name__") and origin.__name__ == "Generic":
                args = typing.get_args(base)
                for arg in args:
                    if isinstance(arg, typing.TypeVar) or isinstance(arg, typing.ParamSpec):
                        module_typevars[arg.__name__] = arg

        for name, attr in tuple(cls.__dict__.items()):
            include_underscored_methods = class_registrations[cls].include_underscored_methods
            if (
                not _should_scan_class_member_name(
                    name,
                    include_underscored_methods=include_underscored_methods,
                )
            ) or _is_manual_wrapper(
                attr,
                manual_wrapper_ids=manual_wrapper_ids,
            ):
                continue
            if isinstance(attr, staticmethod) and not _should_include_private_staticmethod(name, is_manual=False):
                continue
            if isinstance(attr, classmethod | staticmethod):
                method = typing.cast(types.FunctionType, attr.__func__)
            elif inspect.isfunction(attr):
                method = attr
            else:
                continue
            for overload_method in (method, *_iter_overload_functions(method)):
                annotations = _safe_get_annotations(
                    overload_method, forbidden_wrapper_modules=forbidden_wrapper_modules
                )
                module_typevars.update(_extract_typevars_from_function(overload_method, annotations))

    known_impl_types = frozenset(classes)
    typevar_specs = type_parameter_irs_from_collected(
        module_typevars,
        known_impl_types,
        module.target_module,
        impl_modules=frozenset(impl_modules),
        synchronicity1_synchronizer=synchronicity1_synchronizer,
    )
    cross_frozen = {k: frozenset(v) for k, v in cross.items()}

    impl_mods = frozenset(impl_modules)
    wrapped_classes = tuple(
        parse_wrapped_class(
            c,
            module.target_module,
            impl_modules=impl_mods,
            manual_wrapper_ids=manual_wrapper_ids,
            include_underscored_methods=class_registrations[c].include_underscored_methods,
            forbidden_wrapper_modules=forbidden_wrapper_modules,
            synchronicity1_synchronizer=synchronicity1_synchronizer,
        )
        for c in classes
    )
    wrapped_functions_list: list[WrappedFunctionIR] = []
    for f, export_name in functions:
        g = sys.modules[f.__module__].__dict__ if f.__module__ in sys.modules else None
        wrapped_functions_list.append(
            parse_wrapped_function(
                f,
                module.target_module,
                export_name=export_name,
                globals_dict=g,
                impl_modules=impl_mods,
                forbidden_wrapper_modules=forbidden_wrapper_modules,
                synchronicity1_synchronizer=synchronicity1_synchronizer,
            )
        )
    wrapped_functions = tuple(wrapped_functions_list)

    return ModuleIR(
        target_module=module.target_module,
        synchronizer_module=synchronizer_module,
        impl_modules=frozenset(impl_modules),
        cross_module_imports=cross_frozen,
        typevar_specs=typevar_specs,
        wrapped_classes=wrapped_classes,
        wrapped_functions=wrapped_functions,
        manual_reexports=tuple(manual_reexports),
    )


_NO_ASYNC_CONTEXT_MANAGER = object()


def _detect_async_context_manager_wrapper(f, return_annotation) -> typing.Any:
    """Detect functions wrapped by ``@asynccontextmanager``.

    Returns the yield type (context manager value type) if detected, or
    ``_NO_ASYNC_CONTEXT_MANAGER`` otherwise. ``None`` is a valid yield type.
    """
    if return_annotation == inspect.Signature.empty:
        return _NO_ASYNC_CONTEXT_MANAGER

    origin = typing.get_origin(return_annotation)
    if origin is not collections.abc.AsyncGenerator:
        return _NO_ASYNC_CONTEXT_MANAGER

    if inspect.isasyncgenfunction(f):
        return _NO_ASYNC_CONTEXT_MANAGER  # Actually is a raw async generator

    # Annotation says AsyncGenerator but function isn't one — check for @asynccontextmanager
    wrapped = getattr(f, "__wrapped__", None)
    while wrapped is not None:
        if inspect.isasyncgenfunction(wrapped):
            args = typing.get_args(return_annotation)
            if args:
                return args[0]
            return typing.Any
        wrapped = getattr(wrapped, "__wrapped__", None)

    return _NO_ASYNC_CONTEXT_MANAGER


def _normalize_return_annotation_ir(
    return_ir: AnnotationIR,
    *,
    is_async_gen: bool,
    func_qualname: str = "",
) -> AnnotationIR:
    if is_async_gen and isinstance(return_ir, AsyncIteratorAnnotationIR):
        raise TypeError(
            f"Async generator function {func_qualname!r} is declared as returning AsyncIterator, "
            f"but should use AsyncGenerator. AsyncIterator hides the generator interface "
            f"(.asend(), .athrow(), .aclose()) from callers."
        )
    return return_ir


def _parse_signature(
    f: types.FunctionType,
    *,
    annotations: dict[str, typing.Any],
    sig: inspect.Signature,
    impl_module: types.ModuleType,
    skip_first_param: bool,
    owner_impl_type: type | None,
    owner_has_type_parameters: bool,
    impl_modules: frozenset[str],
    synchronicity1_synchronizer: Synchronicity1Synchronizer | None = None,
) -> tuple[SignatureIR, bool]:
    source_label_prefix = f"{f.__module__}.{f.__qualname__}"
    return_annotation = annotations.get("return", sig.return_annotation)

    cm_value_type = _detect_async_context_manager_wrapper(f, return_annotation)
    if cm_value_type is not _NO_ASYNC_CONTEXT_MANAGER:
        value_ir = parse_annotation(
            cm_value_type,
            owner_impl_type=owner_impl_type,
            owner_has_type_parameters=owner_has_type_parameters,
            impl_modules=impl_modules,
            source_label=f"{source_label_prefix} return",
            synchronicity1_synchronizer=synchronicity1_synchronizer,
        )
        context_manager_return_ir = AsyncContextManagerAnnotationIR(value_ir=value_ir)
        parameters = parse_parameters_to_ir(
            f,
            sig,
            annotations,
            impl_module=impl_module,
            skip_first_param=skip_first_param,
            owner_impl_type=owner_impl_type,
            owner_has_type_parameters=owner_has_type_parameters,
            impl_modules=impl_modules,
            source_label_prefix=source_label_prefix,
            synchronicity1_synchronizer=synchronicity1_synchronizer,
        )
        return SignatureIR(parameters=parameters, return_annotation_ir=context_manager_return_ir), False

    if is_async_generator(f, return_annotation) and return_annotation == inspect.Signature.empty:
        return_annotation = collections.abc.AsyncGenerator[typing.Any, None]
    return_annotation = _normalize_async_annotation(f, return_annotation)

    parsed_return_ir = parse_annotation(
        return_annotation,
        owner_impl_type=owner_impl_type,
        owner_has_type_parameters=owner_has_type_parameters,
        impl_modules=impl_modules,
        source_label=f"{source_label_prefix} return",
        synchronicity1_synchronizer=synchronicity1_synchronizer,
    )
    parameters = parse_parameters_to_ir(
        f,
        sig,
        annotations,
        impl_module=impl_module,
        skip_first_param=skip_first_param,
        owner_impl_type=owner_impl_type,
        owner_has_type_parameters=owner_has_type_parameters,
        impl_modules=impl_modules,
        source_label_prefix=source_label_prefix,
        synchronicity1_synchronizer=synchronicity1_synchronizer,
    )

    is_async_gen = is_async_generator(f, return_annotation)
    parsed_return_ir = _normalize_return_annotation_ir(
        parsed_return_ir,
        is_async_gen=is_async_gen,
        func_qualname=f.__qualname__,
    )
    return SignatureIR(parameters=parameters, return_annotation_ir=parsed_return_ir), is_async_gen


def _parse_overload_signatures(
    f: types.FunctionType,
    *,
    globals_dict: dict[str, typing.Any] | None,
    skip_first_param: bool,
    owner_impl_type: type | None,
    owner_has_type_parameters: bool,
    impl_modules: frozenset[str],
    forbidden_wrapper_modules: frozenset[str] | None = None,
    synchronicity1_synchronizer: Synchronicity1Synchronizer | None = None,
) -> tuple[SignatureIR, ...]:
    overload_irs: list[SignatureIR] = []
    for overload_func in _iter_overload_functions(f):
        annotations = _safe_get_annotations(
            overload_func,
            globals_dict,
            forbidden_wrapper_modules=forbidden_wrapper_modules,
        )
        sig = inspect.signature(overload_func)
        overload_ir, _ = _parse_signature(
            overload_func,
            annotations=annotations,
            sig=sig,
            impl_module=sys.modules[overload_func.__module__],
            skip_first_param=skip_first_param,
            owner_impl_type=owner_impl_type,
            owner_has_type_parameters=owner_has_type_parameters,
            impl_modules=impl_modules,
            synchronicity1_synchronizer=synchronicity1_synchronizer,
        )
        overload_irs.append(overload_ir)
    return tuple(overload_irs)


def parse_wrapped_function(
    f: types.FunctionType,
    target_module: str,
    *,
    export_name: str | None = None,
    globals_dict: dict[str, typing.Any] | None = None,
    runtime_package: str = "synchronicity2",
    impl_modules: frozenset[str] | None = None,
    forbidden_wrapper_modules: frozenset[str] | None = None,
    synchronicity1_synchronizer: Synchronicity1Synchronizer | None = None,
) -> WrappedFunctionIR:
    _ = runtime_package  # reserved for parity with API; IR does not embed runtime package on nodes
    if impl_modules is None:
        impl_modules = frozenset({f.__module__})
    annotations = _safe_get_annotations(f, globals_dict, forbidden_wrapper_modules=forbidden_wrapper_modules)
    sig = inspect.signature(f)
    impl_module = sys.modules[f.__module__]
    signature_ir, is_async_gen = _parse_signature(
        f,
        annotations=annotations,
        sig=sig,
        impl_module=impl_module,
        skip_first_param=False,
        owner_impl_type=None,
        owner_has_type_parameters=False,
        impl_modules=impl_modules,
        synchronicity1_synchronizer=synchronicity1_synchronizer,
    )
    overloads = _parse_overload_signatures(
        f,
        globals_dict=globals_dict,
        skip_first_param=False,
        owner_impl_type=None,
        owner_has_type_parameters=False,
        impl_modules=impl_modules,
        forbidden_wrapper_modules=forbidden_wrapper_modules,
        synchronicity1_synchronizer=synchronicity1_synchronizer,
    )

    needs_async_wrapper = is_async_gen or isinstance(
        signature_ir.return_annotation_ir, (AwaitableAnnotationIR, CoroutineAnnotationIR)
    )

    return WrappedFunctionIR(
        impl_ref=ObjectReferenceIR(f.__module__, f.__qualname__),
        needs_async_wrapper=needs_async_wrapper,
        is_async_gen=is_async_gen,
        parameters=signature_ir.parameters,
        return_annotation_ir=signature_ir.return_annotation_ir,
        overloads=overloads,
        docstring=f.__doc__,
        export_name=export_name,
    )


def parse_wrapped_method(
    method: types.FunctionType,
    method_name: str,
    impl_class: type,
    *,
    owner_has_type_parameters: bool = False,
    method_type: MethodBindingKind = MethodBindingKind.INSTANCE,
    globals_dict: dict[str, typing.Any] | None = None,
    generic_typevars: dict[str, typing.TypeVar | typing.ParamSpec] | None = None,
    impl_modules: frozenset[str] | None = None,
    forbidden_wrapper_modules: frozenset[str] | None = None,
    synchronicity1_synchronizer: Synchronicity1Synchronizer | None = None,
) -> WrappedMethodIR:
    if impl_modules is None:
        impl_modules = frozenset({impl_class.__module__})
    annotations = _safe_get_annotations(method, globals_dict, forbidden_wrapper_modules=forbidden_wrapper_modules)
    sig = inspect.signature(method)
    impl_module = sys.modules[method.__module__]
    return_annotation = annotations.get("return", sig.return_annotation)

    # Validate __aiter__ isn't typed as a sync generator/iterator
    if method_name == "__aiter__":
        origin = typing.get_origin(return_annotation)
        if origin in (collections.abc.Generator, collections.abc.Iterator) or (
            inspect.isgeneratorfunction(method) and not inspect.isasyncgenfunction(method)
        ):
            raise TypeError(
                f"{impl_class.__module__}.{impl_class.__qualname__}.__aiter__ "
                "has a sync generator/iterator return type "
                f"but must return an async iterable (AsyncIterator, AsyncGenerator, etc.)."
            )

    skip_first_param = method_type in (MethodBindingKind.INSTANCE, MethodBindingKind.CLASSMETHOD)
    signature_ir, is_async_gen = _parse_signature(
        method,
        annotations=annotations,
        sig=sig,
        impl_module=impl_module,
        skip_first_param=skip_first_param,
        owner_impl_type=impl_class,
        owner_has_type_parameters=owner_has_type_parameters,
        impl_modules=impl_modules,
        synchronicity1_synchronizer=synchronicity1_synchronizer,
    )
    overloads = _parse_overload_signatures(
        method,
        globals_dict=globals_dict,
        skip_first_param=skip_first_param,
        owner_impl_type=impl_class,
        owner_has_type_parameters=owner_has_type_parameters,
        impl_modules=impl_modules,
        forbidden_wrapper_modules=forbidden_wrapper_modules,
        synchronicity1_synchronizer=synchronicity1_synchronizer,
    )

    is_async = is_async_gen or isinstance(
        signature_ir.return_annotation_ir, (AwaitableAnnotationIR, CoroutineAnnotationIR)
    )

    return WrappedMethodIR(
        method_name=method_name,
        method_type=method_type,
        parameters=signature_ir.parameters,
        is_async_gen=is_async_gen,
        is_async=is_async,
        return_annotation_ir=signature_ir.return_annotation_ir,
        overloads=overloads,
        docstring=method.__doc__,
    )


def parse_wrapped_class(
    cls: type,
    target_module: str,
    *,
    globals_dict: dict[str, typing.Any] | None = None,
    runtime_package: str = "synchronicity2",
    impl_modules: frozenset[str] | None = None,
    manual_wrapper_ids: frozenset[int] = frozenset(),
    include_underscored_methods: bool = False,
    forbidden_wrapper_modules: frozenset[str] | None = None,
    synchronicity1_synchronizer: Synchronicity1Synchronizer | None = None,
) -> WrappedClassIR:
    """Collect :class:`WrappedClassIR` from a live implementation class (parse-time only)."""
    if impl_modules is None:
        impl_modules = frozenset({cls.__module__})
    if globals_dict is None and cls.__module__ in sys.modules:
        globals_dict = sys.modules[cls.__module__].__dict__

    wrapped_bases: list[tuple[ObjectReferenceIR, ObjectReferenceIR]] = []
    generic_type_parameters: tuple[str, ...] | None = None
    generic_typevars: dict[str, typing.TypeVar | typing.ParamSpec] = {}

    bases_to_check = getattr(cls, "__orig_bases__", cls.__bases__)

    for base in bases_to_check:
        origin = typing.get_origin(base)
        if origin is not None and hasattr(origin, "__name__") and origin.__name__ == "Generic":
            args = typing.get_args(base)
            if args:
                for arg in args:
                    if isinstance(arg, typing.TypeVar) or isinstance(arg, typing.ParamSpec):
                        generic_typevars[arg.__name__] = arg

                typevar_names = [arg.__name__ for arg in args if isinstance(arg, (typing.TypeVar, typing.ParamSpec))]
                if typevar_names:
                    generic_type_parameters = tuple(typevar_names)
        elif base is not object and isinstance(base, type):
            wrapper_ref = _wrapper_ref_for_impl_type(base, synchronicity1_synchronizer)
            if wrapper_ref is not None:
                wrapped_bases.append((ObjectReferenceIR(base.__module__, base.__qualname__), wrapper_ref))

    # Collect all source methods: __init__, public methods, and async iterator dunders.
    source_methods: list[tuple[str, types.FunctionType, MethodBindingKind]] = []
    classmethod_staticmethod_names: set[str] = set()

    # __init__ (resolved through MRO — subclasses inherit a non-trivial __init__)
    init_method = getattr(cls, "__init__", None)
    if init_method and init_method is not object.__init__:
        source_methods.append(("__init__", typing.cast(types.FunctionType, init_method), MethodBindingKind.INSTANCE))

    # classmethods, staticmethods, and properties (descriptor unwrapping requires cls.__dict__)
    property_names: set[str] = set()
    classproperty_names: set[str] = set()
    manual_attribute_names: set[str] = set()
    property_irs: list[WrappedPropertyIR] = []
    classproperty_irs: list[WrappedClassPropertyIR] = []
    manual_attributes: list[ManualClassAttributeIR] = []
    for name, attr in tuple(cls.__dict__.items()):
        if _is_manual_wrapper(attr, manual_wrapper_ids=manual_wrapper_ids):
            if isinstance(attr, classmethod):
                source_methods.append(
                    (name, typing.cast(types.FunctionType, attr.__func__), MethodBindingKind.CLASSMETHOD)
                )
                classmethod_staticmethod_names.add(name)
                continue
            if isinstance(attr, staticmethod):
                source_methods.append(
                    (name, typing.cast(types.FunctionType, attr.__func__), MethodBindingKind.STATICMETHOD)
                )
                classmethod_staticmethod_names.add(name)
                continue
            if inspect.isfunction(attr):
                source_methods.append((name, attr, MethodBindingKind.INSTANCE))
                continue
            manual_attribute_names.add(name)
            manual_attributes.append(
                ManualClassAttributeIR(
                    name=name,
                    access_kind=_manual_class_attribute_access_kind(attr),
                )
            )
            continue
        if not _should_scan_class_member_name(name, include_underscored_methods=include_underscored_methods):
            continue
        if isinstance(attr, classmethod):
            source_methods.append((name, typing.cast(types.FunctionType, attr.__func__), MethodBindingKind.CLASSMETHOD))
            classmethod_staticmethod_names.add(name)
        elif isinstance(attr, staticmethod):
            if not _should_include_private_staticmethod(name, is_manual=False):
                classmethod_staticmethod_names.add(name)
                continue
            source_methods.append(
                (name, typing.cast(types.FunctionType, attr.__func__), MethodBindingKind.STATICMETHOD)
            )
            classmethod_staticmethod_names.add(name)
        elif isinstance(attr, property):
            if name.startswith("_") and include_underscored_methods:
                continue
            property_names.add(name)
            fget = attr.fget
            if fget is not None and inspect.iscoroutinefunction(fget):
                raise TypeError(
                    f"Property {cls.__qualname__}.{name} has an async getter. "
                    f"Properties must be synchronous; use an async method instead."
                )
            # Parse getter return type
            property_return_ir: AnnotationIR | None = None
            if fget is not None:
                getter_annotations = _safe_get_annotations(
                    fget, globals_dict, forbidden_wrapper_modules=forbidden_wrapper_modules
                )
                return_annotation = getter_annotations.get("return", inspect.Signature.empty)
                if return_annotation != inspect.Signature.empty:
                    property_return_ir = parse_annotation(
                        return_annotation,
                        owner_impl_type=cls,
                        owner_has_type_parameters=bool(generic_typevars),
                        impl_modules=impl_modules,
                        source_label=f"{cls.__module__}.{cls.__qualname__}.{name} return",
                        synchronicity1_synchronizer=synchronicity1_synchronizer,
                    )
            # Parse setter value type
            has_setter = attr.fset is not None
            setter_annotation_ir: AnnotationIR | None = None
            if has_setter and attr.fset is not None:
                setter_annotations = _safe_get_annotations(
                    attr.fset, globals_dict, forbidden_wrapper_modules=forbidden_wrapper_modules
                )
                setter_sig = inspect.signature(attr.fset)
                setter_params = list(setter_sig.parameters.values())
                if len(setter_params) >= 2:
                    value_param = setter_params[1]
                    setter_annotation = setter_annotations.get(value_param.name, inspect.Signature.empty)
                    if setter_annotation != inspect.Signature.empty:
                        setter_annotation_ir = parse_annotation(
                            setter_annotation,
                            owner_impl_type=cls,
                            owner_has_type_parameters=bool(generic_typevars),
                            impl_modules=impl_modules,
                            source_label=f"{cls.__module__}.{cls.__qualname__}.{name} parameter {value_param.name!r}",
                            synchronicity1_synchronizer=synchronicity1_synchronizer,
                        )
            property_irs.append(
                WrappedPropertyIR(
                    name=name,
                    return_annotation_ir=property_return_ir,
                    has_setter=has_setter,
                    setter_annotation_ir=setter_annotation_ir,
                )
            )
        elif _is_classproperty_descriptor(attr):
            if name.startswith("_") and include_underscored_methods:
                continue
            classproperty_names.add(name)
            fget = attr.fget.__func__
            if inspect.iscoroutinefunction(fget):
                raise TypeError(
                    f"Class property {cls.__qualname__}.{name} has an async getter. "
                    f"Class properties must be synchronous; use a classmethod instead."
                )
            getter_annotations = _safe_get_annotations(
                fget, globals_dict, forbidden_wrapper_modules=forbidden_wrapper_modules
            )
            return_annotation = getter_annotations.get("return", inspect.Signature.empty)
            class_property_return_ir: AnnotationIR | None = None
            if return_annotation != inspect.Signature.empty:
                class_property_return_ir = parse_annotation(
                    return_annotation,
                    owner_impl_type=cls,
                    owner_has_type_parameters=bool(generic_typevars),
                    impl_modules=impl_modules,
                    source_label=f"{cls.__module__}.{cls.__qualname__}.{name} return",
                    synchronicity1_synchronizer=synchronicity1_synchronizer,
                )
            classproperty_irs.append(
                WrappedClassPropertyIR(
                    name=name,
                    return_annotation_ir=class_property_return_ir,
                )
            )

    # instance methods (only directly defined on cls)
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if (
            (not name.startswith("_") or include_underscored_methods)
            and not _is_dunder_name(name)
            and name in cls.__dict__
            and name not in classmethod_staticmethod_names
            and name not in property_names
            and name not in classproperty_names
            and name not in manual_attribute_names
        ):
            source_methods.append((name, method, MethodBindingKind.INSTANCE))

    for dunder_name in _FORWARDED_DUNDER_METHODS:
        if dunder_name in cls.__dict__:
            source_methods.append((dunder_name, cls.__dict__[dunder_name], MethodBindingKind.INSTANCE))

    # async iterator protocol dunders (only directly defined on cls)
    for dunder_name in ("__aiter__", "__anext__"):
        if dunder_name in cls.__dict__:
            source_methods.append((dunder_name, cls.__dict__[dunder_name], MethodBindingKind.INSTANCE))

    # async context manager protocol dunders (only directly defined on cls)
    for dunder_name in ("__aenter__", "__aexit__"):
        if dunder_name in cls.__dict__:
            source_methods.append((dunder_name, cls.__dict__[dunder_name], MethodBindingKind.INSTANCE))

    attributes: list[tuple[str, AnnotationIR | None]] = []
    class_annotations = cls.__annotations__ if hasattr(cls, "__annotations__") else {}
    for name, annotation in class_annotations.items():
        if not name.startswith("_"):
            annotations_resolved = _safe_get_annotations(
                cls, globals_dict, forbidden_wrapper_modules=forbidden_wrapper_modules
            )
            resolved_annotation = annotations_resolved.get(name, annotation)
            annotation_ir = parse_annotation(
                resolved_annotation,
                owner_impl_type=cls,
                owner_has_type_parameters=bool(generic_typevars),
                impl_modules=impl_modules,
                source_label=f"{cls.__module__}.{cls.__qualname__} attribute {name!r}",
                synchronicity1_synchronizer=synchronicity1_synchronizer,
            )
            attributes.append((name, annotation_ir))

    # Parse all source methods into IR in one pass
    method_irs = tuple(
        parse_wrapped_method(
            method,
            method_name,
            cls,
            owner_has_type_parameters=bool(generic_typevars),
            method_type=method_type,
            globals_dict=globals_dict,
            generic_typevars=generic_typevars if generic_typevars else None,
            impl_modules=impl_modules,
            forbidden_wrapper_modules=forbidden_wrapper_modules,
            synchronicity1_synchronizer=synchronicity1_synchronizer,
        )
        for method_name, method, method_type in source_methods
    )

    # Read wrapper location from marker attribute
    wrapper_loc = _get_wrapper_location(cls)
    assert wrapper_loc is not None, f"{cls!r} missing {_IMPL_WRAPPER_LOCATION_ATTR}"
    wrapper_ref = ObjectReferenceIR(*wrapper_loc)

    return WrappedClassIR(
        impl_ref=ObjectReferenceIR(cls.__module__, cls.__qualname__),
        wrapper_ref=wrapper_ref,
        wrapped_bases=tuple(wrapped_bases),
        generic_type_parameters=generic_type_parameters,
        attributes=tuple(attributes),
        properties=tuple(property_irs),
        class_properties=tuple(classproperty_irs),
        methods=method_irs,
        manual_attributes=tuple(manual_attributes),
    )
