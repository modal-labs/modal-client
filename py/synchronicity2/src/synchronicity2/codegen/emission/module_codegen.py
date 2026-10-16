# Copyright Modal Labs 2026
"""Emit sync and async public wrapper modules from declaration IR."""

from __future__ import annotations

import dataclasses
import re
import textwrap

from ..ir.declarations import (
    ManualClassAttributeAccessKind,
    ManualReexportIR,
    MethodBindingKind,
    ModuleIR,
    SignatureIR,
    TypeParameterIR,
    WrappedClassIR,
    WrappedFunctionIR,
    WrappedMethodIR,
)
from ..ir.references import ObjectReferenceIR
from .imports import module_import_modules
from .signatures import (
    _build_call_with_wrap,
    _format_return_annotation,
    format_parameters_for_emit,
    quote_annotation_for_local_names,
)
from .type_codegen import (
    AwaitableTypeCodegen,
    CoroutineTypeCodegen,
    TypeCodegen,
    TypeCodegenContext,
    codegen_for_annotation,
)
from .type_parameters import type_parameter_definition_lines


@dataclasses.dataclass(frozen=True)
class MethodEmitOwner:
    """Class-wrapper context needed when emitting method wrappers (emitter-only)."""

    impl_ref: ObjectReferenceIR
    wrapper_name: str
    target_module: str
    generic_type_parameters: tuple[str, ...] | None


def _module_function_name(ir: WrappedFunctionIR) -> str:
    if ir.export_name is not None:
        return ir.export_name
    return ir.impl_ref.qualname.rpartition(".")[2]


def _wrapper_class_reference(
    wrapper: ObjectReferenceIR,
    target_module: str,
) -> str:
    """Resolve a wrapper ref to the identifier used on a generated class line."""

    if wrapper.module == target_module:
        return wrapper.qualname
    return f"{wrapper.module}.{wrapper.qualname}"


def _impl_type_dotted(impl_ref: ObjectReferenceIR) -> str:
    """Dotted path to the implementation type for emitted source.

    Uses ``module`` + ``.__qualname__`` when qualname is a normal attribute path. If
    ``__qualname__`` contains ``<locals>`` (nested in a function), only the last
    segment is appended so the reference matches the previous
    ``impl_ref.module`` + short-name emission (full qualname is not valid Python).
    """

    q = impl_ref.qualname
    if ".<locals>." in q or q.startswith("<locals>."):
        return f"{impl_ref.module}.{q.rpartition('.')[2]}"
    return f"{impl_ref.module}.{q}"


def _impl_value_dotted(impl_ref: ObjectReferenceIR) -> str:
    return _impl_type_dotted(impl_ref)


def method_emit_owner(class_ir: WrappedClassIR, target_module: str) -> MethodEmitOwner:
    return MethodEmitOwner(
        impl_ref=class_ir.impl_ref,
        wrapper_name=class_ir.wrapper_ref.qualname,
        target_module=target_module,
        generic_type_parameters=class_ir.generic_type_parameters,
    )


def _class_annotation_shadow_names(ir: WrappedClassIR) -> set[str]:
    return {
        *[name for name, _annotation in ir.attributes],
        *[prop.name for prop in ir.properties],
        *[prop.name for prop in ir.class_properties],
        *[attr.name for attr in ir.manual_attributes],
        *[method.method_name for method in ir.methods],
    }


def _type_codegen_context_for_module(specs: tuple[TypeParameterIR, ...]) -> TypeCodegenContext | None:
    if not specs:
        return None
    return TypeCodegenContext(type_parameters_by_name={s.name: s for s in specs})


_CLASS_INIT_NAME = "__init__"
_DESCRIPTOR_DUNDERS = frozenset({"__get__"})
_DYNAMIC_ATTR_DUNDERS = frozenset({"__getattr__"})
_ASYNC_ITERATOR_DUNDERS = frozenset({"__aiter__", "__anext__"})
_ASYNC_CONTEXT_MANAGER_DUNDERS = frozenset({"__aenter__", "__aexit__"})


def _partition_class_methods(
    methods: tuple[WrappedMethodIR, ...],
) -> tuple[
    WrappedMethodIR | None,
    tuple[WrappedMethodIR, ...],
    tuple[WrappedMethodIR, ...],
    tuple[WrappedMethodIR, ...],
    tuple[WrappedMethodIR, ...],
    tuple[WrappedMethodIR, ...],
]:
    """Split ``__init__``, descriptor/iterator/context-manager dunders, and everything else."""

    init_mir: WrappedMethodIR | None = None
    descriptor: list[WrappedMethodIR] = []
    dynamic_attr: list[WrappedMethodIR] = []
    iterator: list[WrappedMethodIR] = []
    context_manager: list[WrappedMethodIR] = []
    normal: list[WrappedMethodIR] = []
    descriptor_order = {"__get__": 0}
    dynamic_attr_order = {"__getattr__": 0}
    iter_order = {"__aiter__": 0, "__anext__": 1}
    cm_order = {"__aenter__": 0, "__aexit__": 1}
    for mir in methods:
        if mir.method_name == _CLASS_INIT_NAME:
            init_mir = mir
        elif mir.method_name in _DESCRIPTOR_DUNDERS:
            descriptor.append(mir)
        elif mir.method_name in _DYNAMIC_ATTR_DUNDERS:
            dynamic_attr.append(mir)
        elif mir.method_name in _ASYNC_ITERATOR_DUNDERS:
            iterator.append(mir)
        elif mir.method_name in _ASYNC_CONTEXT_MANAGER_DUNDERS:
            context_manager.append(mir)
        else:
            normal.append(mir)
    descriptor.sort(key=lambda m: descriptor_order[m.method_name])
    dynamic_attr.sort(key=lambda m: dynamic_attr_order[m.method_name])
    iterator.sort(key=lambda m: iter_order[m.method_name])
    context_manager.sort(key=lambda m: cm_order[m.method_name])
    return init_mir, tuple(descriptor), tuple(dynamic_attr), tuple(iterator), tuple(context_manager), tuple(normal)


def _async_iterator_dunder_pair(
    impl_dunder: str,
) -> tuple[str, str, bool, bool] | None:
    """Map impl ``__aiter__`` / ``__anext__`` to wrapper sync name, async name, ``async def`` flag, bridge."""
    if impl_dunder == "__aiter__":
        return ("__iter__", "__aiter__", False, False)
    if impl_dunder == "__anext__":
        return ("__next__", "__anext__", True, True)
    return None


def _async_context_manager_dunder_pair(
    impl_dunder: str,
) -> tuple[str, str] | None:
    """Map impl ``__aenter__`` / ``__aexit__`` to wrapper sync name, async name."""
    if impl_dunder == "__aenter__":
        return ("__enter__", "__aenter__")
    if impl_dunder == "__aexit__":
        return ("__exit__", "__aexit__")
    return None


def _method_impl_call_expr(
    method_type: MethodBindingKind,
    impl_class_dotted: str,
    method_name: str,
    call_args_str: str,
) -> str:
    callable_expr = f"{impl_class_dotted}.{method_name}"
    if method_type == MethodBindingKind.INSTANCE:
        args = (
            f"wrapper_instance._impl_instance, {call_args_str}" if call_args_str else "wrapper_instance._impl_instance"
        )
        return f"{callable_expr}({args})"
    if method_type in (MethodBindingKind.CLASSMETHOD, MethodBindingKind.STATICMETHOD):
        return f"{callable_expr}({call_args_str})" if call_args_str else f"{callable_expr}()"
    args = f"wrapper_instance._impl_instance, {call_args_str}" if call_args_str else "wrapper_instance._impl_instance"
    return f"{callable_expr}({args})"


def _runtime_import_header(runtime_package: str) -> str:
    return f"""import {runtime_package}.types
from {runtime_package}.descriptor import (
    FunctionWithAio,
    MethodWithAio,
    classproperty,
    classmethod_with_aio,
    function_with_aio,
    method_with_aio,
    staticmethod_with_aio,
)
from {runtime_package}.synchronizer import _wrapped_from_impl
"""


_GENERATED_MODULE_BANNER = """# Generated by synchronicity2.
# This module should not be modified directly.
"""


def _wrapper_registration_lines(wrapped_classes: tuple[WrappedClassIR, ...]) -> list[str]:
    lines: list[str] = []
    for ir in wrapped_classes:
        impl_dot = _impl_type_dotted(ir.impl_ref)
        wshort = ir.wrapper_ref.qualname
        lines.append(f"_synchronizer.register_wrapper_class({impl_dot}, {wshort})")
    return lines


def _default_import_modules_for_signatures(signatures: tuple[SignatureIR, ...]) -> set[str]:
    modules: set[str] = set()
    for signature in signatures:
        for parameter in signature.parameters:
            modules.update(parameter.default_import_modules)
    return modules


def _default_import_modules_for_module(ir: ModuleIR) -> set[str]:
    modules: set[str] = set()
    for class_ir in ir.wrapped_classes:
        for method_ir in class_ir.methods:
            modules.update(
                _default_import_modules_for_signatures(
                    method_ir.overloads or (SignatureIR(method_ir.parameters, method_ir.return_annotation_ir),)
                )
            )
    for function_ir in ir.wrapped_functions:
        modules.update(
            _default_import_modules_for_signatures(
                function_ir.overloads or (SignatureIR(function_ir.parameters, function_ir.return_annotation_ir),)
            )
        )
    return modules


def emit_manual_reexport(ir: ManualReexportIR) -> str:
    return f"{ir.export_name} = {_impl_value_dotted(ir.impl_ref)}"


def _emit_module_function_overloads(
    overloads: tuple[SignatureIR, ...],
    function_name: str,
    target_module: str,
    runtime_package: str,
    *,
    type_codegen_context: TypeCodegenContext | None,
    is_async: bool,
) -> str:
    blocks: list[str] = []
    for overload in overloads:
        param_str, _, _ = format_parameters_for_emit(
            overload.parameters,
            target_module,
            runtime_package,
            type_codegen_context=type_codegen_context,
        )
        return_codegen = codegen_for_annotation(
            overload.return_annotation_ir, runtime_package, context=type_codegen_context
        )
        sync_return_str, async_return_str = _format_return_annotation(return_codegen, target_module)
        return_str = async_return_str if is_async else sync_return_str
        async_prefix = "async " if is_async else ""
        blocks.append(f"@typing.overload\n{async_prefix}def {function_name}({param_str}){return_str}: ...")
    return "\n".join(blocks)


def _method_signature_line(
    method_name: str,
    method_type: MethodBindingKind,
    param_str: str,
    return_str: str,
    *,
    is_async: bool,
) -> str:
    async_prefix = "async " if is_async else ""
    if method_type == MethodBindingKind.INSTANCE:
        all_params = "self" if not param_str else f"self, {param_str}"
    elif method_type == MethodBindingKind.CLASSMETHOD:
        all_params = "cls" if not param_str else f"cls, {param_str}"
    else:
        all_params = param_str
    return f"{async_prefix}def {method_name}({all_params}){return_str}: ..."


def _emit_method_overloads(
    overloads: tuple[SignatureIR, ...],
    method_name: str,
    method_type: MethodBindingKind,
    target_module: str,
    runtime_package: str,
    *,
    type_codegen_context: TypeCodegenContext | None,
    is_async: bool,
    annotation_local_names: set[str] | None = None,
) -> str:
    blocks: list[str] = []
    for overload in overloads:
        param_str, _, _ = format_parameters_for_emit(
            overload.parameters,
            target_module,
            runtime_package,
            type_codegen_context=type_codegen_context,
            annotation_local_names=annotation_local_names,
        )
        return_codegen = codegen_for_annotation(
            overload.return_annotation_ir, runtime_package, context=type_codegen_context
        )
        sync_return_str, async_return_str = _format_return_annotation(
            return_codegen,
            target_module,
            annotation_local_names=annotation_local_names,
        )
        return_str = async_return_str if is_async else sync_return_str
        lines = ["    @typing.overload"]
        if method_type == MethodBindingKind.CLASSMETHOD:
            lines.append("    @classmethod")
        elif method_type == MethodBindingKind.STATICMETHOD:
            lines.append("    @staticmethod")
        lines.append(
            "    "
            + _method_signature_line(
                method_name,
                method_type,
                param_str,
                return_str,
                is_async=is_async,
            )
        )
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _method_with_aio_protocol_name(owner: MethodEmitOwner, method_name: str) -> str:
    if owner.wrapper_name.startswith("_"):
        owner_name = owner.wrapper_name.lstrip("_") or owner.wrapper_name
        return f"_synchronicity_{owner_name}_{method_name}_MethodWithAio"
    return f"_{owner.wrapper_name}_{method_name}_MethodWithAio"


def _method_with_aio_self_type_name(owner: MethodEmitOwner, method_name: str) -> str:
    if owner.wrapper_name.startswith("_"):
        owner_name = owner.wrapper_name.lstrip("_") or owner.wrapper_name
        return f"_synchronicity_{owner_name}_{method_name}_SelfType"
    return f"_{owner.wrapper_name}_{method_name}_SelfType"


def _bound_with_aio_method_signature_line(
    method_name: str,
    param_str: str,
    return_str: str,
    *,
    is_async: bool = False,
) -> str:
    all_params = "self" if not param_str else f"self, {param_str}"
    prefix = "async def" if is_async else "def"
    return f"{prefix} {method_name}({all_params}){return_str}: ..."


def _bound_with_aio_method_definition_line(
    method_name: str,
    param_str: str,
    return_str: str,
    *,
    is_async: bool = False,
) -> str:
    all_params = "self" if not param_str else f"self, {param_str}"
    prefix = "async def" if is_async else "def"
    return f"{prefix} {method_name}({all_params}){return_str}:"


def _with_aio_call_args_str(parameters: tuple) -> str:
    call_args: list[str] = []
    for param_ir in parameters:
        name = param_ir.name
        kind = param_ir.kind
        if kind == 2:
            call_args.append(f"*{name}")
        elif kind == 4:
            call_args.append(f"**{name}")
        elif kind == 3:
            call_args.append(f"{name}={name}")
        else:
            call_args.append(name)
    return ", ".join(call_args)


def _strip_return_annotation(return_str: str) -> str:
    if not return_str:
        return ""
    prefix = " -> "
    if return_str.startswith(prefix):
        return return_str[len(prefix) :]
    return return_str


def _with_aio_return_str(async_return_str: str) -> str:
    async_return_type = _strip_return_annotation(async_return_str)
    if not async_return_type:
        return ""
    return f" -> {async_return_type}"


def _indent_block(block: str, indent: str = "    ") -> str:
    return "\n".join(f"{indent}{line}" if line else "" for line in block.split("\n"))


def _emit_docstring_statement(docstring: str | None, *, indent: str) -> str:
    if docstring is None:
        return ""
    delimiter = '"""'
    if delimiter in docstring and "'''" not in docstring:
        delimiter = "'''"
    escaped_docstring = docstring.replace("\\", "\\\\").replace(delimiter, f"\\{delimiter}")
    return f"{indent}{delimiter}{escaped_docstring}{delimiter}"


def _prepend_docstring(body: str, docstring: str | None, *, indent: str) -> str:
    if docstring is None:
        return body
    docstring_line = _emit_docstring_statement(docstring, indent=indent)
    if not body:
        return docstring_line
    return f"{docstring_line}\n{body}"


def _rewrite_with_aio_annotation_text(text: str, self_type_name: str | None) -> str:
    if self_type_name is None:
        return text
    return text.replace('"typing_extensions.Self"', self_type_name).replace("typing_extensions.Self", self_type_name)


def _candidate_type_parameters(
    *,
    preferred_names: tuple[str, ...] = (),
    type_codegen_context: TypeCodegenContext | None,
) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for name in preferred_names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    if type_codegen_context is not None and type_codegen_context.type_parameters_by_name is not None:
        for name in type_codegen_context.type_parameters_by_name:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def _used_type_parameters(
    candidate_names: tuple[str, ...],
    signature_text: str,
) -> tuple[str, ...]:
    used: list[str] = []
    for type_param in candidate_names:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(type_param)}(?![A-Za-z0-9_])", signature_text):
            used.append(type_param)
    return tuple(used)


def _method_with_aio_signature_variants(
    owner: MethodEmitOwner,
    mir: WrappedMethodIR,
    runtime_package: str,
    *,
    type_codegen_context: TypeCodegenContext | None,
) -> tuple[list[tuple[str, str, str]], tuple[str, ...], bool]:
    signatures = mir.overloads or (SignatureIR(mir.parameters, mir.return_annotation_ir),)
    rendered: list[tuple[str, str, str]] = []
    all_signature_text = ""
    for signature in signatures:
        param_str, _, _ = format_parameters_for_emit(
            signature.parameters,
            owner.target_module,
            runtime_package,
            type_codegen_context=type_codegen_context,
        )
        return_codegen = codegen_for_annotation(
            signature.return_annotation_ir, runtime_package, context=type_codegen_context
        )
        sync_return_str, async_return_str = _format_return_annotation(return_codegen, owner.target_module)
        rendered.append((param_str, sync_return_str, async_return_str))
        all_signature_text += param_str + sync_return_str + async_return_str

    needs_self_type = "typing_extensions.Self" in all_signature_text
    self_type_name = _method_with_aio_self_type_name(owner, mir.method_name) if needs_self_type else None
    used_type_parameters = list(
        _used_type_parameters(
            _candidate_type_parameters(
                preferred_names=owner.generic_type_parameters or (),
                type_codegen_context=type_codegen_context,
            ),
            all_signature_text,
        )
    )
    if self_type_name is not None:
        used_type_parameters.append(self_type_name)
    with_aio_variants: list[tuple[str, str, str]] = []
    for param_str, sync_return_str, async_return_str in rendered:
        rewritten_param_str = _rewrite_with_aio_annotation_text(param_str, self_type_name)
        rewritten_sync_return_str = _rewrite_with_aio_annotation_text(sync_return_str, self_type_name)
        rewritten_async_return_type = _rewrite_with_aio_annotation_text(
            _strip_return_annotation(async_return_str),
            self_type_name,
        )
        aio_return_str = f" -> {rewritten_async_return_type}" if rewritten_async_return_type else ""
        with_aio_variants.append((rewritten_param_str, rewritten_sync_return_str, aio_return_str))

    return with_aio_variants, tuple(used_type_parameters), needs_self_type


def _function_with_aio_protocol_name(function_name: str) -> str:
    return f"_{function_name}_FunctionWithAio"


def _function_with_aio_signature_variants(
    ir: WrappedFunctionIR,
    target_module: str,
    runtime_package: str,
    *,
    type_codegen_context: TypeCodegenContext | None,
) -> tuple[list[tuple[str, str, str]], tuple[str, ...]]:
    signatures = ir.overloads or (SignatureIR(ir.parameters, ir.return_annotation_ir),)
    rendered: list[tuple[str, str, str]] = []
    all_signature_text = ""
    for signature in signatures:
        param_str, _, _ = format_parameters_for_emit(
            signature.parameters,
            target_module,
            runtime_package,
            type_codegen_context=type_codegen_context,
        )
        return_codegen = codegen_for_annotation(
            signature.return_annotation_ir, runtime_package, context=type_codegen_context
        )
        sync_return_str, async_return_str = _format_return_annotation(return_codegen, target_module)
        rendered.append((param_str, sync_return_str, async_return_str))
        all_signature_text += param_str + sync_return_str + async_return_str

    used_type_parameters = _used_type_parameters(
        _candidate_type_parameters(type_codegen_context=type_codegen_context),
        all_signature_text,
    )

    with_aio_variants: list[tuple[str, str, str]] = []
    for param_str, sync_return_str, async_return_str in rendered:
        async_return_type = _strip_return_annotation(async_return_str)
        aio_return_str = f" -> {async_return_type}" if async_return_type else ""
        with_aio_variants.append((param_str, sync_return_str, aio_return_str))

    return with_aio_variants, used_type_parameters


def emit_function_with_aio_protocol(
    ir: WrappedFunctionIR,
    target_module: str,
    runtime_package: str,
    *,
    type_codegen_context: TypeCodegenContext | None,
) -> str:
    variants, generic_parameters = _function_with_aio_signature_variants(
        ir,
        target_module,
        runtime_package,
        type_codegen_context=type_codegen_context,
    )
    function_name = _module_function_name(ir)
    with_aio_name = _function_with_aio_protocol_name(function_name)
    return_codegen = codegen_for_annotation(ir.return_annotation_ir, runtime_package, context=type_codegen_context)
    param_str, call_args_str, unwrap_code = format_parameters_for_emit(
        ir.parameters,
        target_module,
        runtime_package,
        unwrap_indent="",
        type_codegen_context=type_codegen_context,
    )
    with_aio_call_args_str = _with_aio_call_args_str(ir.parameters)
    sync_return_str, async_return_str = _format_return_annotation(return_codegen, target_module)
    aio_return_str = _with_aio_return_str(async_return_str)
    impl_dotted = _impl_value_dotted(ir.impl_ref)
    function_call_expr = f"{impl_dotted}({call_args_str})" if call_args_str else f"{impl_dotted}()"

    helpers_dict = return_codegen.helper_definitions(target_module, indent="    ")
    if isinstance(return_codegen, (AwaitableTypeCodegen, CoroutineTypeCodegen)):
        helpers_dict.update(return_codegen.return_codegen.helper_definitions(target_module, indent="    "))
    helpers_code = "\n".join(helpers_dict.values()) if helpers_dict else ""

    if ir.is_async_gen:
        # TODO: Migrate away from raw async-generator wrappers requiring `.aio()`.
        # This is inconsistent with wrappers for callables returning AsyncGenerator,
        # where the wrapper result is already directly async-iterable.
        aio_body = ""
        if unwrap_code:
            aio_body = unwrap_code + "\n"
        aio_body += (
            f"gen = {function_call_expr}\n"
            f"_wrapped = {return_codegen.impl_to_wrapper_expr(target_module, 'gen')}\n"
            f"_sent = None\n"
            f"try:\n"
            f"    while True:\n"
            f"        try:\n"
            f"            _item = await _wrapped.asend(_sent)\n"
            f"            _sent = yield _item\n"
            f"        except StopAsyncIteration:\n"
            f"            break\n"
            f"finally:\n"
            f"    await _wrapped.aclose()"
        )
    else:
        aio_body = ""
        if unwrap_code:
            aio_body = unwrap_code + "\n"
        aio_body += _build_call_with_wrap(
            function_call_expr,
            return_codegen,
            target_module,
            indent="",
            is_async=True,
            is_function=False,
        )

    if generic_parameters:
        header = f"class {with_aio_name}(FunctionWithAio, typing.Generic[{', '.join(generic_parameters)}]):"
    else:
        header = f"class {with_aio_name}(FunctionWithAio):"

    body_lines: list[str] = []
    if ir.overloads:
        for overload_param_str, overload_sync_return_str, _overload_aio_return_str in variants:
            body_lines.append("    @typing.overload")
            body_lines.append(
                "    " + _bound_with_aio_method_signature_line("__call__", overload_param_str, overload_sync_return_str)
            )
        body_lines.append("")
        for overload_param_str, _overload_sync_return_str, overload_aio_return_str in variants:
            body_lines.append("    @typing.overload")
            body_lines.append(
                "    "
                + _bound_with_aio_method_signature_line(
                    "aio",
                    overload_param_str,
                    overload_aio_return_str,
                    is_async=True,
                )
            )
    body_lines.append("")
    body_lines.append("    " + _bound_with_aio_method_definition_line("__call__", param_str, sync_return_str))
    if ir.docstring is not None:
        body_lines.append(_emit_docstring_statement(ir.docstring, indent="        "))
    body_lines.append(f"        return self._sync_impl({with_aio_call_args_str})")
    body_lines.append("")
    body_lines.append("    " + _bound_with_aio_method_definition_line("aio", param_str, aio_return_str, is_async=True))
    if ir.docstring is not None:
        body_lines.append(_emit_docstring_statement(ir.docstring, indent="        "))
    body_lines.append(_indent_block(textwrap.dedent(aio_body).rstrip(), "        "))
    if helpers_code:
        body_lines.append("")
        body_lines.append(helpers_code)
    return f"{header}\n" + "\n".join(body_lines)


def _function_with_aio_type_expr(
    ir: WrappedFunctionIR,
    target_module: str,
    runtime_package: str,
    *,
    type_codegen_context: TypeCodegenContext | None,
) -> str:
    _variants, generic_parameters = _function_with_aio_signature_variants(
        ir,
        target_module,
        runtime_package,
        type_codegen_context=type_codegen_context,
    )
    protocol_name = _function_with_aio_protocol_name(_module_function_name(ir))
    if not generic_parameters:
        return protocol_name
    return f"{protocol_name}[{', '.join(generic_parameters)}]"


def emit_method_with_aio_protocol(
    owner: MethodEmitOwner,
    mir: WrappedMethodIR,
    runtime_package: str,
    *,
    type_codegen_context: TypeCodegenContext | None,
) -> str:
    variants, generic_parameters, _needs_self_type = _method_with_aio_signature_variants(
        owner,
        mir,
        runtime_package,
        type_codegen_context=type_codegen_context,
    )
    supported_generic_parameters = _supported_method_with_aio_parameters(owner, mir.method_name, generic_parameters)
    with_aio_name = _method_with_aio_protocol_name(owner, mir.method_name)
    current_target_module = owner.target_module
    self_type_name = _method_with_aio_self_type_name(owner, mir.method_name) if _needs_self_type else None
    return_codegen = codegen_for_annotation(mir.return_annotation_ir, runtime_package, context=type_codegen_context)
    param_str, call_args_str, unwrap_code = format_parameters_for_emit(
        mir.parameters,
        current_target_module,
        runtime_package,
        unwrap_indent="",
        type_codegen_context=type_codegen_context,
    )
    with_aio_call_args_str = _with_aio_call_args_str(mir.parameters)
    sync_return_str, async_return_str = _format_return_annotation(return_codegen, current_target_module)
    aio_return_str = _with_aio_return_str(async_return_str)
    param_str = _rewrite_with_aio_annotation_text(param_str, self_type_name)
    sync_return_str = _rewrite_with_aio_annotation_text(sync_return_str, self_type_name)
    aio_return_str = _rewrite_with_aio_annotation_text(aio_return_str, self_type_name)
    impl_dotted = _impl_type_dotted(owner.impl_ref)
    call_expr_prefix = _method_impl_call_expr(
        mir.method_type,
        impl_dotted,
        mir.method_name,
        call_args_str,
    )

    helpers_dict = return_codegen.helper_definitions(current_target_module, indent="    ")
    if isinstance(return_codegen, (AwaitableTypeCodegen, CoroutineTypeCodegen)):
        helpers_dict.update(return_codegen.return_codegen.helper_definitions(current_target_module, indent="    "))
    helpers_code = "\n".join(helpers_dict.values()) if helpers_dict else ""

    aio_body: str
    if mir.method_type == MethodBindingKind.INSTANCE:
        if mir.is_async_gen:
            # TODO: Migrate away from raw async-generator method wrappers requiring `.aio()`.
            # This is inconsistent with wrappers for methods returning AsyncGenerator,
            # where the wrapper result is already directly async-iterable.
            impl_to_wrapper_expr = return_codegen.impl_to_wrapper_expr(current_target_module, "gen")
            aio_body = ""
            if unwrap_code:
                aio_body = unwrap_code + "\n"
            aio_body += (
                f"gen = {call_expr_prefix}\n"
                f"_wrapped = {impl_to_wrapper_expr}\n"
                f"_sent = None\n"
                f"try:\n"
                f"    while True:\n"
                f"        try:\n"
                f"            _item = await _wrapped.asend(_sent)\n"
                f"            _sent = yield _item\n"
                f"        except StopAsyncIteration:\n"
                f"            break\n"
                f"finally:\n"
                f"    await _wrapped.aclose()"
            )
            aio_body = aio_body.replace("wrapper_instance", "self._wrapper_instance")
        else:
            aio_body = _build_call_with_wrap(
                call_expr_prefix,
                return_codegen,
                current_target_module,
                indent="",
                is_async=True,
                method_type=mir.method_type,
                method_owner_impl_ref=owner.impl_ref,
            )
            if unwrap_code:
                aio_body = unwrap_code + "\n" + aio_body
            aio_body = aio_body.replace("wrapper_instance", "self._wrapper_instance")
    elif mir.method_type == MethodBindingKind.CLASSMETHOD:
        if mir.is_async_gen:
            impl_to_wrapper_expr = return_codegen.impl_to_wrapper_expr(current_target_module, "gen")
            aio_body = ""
            if unwrap_code:
                aio_body = unwrap_code + "\n"
            aio_body += (
                f"gen = {call_expr_prefix}\n"
                f"_wrapped = {impl_to_wrapper_expr}\n"
                f"_sent = None\n"
                f"try:\n"
                f"    while True:\n"
                f"        try:\n"
                f"            _item = await _wrapped.asend(_sent)\n"
                f"            _sent = yield _item\n"
                f"        except StopAsyncIteration:\n"
                f"            break\n"
                f"finally:\n"
                f"    await _wrapped.aclose()"
            )
            aio_body = aio_body.replace("wrapper_class", "self._wrapper_class")
        else:
            aio_body = _build_call_with_wrap(
                call_expr_prefix,
                return_codegen,
                current_target_module,
                indent="",
                is_async=True,
                method_type=mir.method_type,
                method_owner_impl_ref=owner.impl_ref,
            )
            if unwrap_code:
                aio_body = unwrap_code + "\n" + aio_body
        aio_body = aio_body.replace("wrapper_class", "self._wrapper_class").replace("cls.", "self._wrapper_class.")
    else:
        if mir.is_async_gen:
            impl_to_wrapper_expr = return_codegen.impl_to_wrapper_expr(current_target_module, "gen")
            aio_body = ""
            if unwrap_code:
                aio_body = unwrap_code + "\n"
            aio_body += (
                f"gen = {call_expr_prefix}\n"
                f"_wrapped = {impl_to_wrapper_expr}\n"
                f"_sent = None\n"
                f"try:\n"
                f"    while True:\n"
                f"        try:\n"
                f"            _item = await _wrapped.asend(_sent)\n"
                f"            _sent = yield _item\n"
                f"        except StopAsyncIteration:\n"
                f"            break\n"
                f"finally:\n"
                f"    await _wrapped.aclose()"
            )
        else:
            aio_body = _build_call_with_wrap(
                call_expr_prefix,
                return_codegen,
                current_target_module,
                indent="",
                is_async=True,
                method_type=mir.method_type,
                method_owner_impl_ref=owner.impl_ref,
            )
            if unwrap_code:
                aio_body = unwrap_code + "\n" + aio_body

    aio_body = _rewrite_with_aio_annotation_text(aio_body, self_type_name)

    if supported_generic_parameters:
        header = f"class {with_aio_name}(MethodWithAio, typing.Generic[{', '.join(supported_generic_parameters)}]):"
    else:
        header = f"class {with_aio_name}(MethodWithAio):"

    body_lines: list[str] = []
    if mir.overloads:
        for overload_param_str, overload_sync_return_str, _overload_aio_return_str in variants:
            body_lines.append("    @typing.overload")
            body_lines.append(
                "    " + _bound_with_aio_method_signature_line("__call__", overload_param_str, overload_sync_return_str)
            )
        body_lines.append("")
        for overload_param_str, _overload_sync_return_str, overload_aio_return_str in variants:
            body_lines.append("    @typing.overload")
            body_lines.append(
                "    "
                + _bound_with_aio_method_signature_line(
                    "aio",
                    overload_param_str,
                    overload_aio_return_str,
                    is_async=True,
                )
            )
        body_lines.append("")
    body_lines.append("    " + _bound_with_aio_method_definition_line("__call__", param_str, sync_return_str))
    if mir.docstring is not None:
        body_lines.append(_emit_docstring_statement(mir.docstring, indent="        "))
    body_lines.append(f"        return self._sync_impl({with_aio_call_args_str})")
    body_lines.append("")
    body_lines.append("    " + _bound_with_aio_method_definition_line("aio", param_str, aio_return_str, is_async=True))
    if mir.docstring is not None:
        body_lines.append(_emit_docstring_statement(mir.docstring, indent="        "))
    body_lines.append(_indent_block(textwrap.dedent(aio_body).rstrip(), "        "))
    if helpers_code:
        body_lines.append("")
        body_lines.append(helpers_code)
    prelude = ""
    if self_type_name is not None and self_type_name in supported_generic_parameters:
        prelude = f'{self_type_name} = typing.TypeVar("{self_type_name}", bound="{owner.wrapper_name}")\n\n'

    return prelude + f"{header}\n" + "\n".join(body_lines)


def _method_with_aio_type_data(
    owner: MethodEmitOwner,
    mir: WrappedMethodIR,
    runtime_package: str,
    *,
    type_codegen_context: TypeCodegenContext | None,
) -> tuple[str, bool]:
    _variants, generic_parameters, needs_self_type = _method_with_aio_signature_variants(
        owner,
        mir,
        runtime_package,
        type_codegen_context=type_codegen_context,
    )
    protocol_name = _method_with_aio_protocol_name(owner, mir.method_name)
    supported_parameters = _supported_method_with_aio_parameters(owner, mir.method_name, generic_parameters)
    if not supported_parameters:
        return protocol_name, bool(generic_parameters)
    args: list[str] = []
    for parameter in supported_parameters:
        if needs_self_type and parameter == _method_with_aio_self_type_name(owner, mir.method_name):
            args.append("typing_extensions.Self")
        else:
            args.append(parameter)
    return f"{protocol_name}[{', '.join(args)}]", len(supported_parameters) != len(generic_parameters)


def _supported_method_with_aio_parameters(
    owner: MethodEmitOwner,
    method_name: str,
    generic_parameters: tuple[str, ...],
) -> tuple[str, ...]:
    owner_type_parameters = set(owner.generic_type_parameters or ())
    self_type_name = _method_with_aio_self_type_name(owner, method_name)
    return tuple(
        parameter
        for parameter in generic_parameters
        if parameter in owner_type_parameters or parameter == self_type_name
    )


def emit_wrapped_function(
    ir: WrappedFunctionIR,
    target_module: str,
    runtime_package: str = "synchronicity2",
    *,
    type_codegen_context: TypeCodegenContext | None = None,
) -> str:
    current_target_module = target_module
    return_codegen = codegen_for_annotation(ir.return_annotation_ir, runtime_package, context=type_codegen_context)
    param_str, call_args_str, unwrap_code = format_parameters_for_emit(
        ir.parameters,
        current_target_module,
        runtime_package,
        unwrap_indent="    ",
        type_codegen_context=type_codegen_context,
    )
    is_async_gen = ir.is_async_gen
    f = _module_function_name(ir)
    impl_dotted = _impl_value_dotted(ir.impl_ref)
    function_call_expr = f"{impl_dotted}({call_args_str})" if call_args_str else f"{impl_dotted}()"

    if not ir.needs_async_wrapper:
        sync_return_str, _ = _format_return_annotation(return_codegen, current_target_module)

        inline_helpers_dict = return_codegen.helper_definitions(current_target_module, indent="")
        if inline_helpers_dict:
            cleaned_helpers = {}
            for name, helper_code in inline_helpers_dict.items():
                lines = helper_code.split("\n")
                cleaned_lines = []
                for line in lines:
                    if line.strip().startswith("@staticmethod"):
                        continue
                    cleaned_lines.append(line)
                cleaned_helpers[name] = "\n".join(cleaned_lines)
            helpers_code = "\n".join(cleaned_helpers.values())
        else:
            helpers_code = ""

        function_body = _build_call_with_wrap(
            function_call_expr,
            return_codegen,
            current_target_module,
            indent="    ",
            is_async=False,
            is_function=True,
        )

        if unwrap_code:
            function_body = f"{unwrap_code}\n{function_body}"
        function_body = _prepend_docstring(function_body, ir.docstring, indent="    ")

        function_code = f"""def {f}({param_str}){sync_return_str}:
{function_body}"""

        overload_code = _emit_module_function_overloads(
            ir.overloads,
            f,
            current_target_module,
            runtime_package,
            type_codegen_context=type_codegen_context,
            is_async=False,
        )
        if overload_code:
            function_code = f"{overload_code}\n{function_code}"

        if helpers_code:
            return f"{helpers_code}\n\n{function_code}"
        return function_code

    sync_return_str, async_return_str = _format_return_annotation(return_codegen, current_target_module)

    inline_helpers_dict = return_codegen.helper_definitions(current_target_module, indent="    ")

    if isinstance(return_codegen, (AwaitableTypeCodegen, CoroutineTypeCodegen)):
        inner_helpers = return_codegen.return_codegen.helper_definitions(current_target_module, indent="    ")
        inline_helpers_dict.update(inner_helpers)
    if inline_helpers_dict:
        cleaned_helpers = {}
        for name, helper_code in inline_helpers_dict.items():
            lines = helper_code.split("\n")
            cleaned_lines = []
            for line in lines:
                if line.strip().startswith("@staticmethod"):
                    continue
                if line.startswith("    "):
                    cleaned_lines.append(line[4:])
                else:
                    cleaned_lines.append(line)
            cleaned_helpers[name] = "\n".join(cleaned_lines)
        helpers_code = "\n".join(cleaned_helpers.values())
    else:
        helpers_code = ""

    sync_unwrap_section = unwrap_code

    if is_async_gen:
        sync_impl_to_wrapper_expr_raw = return_codegen.impl_to_wrapper_expr(
            current_target_module, "gen", is_async=False
        )
        sync_impl_to_wrapper_expr = sync_impl_to_wrapper_expr_raw.replace("self.", "")
        sync_function_body = f"    gen = {function_call_expr}\n    yield from {sync_impl_to_wrapper_expr}"
    else:
        sync_function_body = _build_call_with_wrap(
            function_call_expr,
            return_codegen,
            current_target_module,
            indent="    ",
            is_async=False,
            is_function=True,
        )

    with_aio_type_expr = _function_with_aio_type_expr(
        ir,
        current_target_module,
        runtime_package,
        type_codegen_context=type_codegen_context,
    )
    decorator_line = f"@function_with_aio({with_aio_type_expr})"
    sync_body = f"{sync_unwrap_section}\n{sync_function_body}" if sync_unwrap_section else sync_function_body

    sync_function_code = f"""{decorator_line}
def {f}({param_str}){sync_return_str}:
{_prepend_docstring(sync_body, ir.docstring, indent="    ")}
"""
    with_aio_code = emit_function_with_aio_protocol(
        ir, current_target_module, runtime_package, type_codegen_context=type_codegen_context
    )
    if helpers_code:
        return f"{helpers_code}\n\n{with_aio_code}\n\n{sync_function_code}"
    return f"{with_aio_code}\n\n{sync_function_code}"


def emit_wrapped_method_pair(
    owner: MethodEmitOwner,
    mir: WrappedMethodIR,
    *,
    runtime_package: str = "synchronicity2",
    type_codegen_context: TypeCodegenContext | None = None,
    return_codegen: TypeCodegen | None = None,
    annotation_local_names: set[str] | None = None,
) -> tuple[str, str]:
    method_name = mir.method_name
    method_type = mir.method_type
    impl_dotted = _impl_type_dotted(owner.impl_ref)
    short_name = owner.wrapper_name
    current_target_module = owner.target_module
    if return_codegen is None:
        return_codegen = codegen_for_annotation(mir.return_annotation_ir, runtime_package, context=type_codegen_context)
    param_str, call_args_str, unwrap_code = format_parameters_for_emit(
        mir.parameters,
        current_target_module,
        runtime_package,
        unwrap_indent="    ",
        type_codegen_context=type_codegen_context,
        annotation_local_names=annotation_local_names,
    )
    call_expr_prefix = _method_impl_call_expr(
        method_type,
        impl_dotted,
        method_name,
        call_args_str,
    )
    dummy_param_str = param_str
    is_async_gen = mir.is_async_gen
    is_async = mir.is_async
    sync_return_str, async_return_str = _format_return_annotation(
        return_codegen,
        current_target_module,
        annotation_local_names=annotation_local_names,
    )
    uses_with_aio_wrapper = is_async
    method_with_aio_type_expr: str | None = None
    if uses_with_aio_wrapper:
        method_with_aio_type_expr, _needs_type_checking_stub = _method_with_aio_type_data(
            owner, mir, runtime_package, type_codegen_context=type_codegen_context
        )

    aio_body = None
    sync_method_body = ""

    if method_type == MethodBindingKind.INSTANCE:
        if not is_async:
            sync_call_expr = call_expr_prefix
            sync_method_body = _build_call_with_wrap(
                sync_call_expr,
                return_codegen,
                current_target_module,
                indent="    ",
                is_async=False,
                method_type=method_type,
                method_owner_impl_ref=owner.impl_ref,
            )
            if unwrap_code:
                sync_method_body = unwrap_code + "\n" + sync_method_body
            aio_body = None
        elif is_async_gen:
            gen_call = call_expr_prefix
            impl_to_wrapper_expr_raw = return_codegen.impl_to_wrapper_expr(current_target_module, "gen")
            impl_to_wrapper_expr = impl_to_wrapper_expr_raw.replace("self.", "wrapper_instance.")
            aio_body = (f"    {unwrap_code}\n" if unwrap_code else "") + (
                f"    gen = {gen_call}\n"
                f"    _wrapped = {impl_to_wrapper_expr}\n"
                f"    _sent = None\n"
                f"    try:\n"
                f"        while True:\n"
                f"            try:\n"
                f"                _item = await _wrapped.asend(_sent)\n"
                f"                _sent = yield _item\n"
                f"            except StopAsyncIteration:\n"
                f"                break\n"
                f"    finally:\n"
                f"        await _wrapped.aclose()"
            )
            sync_impl_to_wrapper_expr_raw = return_codegen.impl_to_wrapper_expr(
                current_target_module, "gen", is_async=False
            )
            sync_impl_to_wrapper_expr = sync_impl_to_wrapper_expr_raw
            if unwrap_code:
                sync_method_body = f"{unwrap_code}\n    gen = {gen_call}\n    yield from {sync_impl_to_wrapper_expr}"
            else:
                sync_method_body = f"    gen = {gen_call}\n    yield from {sync_impl_to_wrapper_expr}"
        else:
            aio_body = _build_call_with_wrap(
                call_expr_prefix,
                return_codegen,
                current_target_module,
                indent="    ",
                is_async=True,
                method_type=method_type,
                method_owner_impl_ref=owner.impl_ref,
            )
            if unwrap_code:
                aio_body = unwrap_code + "\n" + aio_body

            sync_method_body = _build_call_with_wrap(
                call_expr_prefix,
                return_codegen,
                current_target_module,
                indent="    ",
                is_async=False,
                method_type=method_type,
                method_owner_impl_ref=owner.impl_ref,
            )
            if unwrap_code:
                sync_method_body = unwrap_code + "\n" + sync_method_body
    elif method_type == MethodBindingKind.CLASSMETHOD:
        if not is_async:
            sync_call_expr = call_expr_prefix
            sync_method_body = _build_call_with_wrap(
                sync_call_expr,
                return_codegen,
                current_target_module,
                indent="    ",
                is_async=False,
                method_type=method_type,
                method_owner_impl_ref=owner.impl_ref,
            )
            if unwrap_code:
                sync_method_body = unwrap_code + "\n" + sync_method_body
            aio_body = None
        elif is_async_gen:
            gen_call = call_expr_prefix
            impl_to_wrapper_expr_raw = return_codegen.impl_to_wrapper_expr(current_target_module, "gen")
            impl_to_wrapper_expr = impl_to_wrapper_expr_raw.replace("self.", "wrapper_class.")
            unwrap_lines = f"{unwrap_code}\n    " if unwrap_code else ""
            aio_body = (
                f"    {unwrap_lines}gen = {gen_call}\n"
                f"    _wrapped = {impl_to_wrapper_expr}\n"
                f"    _sent = None\n"
                f"    try:\n"
                f"        while True:\n"
                f"            try:\n"
                f"                _item = await _wrapped.asend(_sent)\n"
                f"                _sent = yield _item\n"
                f"            except StopAsyncIteration:\n"
                f"                break\n"
                f"    finally:\n"
                f"        await _wrapped.aclose()"
            )
            sync_impl_to_wrapper_expr_raw = return_codegen.impl_to_wrapper_expr(
                current_target_module, "gen", is_async=False
            )
            sync_impl_to_wrapper_expr = sync_impl_to_wrapper_expr_raw
            if unwrap_code:
                sync_method_body = f"{unwrap_code}\n    gen = {gen_call}\n    yield from {sync_impl_to_wrapper_expr}"
            else:
                sync_method_body = f"    gen = {gen_call}\n    yield from {sync_impl_to_wrapper_expr}"
        else:
            aio_body = _build_call_with_wrap(
                call_expr_prefix,
                return_codegen,
                current_target_module,
                indent="    ",
                is_async=True,
                method_type=method_type,
                method_owner_impl_ref=owner.impl_ref,
            )
            if unwrap_code:
                aio_body = unwrap_code + "\n" + aio_body

            sync_method_body = _build_call_with_wrap(
                call_expr_prefix,
                return_codegen,
                current_target_module,
                indent="    ",
                is_async=False,
                method_type=method_type,
                method_owner_impl_ref=owner.impl_ref,
            )
            if unwrap_code:
                sync_method_body = unwrap_code + "\n" + sync_method_body
    elif method_type == MethodBindingKind.STATICMETHOD:
        if not is_async:
            sync_call_expr = call_expr_prefix
            sync_method_body = _build_call_with_wrap(
                sync_call_expr,
                return_codegen,
                current_target_module,
                indent="    ",
                is_async=False,
                method_type=method_type,
                method_owner_impl_ref=owner.impl_ref,
            )
            if unwrap_code:
                sync_method_body = unwrap_code + "\n" + sync_method_body
            aio_body = None
        elif is_async_gen:
            gen_call = call_expr_prefix
            impl_to_wrapper_expr_raw = return_codegen.impl_to_wrapper_expr(current_target_module, "gen")
            if "self." in impl_to_wrapper_expr_raw:
                impl_to_wrapper_expr = impl_to_wrapper_expr_raw.replace("self.", f"{short_name}()._").replace("_(", "(")
            else:
                impl_to_wrapper_expr = impl_to_wrapper_expr_raw
            unwrap_lines = f"{unwrap_code}\n    " if unwrap_code else ""
            aio_body = (
                f"    {unwrap_lines}gen = {gen_call}\n"
                f"    _wrapped = {impl_to_wrapper_expr}\n"
                f"    _sent = None\n"
                f"    try:\n"
                f"        while True:\n"
                f"            try:\n"
                f"                _item = await _wrapped.asend(_sent)\n"
                f"                _sent = yield _item\n"
                f"            except StopAsyncIteration:\n"
                f"                break\n"
                f"    finally:\n"
                f"        await _wrapped.aclose()"
            )
            sync_impl_to_wrapper_expr_raw = return_codegen.impl_to_wrapper_expr(
                current_target_module, "gen", is_async=False
            )
            if "self." in sync_impl_to_wrapper_expr_raw:
                sync_impl_to_wrapper_expr = sync_impl_to_wrapper_expr_raw.replace("self.", f"{short_name}()._").replace(
                    "_(", "("
                )
            else:
                sync_impl_to_wrapper_expr = sync_impl_to_wrapper_expr_raw
            if unwrap_code:
                sync_method_body = f"{unwrap_code}\n    gen = {gen_call}\n    yield from {sync_impl_to_wrapper_expr}"
            else:
                sync_method_body = f"    gen = {gen_call}\n    yield from {sync_impl_to_wrapper_expr}"
        else:
            aio_body = _build_call_with_wrap(
                call_expr_prefix,
                return_codegen,
                current_target_module,
                indent="    ",
                is_async=True,
                method_type=method_type,
                method_owner_impl_ref=owner.impl_ref,
            )
            if unwrap_code:
                aio_body = unwrap_code + "\n" + aio_body

            sync_method_body = _build_call_with_wrap(
                call_expr_prefix,
                return_codegen,
                current_target_module,
                indent="    ",
                is_async=False,
                method_type=method_type,
                method_owner_impl_ref=owner.impl_ref,
            )
            if unwrap_code:
                sync_method_body = unwrap_code + "\n" + sync_method_body

    aio_method_name = f"__{method_name}_aio"

    if uses_with_aio_wrapper:
        wrapper_functions_code = ""
    elif method_type == MethodBindingKind.INSTANCE:
        if aio_body is not None:
            aio_body_with_self = aio_body.replace("wrapper_instance", "self")
            aio_body_lines = aio_body_with_self.split("\n")
            aio_body_indented = "\n".join(
                (
                    "        " + line[4:]
                    if line.strip() and len(line) > 4 and line.startswith("    ")
                    else "        " + line.lstrip()
                    if line.strip()
                    else ""
                )
                for line in aio_body_lines
            )
            aio_body_indented = _prepend_docstring(aio_body_indented, mir.docstring, indent="        ")
            aio_wrapper_method = (
                f"    async def {aio_method_name}(self, {param_str}){async_return_str}:\n{aio_body_indented}"
            )
            wrapper_functions_code = aio_wrapper_method
        else:
            wrapper_functions_code = ""
            aio_body = None
    elif method_type == MethodBindingKind.CLASSMETHOD:
        if aio_body is not None:
            aio_body_with_cls = aio_body.replace("wrapper_class", "cls")
            aio_body_lines = aio_body_with_cls.split("\n")
            aio_body_indented = "\n".join(
                (
                    "        " + line[4:]
                    if line.strip() and len(line) > 4 and line.startswith("    ")
                    else "        " + line.lstrip()
                    if line.strip()
                    else ""
                )
                for line in aio_body_lines
            )
            aio_body_indented = _prepend_docstring(aio_body_indented, mir.docstring, indent="        ")
            aio_wrapper_method = (
                f"    @classmethod\n"
                f"    async def {aio_method_name}(cls, {param_str}){async_return_str}:\n"
                f"{aio_body_indented}"
            )
            wrapper_functions_code = aio_wrapper_method
        else:
            wrapper_functions_code = ""
            aio_body = None
    elif method_type == MethodBindingKind.STATICMETHOD:
        if aio_body is not None:
            aio_body_lines = aio_body.split("\n")
            aio_body_indented = "\n".join(
                (
                    "        " + line[4:]
                    if line.strip() and len(line) > 4 and line.startswith("    ")
                    else "        " + line.lstrip()
                    if line.strip()
                    else ""
                )
                for line in aio_body_lines
            )
            aio_body_indented = _prepend_docstring(aio_body_indented, mir.docstring, indent="        ")
            aio_wrapper_method = (
                f"    @staticmethod\n"
                f"    async def {aio_method_name}({param_str}){async_return_str}:\n"
                f"{aio_body_indented}"
            )
            wrapper_functions_code = aio_wrapper_method
        else:
            wrapper_functions_code = ""
            aio_body = None
    else:
        wrapper_functions_code = ""

    if uses_with_aio_wrapper:
        assert method_with_aio_type_expr is not None
        if method_type == MethodBindingKind.CLASSMETHOD:
            decorator_line = f"@classmethod_with_aio({method_with_aio_type_expr})\n    @classmethod"
        elif method_type == MethodBindingKind.STATICMETHOD:
            decorator_line = f"@staticmethod_with_aio({method_with_aio_type_expr})\n    @staticmethod"
        else:
            decorator_line = f"@method_with_aio({method_with_aio_type_expr})"
        if method_type == MethodBindingKind.INSTANCE:
            method_body_lines = sync_method_body.replace("wrapper_instance", "self").split("\n")
            method_body = "\n".join(line.lstrip() if line.strip() else "" for line in method_body_lines).strip()
        elif method_type == MethodBindingKind.CLASSMETHOD:
            method_body_lines = sync_method_body.replace("wrapper_class", "cls").split("\n")
            method_body = "\n".join(line.lstrip() if line.strip() else "" for line in method_body_lines).strip()
        else:
            method_body_lines = sync_method_body.split("\n")
            method_body = "\n".join(line.lstrip() if line.strip() else "" for line in method_body_lines).strip()
    else:
        if method_type == MethodBindingKind.INSTANCE:
            decorator_line = ""
            method_body_lines = sync_method_body.replace("wrapper_instance", "self").split("\n")
            method_body = "\n".join(line.lstrip() if line.strip() else "" for line in method_body_lines).strip()
        elif method_type == MethodBindingKind.CLASSMETHOD:
            decorator_line = "@classmethod"
            method_body_lines = sync_method_body.replace("wrapper_class", "cls").split("\n")
            method_body = "\n".join(line.lstrip() if line.strip() else "" for line in method_body_lines).strip()
        else:
            decorator_line = "@staticmethod"
            method_body_lines = sync_method_body.split("\n")
            method_body = "\n".join(line.lstrip() if line.strip() else "" for line in method_body_lines).strip()

    if method_type in (MethodBindingKind.CLASSMETHOD, MethodBindingKind.STATICMETHOD):
        if method_type == MethodBindingKind.CLASSMETHOD:
            if param_str:
                plain_param_str = f"cls, {param_str}"
            else:
                plain_param_str = "cls"
            def_line = f"    def {method_name}({plain_param_str}){sync_return_str}:"
        else:
            def_line = f"    def {method_name}({dummy_param_str}){sync_return_str}:"
    else:
        if param_str:
            instance_param_str = f"self, {param_str}"
        else:
            instance_param_str = "self"
        def_line = f"    def {method_name}({instance_param_str}){sync_return_str}:"
    method_body = _prepend_docstring(method_body, mir.docstring, indent="")
    indented_method_body = _indent_block(method_body, "        ")

    if decorator_line:
        runtime_method_code = f"    {decorator_line}\n{def_line}\n{indented_method_body}"
        sync_method_code = runtime_method_code
    else:
        sync_method_code = f"{def_line}\n{indented_method_body}"

    overload_code = ""
    if mir.overloads and not uses_with_aio_wrapper:
        overload_code = _emit_method_overloads(
            mir.overloads,
            method_name,
            method_type,
            current_target_module,
            runtime_package,
            type_codegen_context=type_codegen_context,
            is_async=False,
            annotation_local_names=annotation_local_names,
        )
    if overload_code:
        sync_method_code = f"{overload_code}\n{sync_method_code}"

    return wrapper_functions_code, sync_method_code


def emit_wrapped_class(
    ir: WrappedClassIR,
    target_module: str,
    *,
    runtime_package: str = "synchronicity2",
    type_codegen_context: TypeCodegenContext | None = None,
) -> str:
    """Emit wrapper class source from :class:`WrappedClassIR` (no live implementation objects)."""
    wshort = ir.wrapper_ref.qualname
    impl_dot = _impl_type_dotted(ir.impl_ref)
    proxy_type_comment = f"# Proxy type for the underlying implementation type {impl_dot}."
    owner = method_emit_owner(ir, target_module)
    annotation_local_names = _class_annotation_shadow_names(ir)
    (
        init_mir,
        descriptor_mirs,
        dynamic_attr_mirs,
        iterator_mirs,
        context_manager_mirs,
        normal_methods,
    ) = _partition_class_methods(ir.methods)

    # Build return code generators once so helpers and method bodies use the same instances.
    method_codegens: dict[str, TypeCodegen] = {}
    all_helpers_dict: dict[str, str] = {}
    for mir in ir.methods:
        rt = codegen_for_annotation(mir.return_annotation_ir, runtime_package, context=type_codegen_context)
        method_codegens[mir.method_name] = rt
        all_helpers_dict.update(rt.helper_definitions(target_module, indent="    "))

    helpers_code = "\n".join(all_helpers_dict.values()) if all_helpers_dict else ""
    helpers_section = f"\n{helpers_code}\n" if helpers_code else ""

    method_with_aio_types: list[str] = []
    for mir in normal_methods:
        if not mir.is_async:
            continue
        method_with_aio_types.append(
            emit_method_with_aio_protocol(
                owner,
                mir,
                runtime_package,
                type_codegen_context=type_codegen_context,
            )
        )

    method_definitions_with_async: list[str] = []
    for mir in normal_methods:
        wrapper_functions_code, sync_method_code = emit_wrapped_method_pair(
            owner,
            mir,
            runtime_package=runtime_package,
            type_codegen_context=type_codegen_context,
            return_codegen=method_codegens[mir.method_name],
            annotation_local_names=annotation_local_names,
        )
        if wrapper_functions_code:
            method_definitions_with_async.append(f"{wrapper_functions_code}\n\n{sync_method_code}")
        else:
            method_definitions_with_async.append(sync_method_code)

    property_definitions = []
    for attr_name, annotation_ir in ir.attributes:
        attr_type_str = ""
        if annotation_ir is not None:
            attr_codegen = codegen_for_annotation(annotation_ir, runtime_package, context=type_codegen_context)
            attr_type_str = attr_codegen.annotation_type(target_module)
            attr_type_str = quote_annotation_for_local_names(attr_type_str, annotation_local_names)
        if attr_type_str:
            property_code = f"""    # Generated properties
    @property
    def {attr_name}(self) -> {attr_type_str}:
        return self._impl_instance.{attr_name}

    @{attr_name}.setter
    def {attr_name}(self, value: {attr_type_str}):
        self._impl_instance.{attr_name} = value"""
        else:
            property_code = f"""    @property
    def {attr_name}(self):
        return self._impl_instance.{attr_name}

    @{attr_name}.setter
    def {attr_name}(self, value):
        self._impl_instance.{attr_name} = value"""
        property_definitions.append(property_code)

    # @property delegation (from @property on impl class)
    for prop_ir in ir.properties:
        prop_codegen = None
        prop_type_str = ""
        if prop_ir.return_annotation_ir is not None:
            prop_codegen = codegen_for_annotation(
                prop_ir.return_annotation_ir, runtime_package, context=type_codegen_context
            )
            prop_type_str = prop_codegen.annotation_type(target_module)
            prop_type_str = quote_annotation_for_local_names(prop_type_str, annotation_local_names)
        # Getter: wrap impl value → wrapper value if needed
        getter_value_expr = f"self._impl_instance.{prop_ir.name}"
        if prop_codegen is not None and prop_codegen.requires_boundary_translation():
            getter_return_expr = prop_codegen.impl_to_wrapper_expr(target_module, "_impl_val", is_async=False)
            getter_body = f"_impl_val = {getter_value_expr}\n        return {getter_return_expr}"
        else:
            getter_body = f"return {getter_value_expr}"
        if prop_type_str:
            getter_code = f"""    @property
    def {prop_ir.name}(self) -> {prop_type_str}:
        {getter_body}"""
        else:
            getter_code = f"""    @property
    def {prop_ir.name}(self):
        {getter_body}"""
        if prop_ir.has_setter:
            setter_codegen = None
            setter_type_str = ""
            if prop_ir.setter_annotation_ir is not None:
                setter_codegen = codegen_for_annotation(
                    prop_ir.setter_annotation_ir, runtime_package, context=type_codegen_context
                )
                setter_type_str = setter_codegen.annotation_type(target_module)
                setter_type_str = quote_annotation_for_local_names(setter_type_str, annotation_local_names)
            # Setter: unwrap wrapper value → impl value if needed
            if setter_codegen is not None and setter_codegen.requires_boundary_translation():
                setter_assign_expr = setter_codegen.wrapper_to_impl_expr("value")
                setter_body = f"self._impl_instance.{prop_ir.name} = {setter_assign_expr}"
            else:
                setter_body = f"self._impl_instance.{prop_ir.name} = value"
            if setter_type_str:
                setter_code = f"""

    @{prop_ir.name}.setter
    def {prop_ir.name}(self, value: {setter_type_str}):
        {setter_body}"""
            else:
                setter_code = f"""

    @{prop_ir.name}.setter
    def {prop_ir.name}(self, value):
        {setter_body}"""
            property_definitions.append(getter_code + setter_code)
        else:
            property_definitions.append(getter_code)

    class_property_definitions = []
    for class_prop_ir in ir.class_properties:
        prop_codegen = None
        prop_type_str = ""
        if class_prop_ir.return_annotation_ir is not None:
            prop_codegen = codegen_for_annotation(
                class_prop_ir.return_annotation_ir,
                runtime_package,
                context=type_codegen_context,
            )
            prop_type_str = prop_codegen.annotation_type(target_module)
            prop_type_str = quote_annotation_for_local_names(prop_type_str, annotation_local_names)

        getter_value_expr = f"{impl_dot}.{class_prop_ir.name}"
        if prop_codegen is not None and prop_codegen.requires_boundary_translation():
            getter_return_expr = prop_codegen.impl_to_wrapper_expr(target_module, "_impl_val", is_async=False)
            getter_body = f"_impl_val = {getter_value_expr}\n        return {getter_return_expr}"
        else:
            getter_body = f"return {getter_value_expr}"

        if prop_type_str:
            getter_code = f"""    @classproperty
    def {class_prop_ir.name}(cls) -> {prop_type_str}:
        {getter_body}"""
        else:
            getter_code = f"""    @classproperty
    def {class_prop_ir.name}(cls):
        {getter_body}"""
        class_property_definitions.append(getter_code)

    descriptor_methods_section = ""
    if descriptor_mirs:
        descriptor_blocks: list[str] = []
        for mir in descriptor_mirs:
            if mir.method_name != "__get__":
                continue
            method_return_codegen = method_codegens[mir.method_name]
            method_sync_return_str, _method_async_return_str = _format_return_annotation(
                method_return_codegen,
                target_module,
                annotation_local_names=annotation_local_names,
            )
            method_param_str, method_call_args_str, method_unwrap_code = format_parameters_for_emit(
                mir.parameters,
                target_module,
                runtime_package,
                unwrap_indent="        ",
                type_codegen_context=type_codegen_context,
                annotation_local_names=annotation_local_names,
            )
            method_params = f"self, {method_param_str}" if method_param_str else "self"
            method_impl_to_wrapper_expr = method_return_codegen.impl_to_wrapper_expr(target_module, "result", False)
            method_body = f"""        result = {impl_dot}.__get__(self._impl_instance, {method_call_args_str})
        return {method_impl_to_wrapper_expr}"""
            if method_unwrap_code:
                method_body = f"{method_unwrap_code}\n{method_body}"
            descriptor_blocks.append(
                f"""    def __get__({method_params}){method_sync_return_str}:
{method_body}"""
            )
        descriptor_methods_section = "\n\n".join(descriptor_blocks)

    dynamic_attr_methods_section = ""
    if dynamic_attr_mirs:
        dynamic_attr_blocks: list[str] = []
        for mir in dynamic_attr_mirs:
            if mir.method_name != "__getattr__":
                continue
            method_return_codegen = method_codegens[mir.method_name]
            method_sync_return_str, _method_async_return_str = _format_return_annotation(
                method_return_codegen,
                target_module,
                annotation_local_names=annotation_local_names,
            )
            method_param_str, method_call_args_str, method_unwrap_code = format_parameters_for_emit(
                mir.parameters,
                target_module,
                runtime_package,
                unwrap_indent="        ",
                type_codegen_context=type_codegen_context,
                annotation_local_names=annotation_local_names,
            )
            method_params = f"self, {method_param_str}" if method_param_str else "self"
            method_impl_to_wrapper_expr = method_return_codegen.impl_to_wrapper_expr(target_module, "result", False)
            method_body = f"""        result = {impl_dot}.__getattr__(self._impl_instance, {method_call_args_str})
        return {method_impl_to_wrapper_expr}"""
            if method_unwrap_code:
                method_body = f"{method_unwrap_code}\n{method_body}"
            method_body = _prepend_docstring(method_body, mir.docstring, indent="        ")
            dynamic_attr_blocks.append(
                f"""    def __getattr__({method_params}){method_sync_return_str}:
{method_body}"""
            )
        dynamic_attr_methods_section = "\n\n".join(dynamic_attr_blocks)

    iterator_methods_section = ""
    if iterator_mirs:
        iterator_blocks: list[str] = []
        for mir in iterator_mirs:
            dunder_pair = _async_iterator_dunder_pair(mir.method_name)
            if dunder_pair is None:
                continue
            sync_method_name, async_method_name, use_async_def, stop_iteration_bridge = dunder_pair
            method_return_codegen = method_codegens[mir.method_name]
            method_sync_return_str, method_async_return_str = _format_return_annotation(
                method_return_codegen,
                target_module,
                annotation_local_names=annotation_local_names,
            )
            method_call_expr = f"{impl_dot}.{mir.method_name}(self._impl_instance)"
            sync_indent = "            " if stop_iteration_bridge else "        "
            sync_body = _build_call_with_wrap(
                method_call_expr,
                method_return_codegen,
                target_module,
                indent=sync_indent,
                is_async=False,
                method_type=MethodBindingKind.INSTANCE,
                method_owner_impl_ref=ir.impl_ref,
            )
            sync_body = _prepend_docstring(sync_body, mir.docstring, indent=sync_indent)
            if stop_iteration_bridge:
                sync_method = f"""    def {sync_method_name}(self){method_sync_return_str}:
        try:
{sync_body}
        except StopAsyncIteration:
            raise StopIteration()"""
            else:
                sync_method = f"""    def {sync_method_name}(self){method_sync_return_str}:
{sync_body}"""
            iterator_blocks.append(sync_method)

            async_body = _build_call_with_wrap(
                method_call_expr,
                method_return_codegen,
                target_module,
                indent="        ",
                is_async=True,
                method_type=MethodBindingKind.INSTANCE,
                method_owner_impl_ref=ir.impl_ref,
            )
            async_body = _prepend_docstring(async_body, mir.docstring, indent="        ")
            async_def_keyword = "async def" if use_async_def else "def"
            async_method = f"""    {async_def_keyword} {async_method_name}(self){method_async_return_str}:
{async_body}"""
            iterator_blocks.append(async_method)
        iterator_methods_section = "\n\n".join(iterator_blocks)

    context_manager_methods_section = ""
    if context_manager_mirs:
        cm_blocks: list[str] = []
        for mir in context_manager_mirs:
            context_manager_dunder_pair = _async_context_manager_dunder_pair(mir.method_name)
            if context_manager_dunder_pair is None:
                continue
            sync_method_name, async_method_name = context_manager_dunder_pair
            method_return_codegen = method_codegens[mir.method_name]
            method_sync_return_str, method_async_return_str = _format_return_annotation(
                method_return_codegen,
                target_module,
                annotation_local_names=annotation_local_names,
            )
            # Build parameter strings for __aexit__ (or empty for __aenter__)
            cm_param_str, cm_call_args_str, cm_unwrap_code = format_parameters_for_emit(
                mir.parameters,
                target_module,
                runtime_package,
                unwrap_indent="        ",
                type_codegen_context=type_codegen_context,
                annotation_local_names=annotation_local_names,
            )
            sync_param_str = f"self, {cm_param_str}" if cm_param_str else "self"

            if cm_call_args_str:
                method_call_expr = f"{impl_dot}.{mir.method_name}(self._impl_instance, {cm_call_args_str})"
            else:
                method_call_expr = f"{impl_dot}.{mir.method_name}(self._impl_instance)"

            # Sync version
            sync_body = _build_call_with_wrap(
                method_call_expr,
                method_return_codegen,
                target_module,
                indent="        ",
                is_async=False,
                method_type=MethodBindingKind.INSTANCE,
                method_owner_impl_ref=ir.impl_ref,
            )
            sync_method_body = sync_body
            if cm_unwrap_code:
                sync_method_body = f"{cm_unwrap_code}\n{sync_body}"
            sync_method_body = _prepend_docstring(sync_method_body, mir.docstring, indent="        ")
            sync_method = f"""    def {sync_method_name}({sync_param_str}){method_sync_return_str}:
{sync_method_body}"""
            cm_blocks.append(sync_method)

            # Async version
            async_body = _build_call_with_wrap(
                method_call_expr,
                method_return_codegen,
                target_module,
                indent="        ",
                is_async=True,
                method_type=MethodBindingKind.INSTANCE,
                method_owner_impl_ref=ir.impl_ref,
            )
            async_method_body = async_body
            if cm_unwrap_code:
                async_method_body = f"{cm_unwrap_code}\n{async_body}"
            async_method_body = _prepend_docstring(async_method_body, mir.docstring, indent="        ")
            async_method = f"""    async def {async_method_name}({sync_param_str}){method_async_return_str}:
{async_method_body}"""
            cm_blocks.append(async_method)
        context_manager_methods_section = "\n\n".join(cm_blocks)

    wrapped_base_strings = [_wrapper_class_reference(wrapper, target_module) for _impl_ref, wrapper in ir.wrapped_bases]
    from_impl_method = f"""    @classmethod
    def _from_impl(cls, impl_instance: typing.Any) -> "{wshort}":
        \"\"\"Create wrapper from implementation instance, preserving identity via cache.\"\"\"
        return _wrapped_from_impl(cls, impl_instance, cls._instance_cache, _synchronizer)"""

    all_bases: list[str] = []
    if wrapped_base_strings:
        all_bases.extend(wrapped_base_strings)
    if ir.generic_type_parameters:
        all_bases.append(f"typing.Generic[{', '.join(ir.generic_type_parameters)}]")

    if all_bases:
        bases_str = ", ".join(all_bases)
        class_declaration = f"""class {wshort}({bases_str}):"""
    else:
        class_declaration = f"""class {wshort}:"""

    class_attrs = (
        f"""    \"\"\"Wrapper class for {impl_dot} """
        f"""with sync/async method support\"\"\"

    _instance_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()"""
    )

    init_sig, init_call, init_unwrap = format_parameters_for_emit(
        init_mir.parameters if init_mir else (),
        target_module,
        runtime_package,
        unwrap_indent="        ",
        type_codegen_context=type_codegen_context,
        annotation_local_names=annotation_local_names,
    )
    init_params = f"self, {init_sig}" if init_sig else "self"
    init_method = f"""    def __init__({init_params}):
{init_unwrap}
        self._impl_instance = {impl_dot}({init_call})
        type(self)._instance_cache[id(self._impl_instance)] = self"""

    properties_section = "\n\n".join(property_definitions) if property_definitions else ""
    class_properties_section = "\n\n".join(class_property_definitions) if class_property_definitions else ""
    methods_section = "\n\n".join(method_definitions_with_async) if method_definitions_with_async else ""
    manual_attributes_section = "\n".join(
        f"    {manual_attr.name} = "
        + (
            f"{impl_dot}.{manual_attr.name}"
            if manual_attr.access_kind == ManualClassAttributeAccessKind.ATTRIBUTE
            else f"{impl_dot}.__dict__[{manual_attr.name!r}]"
        )
        for manual_attr in ir.manual_attributes
    )

    sections = [init_method]
    if from_impl_method:
        sections.append(from_impl_method)
    if properties_section:
        sections.append(properties_section)
    if class_properties_section:
        sections.append(class_properties_section)
    if descriptor_methods_section:
        sections.append(descriptor_methods_section)
    if dynamic_attr_methods_section:
        sections.append(dynamic_attr_methods_section)
    if iterator_methods_section:
        sections.append(iterator_methods_section)
    if context_manager_methods_section:
        sections.append(context_manager_methods_section)
    if methods_section:
        sections.append(methods_section)
    if manual_attributes_section:
        sections.append(manual_attributes_section)

    sections_combined = "\n\n".join(sections)

    wrapper_class_code = f"""{proxy_type_comment}
{class_declaration}
{class_attrs}{helpers_section}

{sections_combined}"""

    if method_with_aio_types:
        return "\n\n".join([*method_with_aio_types, wrapper_class_code])
    return wrapper_class_code


class SyncAsyncModuleEmitter:
    """Default emitter: blocking wrappers + hidden ``__*_aio`` async implementations."""

    def __init__(self, runtime_package: str = "synchronicity2"):
        self.runtime_package = runtime_package

    def emit_module(
        self,
        ir: ModuleIR,
    ) -> str:
        runtime_package = self.runtime_package
        default_import_modules = _default_import_modules_for_module(ir)
        annotation_import_modules = set(module_import_modules(ir))
        module_imports = sorted(
            (set(ir.impl_modules) | default_import_modules | annotation_import_modules) - {ir.target_module}
        )
        imports = "\n".join(f"import {mod}" for mod in module_imports)
        cross_module_import_strs = [
            f"import {m}" for m in sorted(set(ir.cross_module_imports.keys()) - set(module_imports))
        ]
        if ir.has_wrapped_classes:
            cross_module_import_strs.append("import weakref")
        additional_imports = "\n".join(cross_module_import_strs)
        if additional_imports:
            additional_imports += "\n"

        header = f"""{_GENERATED_MODULE_BANNER}
import typing

{imports}

{additional_imports}
{_runtime_import_header(runtime_package)}_synchronizer = {ir.synchronizer_module}.synchronizer

"""

        compiled_code = [header]

        if ir.typevar_specs:
            for definition in type_parameter_definition_lines(ir.typevar_specs):
                compiled_code.append(definition)
            compiled_code.append("")

        module_type_codegen_context = _type_codegen_context_for_module(ir.typevar_specs)
        for i, cw in enumerate(ir.wrapped_classes):
            code = emit_wrapped_class(
                cw, ir.target_module, runtime_package=runtime_package, type_codegen_context=module_type_codegen_context
            )
            if i > 0:
                compiled_code.append("")
            compiled_code.append(code)

        if ir.wrapped_classes:
            compiled_code.append("")
            compiled_code.extend(_wrapper_registration_lines(ir.wrapped_classes))
            compiled_code.append("")

        for func_ir in ir.wrapped_functions:
            code = emit_wrapped_function(
                func_ir,
                ir.target_module,
                runtime_package=runtime_package,
                type_codegen_context=module_type_codegen_context,
            )
            compiled_code.append(code)
            compiled_code.append("")

        for reexport_ir in ir.manual_reexports:
            compiled_code.append(emit_manual_reexport(reexport_ir))
            compiled_code.append("")

        return "\n".join(compiled_code)
