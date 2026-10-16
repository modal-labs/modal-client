# Copyright Modal Labs 2026
"""Generate callable signatures and call-boundary expressions from IR."""

from __future__ import annotations

import inspect
import re
import typing

from ..ir.declarations import MethodBindingKind, ParameterIR
from ..ir.references import ObjectReferenceIR
from .type_codegen import (
    CallableTypeCodegen,
    TypeCodegenContext,
    WrappedClassTypeCodegen,
    codegen_for_annotation,
)

_ANNOTATION_IDENTIFIER_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\b")


def quote_annotation_for_local_names(annotation: str, local_names: typing.AbstractSet[str] | None) -> str:
    """Quote an annotation when local scope names would shadow identifiers inside it."""

    if not annotation or not local_names or annotation[0] in {"'", '"'}:
        return annotation
    if not any(match.group(1) in local_names for match in _ANNOTATION_IDENTIFIER_RE.finditer(annotation)):
        return annotation
    escaped = annotation.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _parameter_annotation_str(codegen, current_target_module: str) -> str:
    if isinstance(codegen, CallableTypeCodegen):
        return codegen.implementation_annotation(current_target_module)
    return codegen.annotation_type(current_target_module)


def format_parameters_for_emit(
    parameters: tuple[ParameterIR, ...],
    current_target_module: str,
    runtime_package: str = "synchronicity2",
    unwrap_indent: str = "    ",
    *,
    type_codegen_context: TypeCodegenContext | None = None,
    annotation_local_names: typing.AbstractSet[str] | None = None,
) -> tuple[str, str, str]:
    """Build ``param_str``, ``call_args_str``, and unwrap lines from :class:`ParameterIR` (emitter-side)."""
    params: list[str] = []
    call_args: list[str] = []
    unwrap_stmts: list[str] = []

    last_positional_only_index = -1
    positional_only_marker_added = False
    keyword_only_marker_added = False

    def _wrapper_to_impl_expr(codegen, source_name: str) -> str:
        return codegen.wrapper_to_impl_expr(source_name, current_target_module)

    def _vararg_wrapper_to_impl_expr(codegen, source_name: str) -> str:
        item_unwrap = _wrapper_to_impl_expr(codegen, "_item")
        return f"tuple({item_unwrap} for _item in {source_name})"

    def _varkw_wrapper_to_impl_expr(codegen, source_name: str) -> str:
        value_unwrap = _wrapper_to_impl_expr(codegen, "_value")
        return f"{{_key: {value_unwrap} for _key, _value in {source_name}.items()}}"

    for param_ir in parameters:
        name = param_ir.name
        kind = param_ir.kind
        codegen = (
            codegen_for_annotation(param_ir.annotation_ir, runtime_package, context=type_codegen_context)
            if param_ir.annotation_ir is not None
            else None
        )

        if kind == inspect.Parameter.POSITIONAL_ONLY:
            last_positional_only_index = len(params)

        if kind == inspect.Parameter.VAR_POSITIONAL:
            if param_ir.annotation_ir is not None:
                assert codegen is not None
                ann = _parameter_annotation_str(codegen, current_target_module)
                ann = quote_annotation_for_local_names(ann, annotation_local_names)
                params.append(f"*{name}: {ann}")
                if codegen.requires_boundary_translation():
                    wrapper_to_impl_expr = _vararg_wrapper_to_impl_expr(codegen, name)
                    unwrap_stmts.append(f"{unwrap_indent}{name}_impl = {wrapper_to_impl_expr}")
                    call_args.append(f"*{name}_impl")
                else:
                    call_args.append(f"*{name}")
            else:
                params.append(f"*{name}")
                call_args.append(f"*{name}")

        elif kind == inspect.Parameter.VAR_KEYWORD:
            if param_ir.annotation_ir is not None:
                assert codegen is not None
                ann = _parameter_annotation_str(codegen, current_target_module)
                ann = quote_annotation_for_local_names(ann, annotation_local_names)
                params.append(f"**{name}: {ann}")
                if codegen.requires_boundary_translation():
                    wrapper_to_impl_expr = _varkw_wrapper_to_impl_expr(codegen, name)
                    unwrap_stmts.append(f"{unwrap_indent}{name}_impl = {wrapper_to_impl_expr}")
                    call_args.append(f"**{name}_impl")
                else:
                    call_args.append(f"**{name}")
            else:
                params.append(f"**{name}")
                call_args.append(f"**{name}")

        elif kind == inspect.Parameter.KEYWORD_ONLY:
            if not keyword_only_marker_added and not any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters
            ):
                params.append("*")
                keyword_only_marker_added = True
            if param_ir.annotation_ir is not None:
                assert codegen is not None
                ann = _parameter_annotation_str(codegen, current_target_module)
                ann = quote_annotation_for_local_names(ann, annotation_local_names)
                param_str = f"{name}: {ann}"

                if codegen.requires_boundary_translation():
                    wrapper_to_impl_expr = _wrapper_to_impl_expr(codegen, name)
                    unwrap_stmts.append(f"{unwrap_indent}{name}_impl = {wrapper_to_impl_expr}")
                    call_args.append(f"{name}={name}_impl")
                else:
                    call_args.append(f"{name}={name}")
            else:
                param_str = name
                call_args.append(f"{name}={name}")

            if param_ir.default_expr is not None:
                param_str += f" = {param_ir.default_expr}"

            params.append(param_str)

        else:
            if param_ir.annotation_ir is not None:
                assert codegen is not None
                ann = _parameter_annotation_str(codegen, current_target_module)
                ann = quote_annotation_for_local_names(ann, annotation_local_names)
                param_str = f"{name}: {ann}"

                if codegen.requires_boundary_translation():
                    wrapper_to_impl_expr = _wrapper_to_impl_expr(codegen, name)
                    unwrap_stmts.append(f"{unwrap_indent}{name}_impl = {wrapper_to_impl_expr}")
                    call_args.append(f"{name}_impl")
                else:
                    call_args.append(name)
            else:
                param_str = name
                call_args.append(name)

            if param_ir.default_expr is not None:
                param_str += f" = {param_ir.default_expr}"

            params.append(param_str)

        if not positional_only_marker_added and last_positional_only_index >= 0:
            if (
                kind != inspect.Parameter.POSITIONAL_ONLY
                and kind != inspect.Parameter.VAR_POSITIONAL
                and len(params) > last_positional_only_index
            ):
                params.insert(last_positional_only_index + 1, "/")
                positional_only_marker_added = True

    param_str = ", ".join(params)
    call_args_str = ", ".join(call_args)
    unwrap_code = "\n".join(unwrap_stmts) if unwrap_stmts else ""

    return param_str, call_args_str, unwrap_code


def _unwrap_to_self_codegen(codegen):
    """If the effective return type is ``typing.Self``, return ``SelfTypeCodegen``; else ``None``."""
    from .type_codegen import AwaitableTypeCodegen, CoroutineTypeCodegen, SelfTypeCodegen

    t = codegen
    while isinstance(t, (AwaitableTypeCodegen, CoroutineTypeCodegen)):
        t = t.return_codegen
    return t if isinstance(t, SelfTypeCodegen) else None


def _effective_inner_codegen(codegen):
    from .type_codegen import AwaitableTypeCodegen, CoroutineTypeCodegen

    t = codegen
    while isinstance(t, (AwaitableTypeCodegen, CoroutineTypeCodegen)):
        t = t.return_codegen
    return t


def _build_call_with_wrap(
    call_expr: str,
    return_codegen,
    current_target_module: str,
    indent: str = "    ",
    is_async: bool = True,
    *,
    is_function: bool = False,
    method_type: MethodBindingKind | None = None,
    method_owner_impl_ref: ObjectReferenceIR | None = None,
) -> str:
    """Build a function call with optional return value wrapping."""
    from .type_codegen import (
        AsyncContextManagerTypeCodegen,
        AwaitableTypeCodegen,
        CoroutineTypeCodegen,
    )

    def _adjust_method_helper_reference(expr: str) -> str:
        if not isinstance(return_codegen, AsyncContextManagerTypeCodegen):
            return expr
        if method_type == MethodBindingKind.CLASSMETHOD:
            return expr.replace("self.", "cls.")
        if method_type == MethodBindingKind.STATICMETHOD and method_owner_impl_ref is not None:
            wrapper_name = method_owner_impl_ref.qualname.rpartition(".")[2]
            return expr.replace("self.", f"{wrapper_name}.")
        return expr

    def _wrap_result_expr(outer_codegen) -> str:
        st = _unwrap_to_self_codegen(outer_codegen)
        if st is not None and method_type is not None:
            return st.impl_to_wrapper_expr_for_method(
                current_target_module,
                "result",
                is_async=is_async,
                method_type=method_type,
            )
        inner = outer_codegen
        while isinstance(inner, (AwaitableTypeCodegen, CoroutineTypeCodegen)):
            inner = inner.return_codegen
        return inner.impl_to_wrapper_expr(current_target_module, "result", is_async=is_async)

    # Check if this is an awaitable type that needs synchronizer wrapping
    if isinstance(return_codegen, (AwaitableTypeCodegen, CoroutineTypeCodegen)):
        # Wrap the call with synchronizer to await/run it
        if is_async:
            wrapped_call = f"await _synchronizer._run_function_async({call_expr})"
        else:
            wrapped_call = f"_synchronizer._run_function_sync({call_expr})"

        inner_codegen = return_codegen.return_codegen
        if inner_codegen.requires_boundary_translation():
            eff = _effective_inner_codegen(return_codegen)
            if (
                method_type == MethodBindingKind.INSTANCE
                and not is_function
                and method_owner_impl_ref is not None
                and isinstance(eff, WrappedClassTypeCodegen)
                and eff.impl_ref == method_owner_impl_ref
            ):
                impl_to_wrapper_expr = "self._from_impl(result)"
            else:
                impl_to_wrapper_expr = _wrap_result_expr(return_codegen)
            impl_to_wrapper_expr = _adjust_method_helper_reference(impl_to_wrapper_expr)
            if is_function:
                impl_to_wrapper_expr = impl_to_wrapper_expr.replace("self.", "")
            return f"""{indent}result = {wrapped_call}
{indent}return {impl_to_wrapper_expr}"""
        else:
            return f"{indent}return {wrapped_call}"

    # Regular wrapping for non-awaitable types
    if return_codegen.requires_boundary_translation():
        eff = _effective_inner_codegen(return_codegen)
        if (
            method_type == MethodBindingKind.INSTANCE
            and not is_function
            and method_owner_impl_ref is not None
            and isinstance(eff, WrappedClassTypeCodegen)
            and eff.impl_ref == method_owner_impl_ref
        ):
            impl_to_wrapper_expr = "self._from_impl(result)"
        else:
            impl_to_wrapper_expr = _wrap_result_expr(return_codegen)
        impl_to_wrapper_expr = _adjust_method_helper_reference(impl_to_wrapper_expr)
        if is_function:
            impl_to_wrapper_expr = impl_to_wrapper_expr.replace("self.", "")
        return f"""{indent}result = {call_expr}
{indent}return {impl_to_wrapper_expr}"""
    else:
        return f"{indent}return {call_expr}"


def _format_return_annotation(
    return_codegen,
    current_target_module: str,
    *,
    annotation_local_names: typing.AbstractSet[str] | None = None,
) -> tuple[str, str]:
    """Format return type annotations for both sync and async versions."""

    sync_return_type = return_codegen.annotation_type(current_target_module, is_async=False)
    async_return_type = return_codegen.annotation_type(current_target_module, is_async=True)
    sync_return_type = quote_annotation_for_local_names(sync_return_type, annotation_local_names)
    async_return_type = quote_annotation_for_local_names(async_return_type, annotation_local_names)

    if not sync_return_type:
        return "", ""

    sync_return_str = f" -> {sync_return_type}"
    async_return_str = f" -> {async_return_type}"

    return sync_return_str, async_return_str
