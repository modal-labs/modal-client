# Copyright Modal Labs 2026
"""Parse and validate source-level parameter default expressions."""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import sys
import textwrap
import types
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedDefaultExpression:
    expression: str
    import_modules: tuple[str, ...] = ()


def _source_label_for_parameter(source_label_prefix: str | None, parameter_name: str) -> str:
    if source_label_prefix is not None:
        return f"{source_label_prefix} parameter {parameter_name!r}"
    return f"parameter {parameter_name!r}"


def _evaluate_expression(expression: str, globals_dict: dict[str, object]) -> object:
    code = compile(expression, "<default_expr>", "eval")
    return eval(code, globals_dict, {})


def _values_match(expected: object, actual: object) -> bool:
    if type(actual) is not type(expected):
        return False

    try:
        return actual == expected
    except Exception:
        return False


def _globals_with_builtins() -> dict[str, object]:
    return {"__builtins__": vars(builtins)}


def _globals_for_module_path(module_path: str) -> dict[str, object]:
    importlib.import_module(module_path)
    globals_dict = _globals_with_builtins()
    top_level_name = module_path.split(".", 1)[0]
    top_level_module = sys.modules.get(top_level_name)
    if top_level_module is None:
        raise ImportError(f"Could not import top-level module {top_level_name!r}")
    globals_dict[top_level_name] = top_level_module
    return globals_dict


def _try_resolve_expression(
    expression: str,
    *,
    expected_value: object,
    globals_dict: dict[str, object],
) -> str | None:
    try:
        resolved = _evaluate_expression(expression, globals_dict)
    except Exception:
        return None
    if _values_match(expected_value, resolved):
        return expression
    return None


def _top_level_name_for_importable_expression(expression: str) -> str | None:
    if "." not in expression:
        return None

    parsed = ast.parse(expression, mode="eval")
    names = sorted({node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)})
    if len(names) != 1:
        return None
    return names[0]


def _resolve_default_expression(
    expression: str,
    *,
    expected_value: object,
    impl_module: types.ModuleType,
    source_label: str,
) -> ResolvedDefaultExpression:
    verbatim_globals = _globals_with_builtins()
    resolved_verbatim = _try_resolve_expression(
        expression,
        expected_value=expected_value,
        globals_dict=verbatim_globals,
    )
    if resolved_verbatim is not None:
        return ResolvedDefaultExpression(expression=resolved_verbatim)

    failure_messages = [f"verbatim expression {expression!r} is not safely executable"]

    importable_name = _top_level_name_for_importable_expression(expression)
    if importable_name is not None:
        try:
            module_globals = _globals_for_module_path(importable_name)
        except Exception as exc:
            failure_messages.append(f"import {importable_name!r} failed: {exc}")
        else:
            resolved_with_import = _try_resolve_expression(
                expression,
                expected_value=expected_value,
                globals_dict=module_globals,
            )
            if resolved_with_import is not None:
                return ResolvedDefaultExpression(
                    expression=resolved_with_import,
                    import_modules=(importable_name,),
                )
            failure_messages.append(
                f"expression {expression!r} did not match runtime default after importing {importable_name!r}"
            )

    prefixed_expression = f"{impl_module.__name__}.{expression}"
    try:
        prefixed_globals = _globals_for_module_path(impl_module.__name__)
    except Exception as exc:
        failure_messages.append(f"import implementation module {impl_module.__name__!r} failed: {exc}")
    else:
        resolved_prefixed = _try_resolve_expression(
            prefixed_expression,
            expected_value=expected_value,
            globals_dict=prefixed_globals,
        )
        if resolved_prefixed is not None:
            return ResolvedDefaultExpression(expression=resolved_prefixed)
        failure_messages.append(
            f"expression {prefixed_expression!r} did not match runtime default in implementation module scope"
        )

    raise TypeError(f"{source_label} has unsupported default expression {expression!r}: " + "; ".join(failure_messages))


def _function_node_for_source(
    source: str,
    *,
    function_name: str,
    signature: inspect.Signature,
    source_label_prefix: str | None,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    parsed = ast.parse(textwrap.dedent(source))
    expected_parameter_names = tuple(signature.parameters)

    function_nodes = [
        node
        for node in ast.walk(parsed)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]
    for node in function_nodes:
        node_parameter_names = tuple(
            [arg.arg for arg in node.args.posonlyargs]
            + [arg.arg for arg in node.args.args]
            + ([node.args.vararg.arg] if node.args.vararg is not None else [])
            + [arg.arg for arg in node.args.kwonlyargs]
            + ([node.args.kwarg.arg] if node.args.kwarg is not None else [])
        )
        if node_parameter_names == expected_parameter_names:
            return node

    label = source_label_prefix or function_name
    raise TypeError(f"Could not locate source definition for {label!r} while parsing default expressions")


def _extract_source_default_expressions(
    func: types.FunctionType,
    sig: inspect.Signature,
    *,
    source_label_prefix: str | None,
) -> dict[str, str]:
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as exc:
        label = source_label_prefix or f"{func.__module__}.{func.__qualname__}"
        raise TypeError(f"Could not recover source for {label!r} while parsing default expressions") from exc

    function_node = _function_node_for_source(
        source,
        function_name=func.__name__,
        signature=sig,
        source_label_prefix=source_label_prefix,
    )

    defaults: dict[str, str] = {}
    positional_parameters = [*function_node.args.posonlyargs, *function_node.args.args]
    positional_defaults = function_node.args.defaults
    positional_offset = len(positional_parameters) - len(positional_defaults)
    for index, default_node in enumerate(positional_defaults):
        parameter_name = positional_parameters[positional_offset + index].arg
        positional_segment = ast.get_source_segment(textwrap.dedent(source), default_node)
        if positional_segment is None:
            raise TypeError(
                f"{_source_label_for_parameter(source_label_prefix, parameter_name)} default expression could not "
                "be extracted from source"
            )
        defaults[parameter_name] = positional_segment.strip()

    for kwarg, keyword_default_node in zip(function_node.args.kwonlyargs, function_node.args.kw_defaults):
        if keyword_default_node is None:
            continue
        keyword_segment = ast.get_source_segment(textwrap.dedent(source), keyword_default_node)
        if keyword_segment is None:
            raise TypeError(
                f"{_source_label_for_parameter(source_label_prefix, kwarg.arg)} default expression could not "
                "be extracted from source"
            )
        defaults[kwarg.arg] = keyword_segment.strip()

    return defaults


def resolve_parameter_default_expressions(
    func: types.FunctionType,
    sig: inspect.Signature,
    *,
    impl_module: types.ModuleType,
    source_label_prefix: str | None = None,
) -> dict[str, ResolvedDefaultExpression]:
    if all(parameter.default is inspect.Parameter.empty for parameter in sig.parameters.values()):
        return {}

    source_defaults = _extract_source_default_expressions(
        func,
        sig,
        source_label_prefix=source_label_prefix,
    )

    resolved: dict[str, ResolvedDefaultExpression] = {}
    for parameter_name, parameter in sig.parameters.items():
        if parameter.default is inspect.Parameter.empty:
            continue

        source_expression = source_defaults.get(parameter_name)
        if source_expression is None:
            raise TypeError(
                f"{_source_label_for_parameter(source_label_prefix, parameter_name)} default expression "
                "is missing from source extraction"
            )

        resolved[parameter_name] = _resolve_default_expression(
            source_expression,
            expected_value=parameter.default,
            impl_module=impl_module,
            source_label=_source_label_for_parameter(source_label_prefix, parameter_name),
        )

    return resolved
