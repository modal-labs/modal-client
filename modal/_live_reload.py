# Copyright Modal Labs 2023
import ast
import inspect
import sys
from typing import Dict, List, Optional, Union

from .functions import _FunctionHandle

LiveReloadResult = Union[List[str], Exception, None]


def process_change(stub, file_changeset) -> Dict[str, LiveReloadResult]:
    """
    Accepts a set of FileChange events and reloads any Modal provider functions
    contained within changed Python modules.
    """
    results = {}
    for file_change in file_changeset:
        change_type, filepath = file_change
        if filepath.endswith(".py"):
            results[filepath] = _reload_providers(stub, filepath)
        else:
            results[filepath] = None
    return results


def _is_stub_assignment(node: ast.Assign) -> bool:
    if not (
        isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "modal"
        and node.value.func.attr == "Stub"
    ):
        return False
    return True


def _is_reloadable_stub_method(stub_name: Optional[str], stub_obj, node: ast.Call) -> bool:
    # TODO: Cache. Don't recalculate.
    methods_to_reload = {
        attr
        for attr in dir(stub_obj)
        if not attr.startswith("_")
        and callable(getattr(stub_obj, attr))
        and inspect.signature(getattr(stub_obj, attr)).return_annotation == _FunctionHandle
    }
    if not (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and (not stub_name or node.func.value.id == stub_name)
        and node.func.attr in methods_to_reload
    ):
        return False

    return True


def _reload_providers(stub_obj, filepath: str) -> LiveReloadResult:
    with open(filepath, "r") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return exc

    stub_name = None
    reloaded_fns = []
    # TODO: Is visitor-pattern more efficient?
    for node in ast.walk(tree):
        # If an obvious modal.Stub assignment in the source is available in the AST
        # use it to determine the stub name.
        if isinstance(node, ast.Assign) and _is_stub_assignment(node):
            stub_name = node.targets[0].id  # type: ignore
        if isinstance(node, ast.FunctionDef):
            fn_name = node.name
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and _is_reloadable_stub_method(stub_name, stub_obj, dec):
                    # Reevaluate only relevant Modal function definitions, don't re-execute the entire
                    # module which may include a myriad side-effects.
                    try:
                        comp_unit = ast.Module(body=[node], type_ignores=[])
                        compiled_code = compile(comp_unit, filename=filepath, mode="exec")
                    except SyntaxError as exc:
                        return exc

                    for name, mod in sys.modules.items():
                        if hasattr(mod, "__file__") and mod.__file__ == filepath:
                            exec(compiled_code, mod.__dict__)
                            reloaded_fns.append(fn_name)
                            break
    return reloaded_fns
