# Copyright Modal Labs 2023
"""Load or import Python modules from the CLI.

For example, the function reference of `modal run some_file.py::app.foo_func`
or the app lookup of `modal deploy some_file.py`.

These functions are only called by the Modal CLI, not in tasks.
"""

import dataclasses
import importlib
import importlib.util
import inspect
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, cast

import click
from rich.console import Console
from rich.markdown import Markdown

from modal.app import App, LocalEntrypoint
from modal.cls import Cls
from modal.exception import InvalidError, _CliUserExecutionError
from modal.functions import Function


@dataclasses.dataclass
class ImportRef:
    file_or_module: str

    # object_path is a .-delimited path to the object to execute, or a parent from which to infer the object
    # e.g.
    # function or local_entrypoint in module scope
    # app in module scope [+ method name]
    # app [+ function/entrypoint on that app]
    object_path: Optional[str]


def parse_import_ref(object_ref: str) -> ImportRef:
    if object_ref.find("::") > 1:
        file_or_module, object_path = object_ref.split("::", 1)
    elif object_ref.find(":") > 1:
        raise InvalidError(f"Invalid object reference: {object_ref}. Did you mean '::' instead of ':'?")
    else:
        file_or_module, object_path = object_ref, None

    return ImportRef(file_or_module, object_path)


DEFAULT_APP_NAME = "app"


def import_file_or_module(file_or_module: str):
    if "" not in sys.path:
        # When running from a CLI like `modal run`
        # the current working directory isn't added to sys.path
        # so we add it in order to make module path specification possible
        sys.path.insert(0, "")  # "" means the current working directory

    if file_or_module.endswith(".py"):
        # when using a script path, that scripts directory should also be on the path as it is
        # with `python some/script.py`
        full_path = Path(file_or_module).resolve()
        if "." in full_path.name.removesuffix(".py"):
            raise InvalidError(
                f"Invalid Modal source filename: {full_path.name!r}."
                "\n\nSource filename cannot contain additional period characters."
            )
        sys.path.insert(0, str(full_path.parent))

        module_name = inspect.getmodulename(file_or_module)
        assert module_name is not None
        # Import the module - see https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        spec = importlib.util.spec_from_file_location(module_name, file_or_module)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            assert spec.loader
            spec.loader.exec_module(module)
        except Exception as exc:
            raise _CliUserExecutionError(str(full_path)) from exc
    else:
        try:
            module = importlib.import_module(file_or_module)
        except Exception as exc:
            raise _CliUserExecutionError(file_or_module) from exc

    return module


@dataclass(frozen=True)
class MethodReference:
    """This helps with deferring method reference until after the class gets instantiated by the CLI"""

    cls: Cls
    method_name: str


def list_runnables(
    module: types.ModuleType,
) -> list[tuple[list[str], Union[Function, MethodReference, LocalEntrypoint]]]:
    """
    Extracts all runnables found either directly in the input module, or in any of the Apps listed in that module

    Runnables includes all Functions, (class) Methods and Local Entrypoints, including web endpoints.

    The returned list consists of tuples:
    ([name1, name2...], Runnable)

    Where the first name is always the module level name if such a name exists
    """
    apps = cast(list[tuple[str, App]], inspect.getmembers(module, lambda x: isinstance(x, App)))

    all_runnables = {}
    for app_name, app in apps:
        for name, local_entrypoint in app.registered_entrypoints.items():
            all_runnables.setdefault(local_entrypoint, []).append(f"{app_name}.{name}")
        for name, function in app.registered_functions.items():
            if name.endswith(".*"):
                continue
            all_runnables.setdefault(function, []).append(f"{app_name}.{name}")
        for cls_name, cls in app.registered_classes.items():
            for method_name in cls._get_method_names():
                method_ref = MethodReference(cls, method_name)
                all_runnables.setdefault(method_ref, []).append(f"{app_name}.{cls_name}.{method_name}")

    # If any class or function is exported as a module level object, use that
    # as the preferred name by putting it first in the list
    module_level_entities = cast(
        list[tuple[str, Union[Function, Cls, LocalEntrypoint]]],
        inspect.getmembers(module, lambda x: isinstance(x, (Function, Cls, LocalEntrypoint))),
    )
    for name, entity in module_level_entities:
        if isinstance(entity, Cls):
            for method_name in entity._get_method_names():
                method_ref = MethodReference(entity, method_name)
                all_runnables.setdefault(method_ref, []).insert(0, f"{name}.{method_name}")
        else:
            all_runnables.setdefault(entity, []).insert(0, name)

    return [(names, runnable) for runnable, names in all_runnables.items()]


def get_by_object_path(obj: Any, obj_path: str) -> Union[Function, LocalEntrypoint, MethodReference, App, None]:
    # Try to evaluate a `.`-delimited object path in a Modal context
    # With the caveat that some object names can actually have `.` in their name (lifecycled methods' tags)

    # Note: this is eager, so no backtracking is performed in case an
    # earlier match fails at some later point in the path expansion
    prefix = ""
    obj_path_segments = obj_path.split(".")
    for i, segment in enumerate(obj_path_segments):
        attr = prefix + segment
        if isinstance(obj, App):
            if attr in obj.registered_entrypoints:
                # local entrypoints can't be accessed via getattr
                obj = obj.registered_entrypoints[attr]
                continue
        if isinstance(obj, Cls):
            remaining_segments = obj_path_segments[i:]
            remaining_path = ".".join(remaining_segments)
            if len(remaining_segments) > 1:
                raise ValueError(f"{obj._get_name()} is a class, but {remaining_path} is not a method reference")
            # TODO: add method inference here?
            return MethodReference(obj, remaining_path)

        try:
            obj = getattr(obj, attr)
        except Exception:
            prefix = f"{prefix}{segment}."
        else:
            prefix = ""

    if prefix:
        return None

    return obj


def _infer_function_or_help(
    app: App, module, accept_local_entrypoint: bool, accept_webhook: bool
) -> Union[Function, LocalEntrypoint, MethodReference]:
    """Using only an app - automatically infer a single "runnable" for a `modal run` invocation

    If a single runnable can't be determined, show CLI help indicating valid choices.
    """
    filtered_local_entrypoints = [
        entrypoint
        for entrypoint in app.registered_entrypoints.values()
        if entrypoint.info.module_name == module.__name__
    ]

    if accept_local_entrypoint:
        if len(filtered_local_entrypoints) == 1:
            # If there is just a single local entrypoint in the target module, use
            # that regardless of other functions.
            return filtered_local_entrypoints[0]
        elif len(app.registered_entrypoints) == 1:
            # Otherwise, if there is just a single local entrypoint in the app as a whole,
            # use that one.
            return list(app.registered_entrypoints.values())[0]

    # TODO: refactor registered_functions to only contain function services, not class services
    function_choices: dict[str, Union[Function, LocalEntrypoint, MethodReference]] = {
        name: f for name, f in app.registered_functions.items() if not name.endswith(".*")
    }
    for cls_name, cls in app.registered_classes.items():
        for method_name in cls._get_method_names():
            function_choices[f"{cls_name}.{method_name}"] = MethodReference(cls, method_name)

    if not accept_webhook:
        for web_endpoint_name in app.registered_web_endpoints:
            function_choices.pop(web_endpoint_name, None)

    if accept_local_entrypoint:
        function_choices.update(app.registered_entrypoints)

    if len(function_choices) == 1:
        return list(function_choices.values())[0]

    if len(function_choices) == 0:
        if app.registered_web_endpoints:
            err_msg = "Modal app has only web endpoints. Use `modal serve` instead of `modal run`."
        else:
            err_msg = "Modal app has no registered functions. Nothing to run."
        raise click.UsageError(err_msg)

    # there are multiple choices - we can't determine which one to use:
    registered_functions_str = "\n".join(sorted(function_choices))
    help_text = f"""You need to specify a Modal function or local entrypoint to run, e.g.

modal run app.py::my_function [...args]

Registered functions and local entrypoints on the selected app are:
{registered_functions_str}
"""
    raise click.UsageError(help_text)


def _show_no_auto_detectable_app(app_ref: ImportRef) -> None:
    object_path = app_ref.object_path
    import_path = app_ref.file_or_module
    error_console = Console(stderr=True)
    error_console.print(f"[bold red]Could not find Modal app '{object_path}' in {import_path}.[/bold red]")

    if object_path is None:
        guidance_msg = (
            f"Expected to find an app variable named **`{DEFAULT_APP_NAME}`** (the default app name). "
            "If your `modal.App` is named differently, "
            "you must specify it in the app ref argument. "
            f"For example an App variable `app_2 = modal.App()` in `{import_path}` would "
            f"be specified as `{import_path}::app_2`."
        )
        md = Markdown(guidance_msg)
        error_console.print(md)


def import_app(app_ref: str) -> App:
    import_ref = parse_import_ref(app_ref)

    module = import_file_or_module(import_ref.file_or_module)
    app = get_by_object_path(module, import_ref.object_path or DEFAULT_APP_NAME)

    if app is None:
        _show_no_auto_detectable_app(import_ref)
        sys.exit(1)

    if not isinstance(app, App):
        raise click.UsageError(f"{app} is not a Modal App")

    return app


def _show_function_ref_help(app_ref: ImportRef, base_cmd: str) -> None:
    object_path = app_ref.object_path
    import_path = app_ref.file_or_module
    error_console = Console(stderr=True)
    if object_path:
        error_console.print(
            f"[bold red]Could not find Modal function or local entrypoint"
            f" '{object_path}' in '{import_path}'.[/bold red]"
        )
    else:
        error_console.print(
            f"[bold red]No function was specified, and no [green]`app`[/green] variable "
            f"could be found in '{import_path}'.[/bold red]"
        )
    guidance_msg = f"""
Usage:
{base_cmd} <file_or_module_path>::<function_name>

Given the following example `app.py`:
```
app = modal.App()

@app.function()
def foo():
    ...
```
You would run foo as [bold green]{base_cmd} app.py::foo[/bold green]"""
    error_console.print(guidance_msg)


def _import_object(func_ref, base_cmd):
    import_ref = parse_import_ref(func_ref)
    module = import_file_or_module(import_ref.file_or_module)
    app_function_or_method_ref = get_by_object_path(module, import_ref.object_path or DEFAULT_APP_NAME)

    if app_function_or_method_ref is None:
        _show_function_ref_help(import_ref, base_cmd)
        raise SystemExit(1)

    return app_function_or_method_ref, module


def _infer_runnable(
    partial_obj: Union[App, Function, MethodReference, LocalEntrypoint],
    module: types.ModuleType,
    accept_local_entrypoint: bool = True,
    accept_webhook: bool = False,
) -> tuple[App, Union[Function, MethodReference, LocalEntrypoint]]:
    if isinstance(partial_obj, App):
        # infer function or display help for how to select one
        app = partial_obj
        function_handle = _infer_function_or_help(app, module, accept_local_entrypoint, accept_webhook)
        return app, function_handle
    elif isinstance(partial_obj, Function):
        return partial_obj.app, partial_obj
    elif isinstance(partial_obj, MethodReference):
        return partial_obj.cls._get_app(), partial_obj
    elif isinstance(partial_obj, LocalEntrypoint):
        if not accept_local_entrypoint:
            raise click.UsageError(
                f"{partial_obj.info.function_name} is not a Modal Function "
                f"(a Modal local_entrypoint can't be used in this context)"
            )
        return partial_obj.app, partial_obj
    else:
        raise click.UsageError(
            f"{partial_obj} is not a Modal entity (should be an App, Local entrypoint, " "Function or Class/Method)"
        )


def import_and_infer(
    func_ref: str, base_cmd: str, accept_local_entrypoint=True, accept_webhook=False
) -> tuple[App, Union[Function, LocalEntrypoint, MethodReference]]:
    """Takes a function ref string and returns something "runnable"

    The function ref can leave out partial information (apart from the file name) as
    long as the runnable is uniquely identifiable by the provided information.

    When there are multiple runnables within the provided ref, the following rules should
    be followed:

    1. if there is a single local_entrypoint, that one is used
    2. if there is a single {function, class} that one is used
    3. if there is a single method (within a class) that one is used
    """
    app_function_or_method_ref, module = _import_object(func_ref, base_cmd)
    return _infer_runnable(app_function_or_method_ref, module, accept_local_entrypoint, accept_webhook)
