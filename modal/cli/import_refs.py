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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

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
        # Import the module - see https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        spec = importlib.util.spec_from_file_location(module_name, file_or_module)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise _CliUserExecutionError(str(full_path)) from exc
    else:
        try:
            module = importlib.import_module(file_or_module)
        except Exception as exc:
            raise _CliUserExecutionError(file_or_module) from exc

    return module


@dataclass
class MethodReference:
    cls: Cls
    method_name: str


def get_by_object_path(obj: Any, obj_path: str) -> Union[Function, LocalEntrypoint, MethodReference, None]:
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
                # local entrypoints are not on stub blueprint
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
    function_choices = set(app.registered_functions)

    if not accept_webhook:
        function_choices -= set(app.registered_web_endpoints)
    if accept_local_entrypoint:
        function_choices |= set(app.registered_entrypoints.keys())

    sorted_function_choices = sorted(function_choices)

    filtered_local_entrypoints = [
        name
        for name, entrypoint in app.registered_entrypoints.items()
        if entrypoint.info.module_name == module.__name__
    ]

    if accept_local_entrypoint and len(filtered_local_entrypoints) == 1:
        # If there is just a single local entrypoint in the target module, use
        # that regardless of other functions.
        function_name = list(filtered_local_entrypoints)[0]
    elif accept_local_entrypoint and len(app.registered_entrypoints) == 1:
        # Otherwise, if there is just a single local entrypoint in the stub as a whole,
        # use that one.
        function_name = list(app.registered_entrypoints.keys())[0]
    elif len(function_choices) == 1:
        function_name = sorted_function_choices[0]
    elif len(function_choices) == 0:
        if app.registered_web_endpoints:
            err_msg = "Modal app has only web endpoints. Use `modal serve` instead of `modal run`."
        else:
            err_msg = "Modal app has no registered functions. Nothing to run."
        raise click.UsageError(err_msg)
    else:
        registered_functions_str = "\n".join(sorted_function_choices)
        help_text = f"""You need to specify a Modal function or local entrypoint to run, e.g.

modal run app.py::my_function [...args]

Registered functions and local entrypoints on the selected app are:
{registered_functions_str}
"""
        raise click.UsageError(help_text)

    if function_name in app.registered_entrypoints:
        # entrypoint is in entrypoint registry, for now
        return app.registered_entrypoints[function_name]

    function = app.registered_functions[function_name]
    assert isinstance(function, Function)
    return function


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


def import_object(
    func_ref: str, base_cmd: str, accept_local_entrypoint=True, accept_webhook=False
) -> Union[Function, LocalEntrypoint, MethodReference]:
    """Takes a function ref string and returns something "runnable"

    The function ref can leave out partial information (apart from the file name) as
    long as the runnable is uniquely identifiable by the provided information.

    When there are multiple runnables within the provided ref, the following rules should
    be followed:

    1. if there is a single local_entrypoint, that one is used
    2. if there is a single {function, class} that one is used
    3. if there is a single method (within a class) that one is used
    """
    import_ref = parse_import_ref(func_ref)

    module = import_file_or_module(import_ref.file_or_module)
    app_function_or_method_ref = get_by_object_path(module, import_ref.object_path or DEFAULT_APP_NAME)

    if app_function_or_method_ref is None:
        _show_function_ref_help(import_ref, base_cmd)
        sys.exit(1)

    if isinstance(app_function_or_method_ref, App):
        # infer function or display help for how to select one
        app = app_function_or_method_ref
        function_handle = _infer_function_or_help(app, module, accept_local_entrypoint, accept_webhook)
        return function_handle
    elif isinstance(app_function_or_method_ref, (Function, MethodReference)):
        return app_function_or_method_ref
    elif isinstance(app_function_or_method_ref, LocalEntrypoint):
        if not accept_local_entrypoint:
            raise click.UsageError(
                f"{func_ref} is not a Modal Function (a Modal local_entrypoint can't be used in this context)"
            )
        return app_function_or_method_ref
    else:
        raise click.UsageError(f"{app_function_or_method_ref} is not a Modal entity (should be an App or Function)")
