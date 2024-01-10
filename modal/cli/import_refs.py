# Copyright Modal Labs 2023
"""Utilities for CLI references to Modal entities

For example, the function reference of `modal run some_file.py::stub.foo_func`
or the stub lookup of `modal deploy some_file.py`
"""

import dataclasses
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Optional, Union

import click
from rich.console import Console
from rich.markdown import Markdown

import modal
from modal.functions import Function
from modal.stub import LocalEntrypoint, Stub


@dataclasses.dataclass
class ImportRef:
    file_or_module: str
    object_path: Optional[str]


def parse_import_ref(object_ref: str) -> ImportRef:
    if object_ref.find("::") > 1:
        file_or_module, object_path = object_ref.split("::", 1)
    elif object_ref.find(":") > 1:
        raise modal.exception.InvalidError(f"Invalid object reference: {object_ref}. Did you mean '::' instead of ':'?")
    else:
        file_or_module, object_path = object_ref, None

    return ImportRef(file_or_module, object_path)


class NoSuchObject(modal.exception.NotFoundError):
    pass


DEFAULT_STUB_NAME = "stub"


def import_file_or_module(file_or_module: str):
    if "" not in sys.path:
        # This seems to happen when running from a CLI
        sys.path.insert(0, "")
    if file_or_module.endswith(".py"):
        # walk to the closest python package in the path and add that to the path
        # before importing, in case of imports etc. of other modules in that package
        # are needed

        # Let's first assume this is not part of any package
        module_name = inspect.getmodulename(file_or_module)

        # Look for any __init__.py in a parent directory and maybe change the module name
        directory = Path(file_or_module).parent
        module_path = [inspect.getmodulename(file_or_module)]
        while directory.parent != directory:
            parent = directory.parent
            module_path.append(directory.name)
            if (directory / "__init__.py").exists():
                # We identified a package, let's store a new module name
                module_name = ".".join(reversed(module_path))
            directory = parent

        # Import the module - see https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        spec = importlib.util.spec_from_file_location(module_name, file_or_module)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(file_or_module)

    return module


def get_by_object_path(obj: Any, obj_path: str):
    # Try to evaluate a `.`-delimited object path in a Modal context
    # With the caveat that some object names can actually have `.` in their name (lifecycled methods' tags)

    # Note: this is eager, so no backtracking is performed in case an
    # earlier match fails at some later point in the path expansion
    orig_obj = obj
    prefix = ""
    for segment in obj_path.split("."):
        attr = prefix + segment
        try:
            if isinstance(obj, Stub):
                if attr in obj.registered_entrypoints:
                    # local entrypoints are not on stub blueprint
                    obj = obj.registered_entrypoints[attr]
                    continue
            obj = getattr(obj, attr)

        except Exception:
            prefix = f"{prefix}{segment}."
        else:
            prefix = ""

    if prefix:
        raise NoSuchObject(f"No object {obj_path} could be found in module {orig_obj}")

    return obj


def infer_function_or_help(
    stub: Stub, module, accept_local_entrypoint: bool, accept_webhook: bool
) -> Union[Function, LocalEntrypoint]:
    function_choices = set(stub.registered_functions.keys())
    if not accept_webhook:
        function_choices -= set(stub.registered_web_endpoints)
    if accept_local_entrypoint:
        function_choices |= set(stub.registered_entrypoints.keys())

    sorted_function_choices = sorted(function_choices)
    registered_functions_str = "\n".join(sorted_function_choices)
    filtered_local_entrypoints = [
        name
        for name, entrypoint in stub.registered_entrypoints.items()
        if entrypoint.info.module_name == module.__name__
    ]

    if accept_local_entrypoint and len(filtered_local_entrypoints) == 1:
        # If there is just a single local entrypoint in the target module, use
        # that regardless of other functions.
        function_name = list(filtered_local_entrypoints)[0]
    elif accept_local_entrypoint and len(stub.registered_entrypoints) == 1:
        # Otherwise, if there is just a single local entrypoint in the stub as a whole,
        # use that one.
        function_name = list(stub.registered_entrypoints.keys())[0]
    elif len(function_choices) == 1:
        function_name = sorted_function_choices[0]
    elif len(function_choices) == 0:
        if stub.registered_web_endpoints:
            err_msg = "Modal stub has only webhook functions. Use `modal serve` instead of `modal run`."
        else:
            err_msg = "Modal stub has no registered functions. Nothing to run."
        raise click.UsageError(err_msg)
    else:
        help_text = f"""You need to specify a Modal function or local entrypoint to run, e.g.

modal run app.py::my_function [...args]

Registered functions and local entrypoints on the selected stub are:
{registered_functions_str}
"""
        raise click.UsageError(help_text)

    if function_name in stub.registered_entrypoints:
        # entrypoint is in entrypoint registry, for now
        return stub.registered_entrypoints[function_name]

    return stub[function_name]  # functions are in blueprint


def _show_no_auto_detectable_stub(stub_ref: ImportRef) -> None:
    object_path = stub_ref.object_path
    import_path = stub_ref.file_or_module
    error_console = Console(stderr=True)
    error_console.print(f"[bold red]Could not find Modal stub '{object_path}' in {import_path}.[/bold red]")

    if object_path is None:
        guidance_msg = (
            f"Expected to find a stub variable named **`{DEFAULT_STUB_NAME}`** (the default stub name). If your `modal.Stub` is named differently, "
            "you must specify it in the stub ref argument. "
            f"For example a stub variable `app_stub = modal.Stub()` in `{import_path}` would "
            f"be specified as `{import_path}::app_stub`."
        )
        md = Markdown(guidance_msg)
        error_console.print(md)


def import_stub(stub_ref: str) -> Stub:
    import_ref = parse_import_ref(stub_ref)
    try:
        module = import_file_or_module(import_ref.file_or_module)
        obj_path = import_ref.object_path or DEFAULT_STUB_NAME  # get variable named "stub" by default
        stub = get_by_object_path(module, obj_path)
    except NoSuchObject:
        _show_no_auto_detectable_stub(import_ref)
        sys.exit(1)

    if not isinstance(stub, Stub):
        raise click.UsageError(f"{stub} is not a Modal Stub")

    return stub


def _show_function_ref_help(stub_ref: ImportRef, base_cmd: str) -> None:
    object_path = stub_ref.object_path
    import_path = stub_ref.file_or_module
    error_console = Console(stderr=True)
    if object_path:
        error_console.print(
            f"[bold red]Could not find Modal function or local entrypoint '{object_path}' in '{import_path}'.[/bold red]"
        )
    else:
        error_console.print(
            f"[bold red]No function was specified, and no [green]`stub`[/green] variable could be found in '{import_path}'.[/bold red]"
        )
    guidance_msg = f"""
Usage:
{base_cmd} <file_or_module_path>::<function_name>

Given the following example `app.py`:
```
stub = modal.Stub()

@stub.function()
def foo():
    ...
```
You would run foo as [bold green]{base_cmd} app.py::foo[/bold green]"""
    error_console.print(guidance_msg)


def import_function(
    func_ref: str, base_cmd: str, accept_local_entrypoint=True, accept_webhook=False
) -> Union[Function, LocalEntrypoint]:
    import_ref = parse_import_ref(func_ref)
    try:
        module = import_file_or_module(import_ref.file_or_module)
        obj_path = import_ref.object_path or DEFAULT_STUB_NAME  # get variable named "stub" by default
        stub_or_function = get_by_object_path(module, obj_path)
    except NoSuchObject:
        _show_function_ref_help(import_ref, base_cmd)
        sys.exit(1)

    if isinstance(stub_or_function, Stub):
        # infer function or display help for how to select one
        stub = stub_or_function
        function_handle = infer_function_or_help(stub, module, accept_local_entrypoint, accept_webhook)
        return function_handle
    elif isinstance(stub_or_function, Function):
        return stub_or_function
    elif isinstance(stub_or_function, LocalEntrypoint):
        if not accept_local_entrypoint:
            raise click.UsageError(
                f"{func_ref} is not a Modal Function (a Modal local_entrypoint can't be used in this context)"
            )
        return stub_or_function
    else:
        raise click.UsageError(f"{stub_or_function} is not a Modal entity (should be a Stub or Function)")
