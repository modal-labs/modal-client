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
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

import modal
from modal.functions import _Function, _FunctionHandle
from modal.stub import AioStub, LocalEntrypoint, Stub, _Stub
from modal_utils.async_utils import synchronizer


@dataclasses.dataclass
class ImportRef:
    file_or_module: str
    object_path: Optional[str]


def parse_import_ref(object_ref: str) -> ImportRef:
    if object_ref.find("::") > 1:
        file_or_module, object_path = object_ref.split("::", 1)
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
    if ".py" in file_or_module:
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
            if isinstance(obj, (Stub, AioStub)):
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


def make_function_panel(idx: int, tag: str, function: _Function, stub: _Stub) -> Panel:
    items = [f"- {i}" for i in function.get_panel_items()]
    return Panel(
        Markdown("\n".join(items)),
        title=f"[bright_magenta]{idx}. [/bright_magenta][bold]{tag}[/bold]",
        title_align="left",
    )


def choose_function_interactive(stub: _Stub, console: Console) -> str:
    # TODO: allow selection of local_entrypoints when used from `modal run`
    functions = list(stub.registered_functions.items())
    function_panels = [make_function_panel(idx, tag, obj, stub) for idx, (tag, obj) in enumerate(functions)]

    renderable = Panel(Group(*function_panels))
    console.print(renderable)

    choice = Prompt.ask(
        "[yellow] Pick a function definition: [/yellow]",
        choices=[str(i) for i in range(len(functions))],
        default="0",
        show_default=False,
    )

    return functions[int(choice)][0]


def infer_function_or_help(_stub: _Stub, interactive: bool):
    function_choices = list(set(_stub.registered_functions.keys()) | set(_stub.registered_entrypoints.keys()))
    registered_functions_str = "\n".join(sorted(function_choices))
    if len(_stub.registered_entrypoints) == 1:
        # if there is a single local_entrypoint, use that regardless of
        # other functions on the stub
        function_name = list(_stub.registered_entrypoints.keys())[0]
        print(f"Using local_entrypoint {function_name}")
    elif len(function_choices) == 1:
        function_name = function_choices[0]
        print(f"Using function {function_name}")
    elif interactive:
        console = Console()
        function_name = choose_function_interactive(_stub, console)
    else:
        help_text = f"""You need to specify a Modal function or local entrypoint to run, e.g.

modal run app.py::my_function [...args]

Registered functions and local entrypoints on the selected stub are:
{registered_functions_str}
"""
        raise click.UsageError(help_text)

    if function_name in _stub.registered_entrypoints:
        # entrypoint is in entrypoint registry, for now
        return _stub.registered_entrypoints[function_name]

    return _stub[function_name]  # functions are in blueprint


def _show_no_auto_detectable_function_help(stub_ref: ImportRef) -> None:
    object_path = stub_ref.object_path
    import_path = stub_ref.file_or_module
    error_console = Console(stderr=True)
    error_console.print(f"[bold red]Could not find Modal stub or function '{object_path}' in {import_path}.[/bold red]")
    guidance_msg = (
        f"Try specifiy"
        f"For example a stub variable `app_stub = modal.Stub()` in `{import_path}` would "
        f"be specified as `{import_path}::app_stub`."
    )
    md = Markdown(guidance_msg)
    error_console.print(md)


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


def import_stub(stub_ref: str) -> _Stub:
    import_ref = parse_import_ref(stub_ref)
    try:
        module = import_file_or_module(import_ref.file_or_module)
        obj_path = import_ref.object_path or DEFAULT_STUB_NAME  # get variable named "stub" by default
        raw_object = get_by_object_path(module, obj_path)
    except NoSuchObject:
        _show_no_auto_detectable_stub(import_ref)
        sys.exit(1)

    try:
        _stub = synchronizer._translate_in(raw_object)
    except:
        raise click.UsageError(f"{raw_object} is not a Modal Stub")

    if not isinstance(_stub, _Stub):
        raise click.UsageError(f"{raw_object} is not a Modal Stub")

    return _stub


def import_function(
    func_ref: str, accept_local_entrypoint=True, interactive=False
) -> Union[_Function, LocalEntrypoint]:
    import_ref = parse_import_ref(func_ref)
    try:
        module = import_file_or_module(import_ref.file_or_module)
        obj_path = import_ref.object_path or DEFAULT_STUB_NAME  # get variable named "stub" by default
        raw_object = get_by_object_path(module, obj_path)
    except NoSuchObject:
        _show_no_auto_detectable_function_help(import_ref)
        sys.exit(1)

    try:
        stub_or_function = synchronizer._translate_in(raw_object)
    except:
        raise click.UsageError(f"{raw_object} is not a Modal entity (should be a Stub or Function)")

    if isinstance(stub_or_function, _Stub):
        # infer function or display help for how to select one
        _stub = stub_or_function
        _function = infer_function_or_help(_stub, interactive)
        return _function
    if isinstance(stub_or_function, _FunctionHandle):
        return stub_or_function._function
    elif isinstance(stub_or_function, (_Function, LocalEntrypoint)):
        return stub_or_function
    else:
        raise click.UsageError(f"{raw_object} is not a Modal entity (should be a Stub or Function)")
