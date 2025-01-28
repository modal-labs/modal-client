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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, cast

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
    object_path: str


def parse_import_ref(object_ref: str) -> ImportRef:
    if object_ref.find("::") > 1:
        file_or_module, object_path = object_ref.split("::", 1)
    elif object_ref.find(":") > 1:
        raise InvalidError(f"Invalid object reference: {object_ref}. Did you mean '::' instead of ':'?")
    else:
        file_or_module, object_path = object_ref, ""

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


Runnable = Union[Function, MethodReference, LocalEntrypoint]


@dataclass(frozen=True)
class CLICommand:
    names: list[str]
    runnable: Runnable
    is_web_endpoint: bool


def list_cli_commands(
    module: types.ModuleType,
) -> list[CLICommand]:
    """
    Extracts all runnables found either directly in the input module, or in any of the Apps listed in that module

    Runnables includes all Functions, (class) Methods and Local Entrypoints, including web endpoints.

    The returned list consists of tuples:
    ([name1, name2...], Runnable)

    Where the first name is always the module level name if such a name exists
    """
    apps = cast(list[tuple[str, App]], inspect.getmembers(module, lambda x: isinstance(x, App)))

    all_runnables: dict[Runnable, list[str]] = defaultdict(list)
    for app_name, app in apps:
        for name, local_entrypoint in app.registered_entrypoints.items():
            all_runnables[local_entrypoint].append(f"{app_name}.{name}")
        for name, function in app.registered_functions.items():
            if name.endswith(".*"):
                continue
            all_runnables[function].append(f"{app_name}.{name}")
        for cls_name, cls in app.registered_classes.items():
            for method_name in cls._get_method_names():
                method_ref = MethodReference(cls, method_name)
                all_runnables[method_ref].append(f"{app_name}.{cls_name}.{method_name}")

    # If any class or function is exported as a module level object, use that
    # as the preferred name by putting it first in the list
    module_level_entities = cast(
        list[tuple[str, Runnable]],
        inspect.getmembers(module, lambda x: isinstance(x, (Function, Cls, LocalEntrypoint))),
    )
    for name, entity in module_level_entities:
        if isinstance(entity, Cls) and entity._is_local():
            for method_name in entity._get_method_names():
                method_ref = MethodReference(entity, method_name)
                all_runnables.setdefault(method_ref, []).insert(0, f"{name}.{method_name}")
        elif (isinstance(entity, Function) and entity._is_local()) or isinstance(entity, LocalEntrypoint):
            all_runnables.setdefault(entity, []).insert(0, name)

    def _is_web_endpoint(runnable: Runnable) -> bool:
        if isinstance(runnable, Function) and runnable._is_web_endpoint():
            return True
        elif isinstance(runnable, MethodReference):
            # this is a bit yucky but can hopefully get cleaned up with Cls cleanup:
            method_partial = runnable.cls._get_partial_functions()[runnable.method_name]
            if method_partial._is_web_endpoint():
                return True

        return False

    return [CLICommand(names, runnable, _is_web_endpoint(runnable)) for runnable, names in all_runnables.items()]


def filter_cli_commands(
    cli_commands: list[CLICommand],
    name_prefix: str,
    accept_local_entrypoints: bool = True,
    accept_web_endpoints: bool = True,
) -> list[CLICommand]:
    """Filters by name and type of runnable

    Returns generator of (matching names list, CLICommand)
    """

    def _is_accepted_type(cli_command: CLICommand) -> bool:
        if not accept_local_entrypoints and isinstance(cli_command.runnable, LocalEntrypoint):
            return False
        if not accept_web_endpoints and cli_command.is_web_endpoint:
            return False
        return True

    res = []
    for cli_command in cli_commands:
        if not _is_accepted_type(cli_command):
            continue

        if name_prefix in cli_command.names:
            # exact name match
            res.append(cli_command)
            continue

        if not name_prefix:
            # no name specified, return all reachable runnables
            res.append(cli_command)
            continue

        # partial matches e.g. app or class name - should we even allow this?
        prefix_matches = [x for x in cli_command.names if x.startswith(f"{name_prefix}.")]
        if prefix_matches:
            res.append(cli_command)
    return res


def import_app(app_ref: str) -> App:
    import_ref = parse_import_ref(app_ref)

    # TODO: default could be to just pick up any app regardless if it's called DEFAULT_APP_NAME
    #  as long as there is a single app in the module?
    import_path = import_ref.file_or_module
    object_path = import_ref.object_path or DEFAULT_APP_NAME

    module = import_file_or_module(import_ref.file_or_module)

    if "." in object_path:
        raise click.UsageError(f"{object_path} is not a Modal App")

    app = getattr(module, object_path)

    if app is None:
        error_console = Console(stderr=True)
        error_console.print(f"[bold red]Could not find Modal app '{object_path}' in {import_path}.[/bold red]")

        if not object_path:
            guidance_msg = Markdown(
                f"Expected to find an app variable named **`{DEFAULT_APP_NAME}`** (the default app name). "
                "If your `modal.App` is assigned to a different variable name, "
                "you must specify it in the app ref argument. "
                f"For example an App variable `app_2 = modal.App()` in `{import_path}` would "
                f"be specified as `{import_path}::app_2`."
            )
            error_console.print(guidance_msg)

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


def _get_runnable_app(runnable: Runnable) -> App:
    if isinstance(runnable, Function):
        return runnable.app
    elif isinstance(runnable, MethodReference):
        return runnable.cls._get_app()
    else:
        assert isinstance(runnable, LocalEntrypoint)
        return runnable.app


def import_and_filter(
    import_ref: ImportRef, accept_local_entrypoint=True, accept_webhook=False
) -> tuple[Optional[Runnable], list[CLICommand]]:
    """Takes a function ref string and returns a single determined "runnable" to use, and a list of all available
    runnables.

    The function ref can leave out partial information (apart from the file name/module)
    as long as the runnable is uniquely identifiable by the provided information.

    When there are multiple runnables within the provided ref, the following rules should
    be followed:

    1. if there is a single local_entrypoint, that one is used
    2. if there is a single {function, class} that one is used
    3. if there is a single method (within a class) that one is used
    """
    # all commands:
    module = import_file_or_module(import_ref.file_or_module)
    cli_commands = list_cli_commands(module)

    # all commands that satisfy local entrypoint/accept webhook limitations AND object path prefix
    filtered_commands = filter_cli_commands(
        cli_commands, import_ref.object_path, accept_local_entrypoint, accept_webhook
    )
    all_usable_commands = filter_cli_commands(cli_commands, "", accept_local_entrypoint, accept_webhook)

    if len(filtered_commands) == 1:
        cli_command = filtered_commands[0]
        return cli_command.runnable, all_usable_commands

    # we are here if there is more than one matching function
    if accept_local_entrypoint:
        local_entrypoint_cmds = [cmd for cmd in filtered_commands if isinstance(cmd.runnable, LocalEntrypoint)]
        if len(local_entrypoint_cmds) == 1:
            # if there is a single local entrypoint - use that
            return local_entrypoint_cmds[0].runnable, all_usable_commands

    return None, all_usable_commands
