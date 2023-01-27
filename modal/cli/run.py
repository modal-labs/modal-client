# Copyright Modal Labs 2022
import asyncio
import datetime
import inspect
import sys
import traceback
from typing import Optional, Union

import click
import typer
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from synchronicity import Interface

from modal.exception import InvalidError
from modal.functions import _Function, _FunctionHandle
from modal.stub import LocalEntrypoint, _Stub
from modal_utils.async_utils import synchronizer

from .import_refs import (
    DEFAULT_STUB_NAME,
    ImportRef,
    NoSuchObject,
    get_by_object_path,
    import_object,
    parse_import_ref,
)

run_cli = typer.Typer(name="run")


# Why do we need to support both types and the strings? Because something weird with
# how __annotations__ works in Python (which inspect.signature uses). See #220.
option_parsers = {
    str: str,
    "str": str,
    int: int,
    "int": int,
    float: float,
    "float": float,
    bool: bool,
    "bool": bool,
    datetime.datetime: click.DateTime(),
    "datetime.datetime": click.DateTime(),
}


class NoParserAvailable(InvalidError):
    pass


def _add_click_options(func, signature: inspect.Signature):
    """Adds @click.option based on function signature

    Kind of like typer, but using options instead of positional arguments
    """
    for param in signature.parameters.values():
        param_type = str if param.annotation is inspect.Signature.empty else param.annotation
        cli_name = "--" + param.name.replace("_", "-")
        parser = option_parsers.get(param_type)
        if parser is None:
            raise NoParserAvailable(repr(param_type))
        kwargs = {
            "type": parser,
        }
        if param.default is not inspect.Signature.empty:
            kwargs["default"] = param.default
        else:
            kwargs["required"] = True

        click.option(cli_name, **kwargs)(func)
    return func


def _get_click_command_for_function(_stub, function_tag):
    blocking_stub = synchronizer._translate_out(_stub, Interface.BLOCKING)

    _function = _stub[function_tag]
    raw_func = _function._info.raw_f

    @click.pass_context
    def f(ctx, *args, **kwargs):
        with blocking_stub.run(detach=ctx.obj["detach"]) as app:
            _function_handle = app[function_tag]
            _function_handle.call(*args, **kwargs)

    # TODO: handle `self` when raw_func is an unbound method (e.g. method on lifecycle class)
    with_click_options = _add_click_options(f, inspect.signature(raw_func))
    return click.command(with_click_options)


def _get_click_command_for_local_entrypoint(_stub, entrypoint: LocalEntrypoint):
    blocking_stub = synchronizer._translate_out(_stub, Interface.BLOCKING)
    func = entrypoint.raw_f
    isasync = inspect.iscoroutinefunction(func)

    @click.pass_context
    def f(ctx, *args, **kwargs):
        if ctx.obj["detach"]:
            print(
                "Note that running a local entrypoint in detached mode only keeps the last triggered Modal function alive after the parent process has been killed or disconnected."
            )

        with blocking_stub.run(detach=ctx.obj["detach"]):
            if isasync:
                asyncio.run(func(*args, **kwargs))
            else:
                func(*args, **kwargs)

    with_click_options = _add_click_options(f, inspect.signature(func))
    return click.command(with_click_options)


def get_main_function(import_ref_str: str, interactive: bool) -> Union[_Function, LocalEntrypoint]:
    import_ref = parse_import_ref(import_ref_str)
    try:
        module = import_object(import_ref)
        obj_path = import_ref.object_path or DEFAULT_STUB_NAME  # get variable named "stub" by default
        raw_object = get_by_object_path(module, obj_path)
    except NoSuchObject:
        _show_no_auto_detectable_function_help(import_ref)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
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


class RunGroup(click.Group):
    def get_command(self, ctx, stub_ref):
        _function = get_main_function(stub_ref, interactive=False)
        _stub = _function._stub
        if isinstance(_function, LocalEntrypoint):
            click_command = _get_click_command_for_local_entrypoint(_stub, _function)
        else:
            if isinstance(_function, _FunctionHandle):
                _function = _function._function
            tag = _function._info.get_tag()
            click_command = _get_click_command_for_function(_stub, tag)

        return click_command


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


@click.group(
    cls=RunGroup,
    subcommand_metavar="STUB_REF",
    help="""Run a Modal function or local entrypoint

STUB_REF should be of the format:

`{file or module}[::[{stub name}].{function name}]`

Examples:
To run the hello_world function (or local entrypoint) of stub `stub` in my_app.py:

 > modal run my_app.py::stub.hello_world

If your module only has a single stub called `stub` and your stub has a single local entrypoint (or single function), you can omit the stub/function part:


 > modal run my_project.my_app

""",
)
@click.option("--detach", is_flag=True, help="Don't stop the app if the local process dies or disconnects.")
@click.pass_context
def run(ctx, detach):
    ctx.ensure_object(dict)
    ctx.obj["detach"] = detach  # if subcommand would be a click command...


def deploy(
    stub_ref: str = typer.Argument(..., help="Path to a Python file with a stub."),
    name: str = typer.Option(None, help="Name of the deployment."),
):
    import_ref = parse_import_ref(stub_ref)
    try:
        stub = import_object(import_ref)
    except NoSuchObject:
        _show_no_auto_detectable_function_help(import_ref)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    if name is None:
        name = stub.name

    res = stub.deploy(name=name)
    if inspect.iscoroutine(res):
        asyncio.run(res)


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


def serve(
    stub_ref: str = typer.Argument(..., help="Path to a Python file with a stub."),
    timeout: Optional[float] = None,
):
    """Run an web endpoint(s) associated with a Modal stub and hot-reload code.
    **Examples:**\n
    \n
    ```bash\n
    modal serve hello_world.py
    ```\n
    """
    import_ref = parse_import_ref(stub_ref)
    try:
        stub = import_object(import_ref)
    except NoSuchObject:
        _show_no_auto_detectable_function_help(import_ref)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    _stub = synchronizer._translate_in(stub)
    blocking_stub = synchronizer._translate_out(_stub, Interface.BLOCKING)
    blocking_stub.serve(timeout=timeout)


def shell(
    stub_ref: str = typer.Argument(..., help="Path to a Python file with a stub."),
    cmd: str = typer.Option(default="/bin/bash", help="Command to run inside the Modal image."),
):
    """Run an interactive shell inside a Modal image.\n
    **Examples:**\n
    \n
    - Start a bash shell using the spec for `my_function` in your stub:\n
    ```bash\n
    modal shell hello_world.py::stub.my_function \n
    ```\n
    Note that you can select the function interactively if you omit the function name.\n
    \n
    - Start a `python` shell: \n
    ```bash\n
    modal shell hello_world.py --cmd=python \n
    ```\n
    """
    import_ref = parse_import_ref(stub_ref)
    try:
        stub_or_func = import_object(import_ref)
    except NoSuchObject:
        _show_no_auto_detectable_function_help(import_ref)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    console = Console()

    if not console.is_terminal:
        print("`modal shell` can only be run from a terminal.")
        sys.exit(1)

    if function is None:
        res = stub.interactive_shell(cmd)
    else:
        res = stub.interactive_shell(
            cmd,
            mounts=function._mounts,
            shared_volumes=function._shared_volumes,
            image=function._image,
            secrets=function._secrets,
            gpu=function._gpu,
        )

    if inspect.iscoroutine(res):
        asyncio.run(res)


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
