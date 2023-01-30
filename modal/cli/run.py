# Copyright Modal Labs 2022
import asyncio
import datetime
import inspect
import sys
import traceback
from typing import List, Optional, Tuple

import click
import typer
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from synchronicity import Interface

from modal.cli.app import _show_stub_ref_failure_help
from modal.exception import InvalidError
from modal.functions import _Function
from modal.stub import _Stub
from modal_utils.async_utils import synchronizer
from modal_utils.package_utils import NoSuchStub, import_stub, parse_stub_ref

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


def _get_click_command_for_function_handle(_stub, function_tag: str):
    blocking_stub = synchronizer._translate_out(_stub, Interface.BLOCKING)
    raw_func = _stub._blueprint[function_tag]._info.raw_f

    @click.pass_context
    def f(ctx, *args, **kwargs):
        with blocking_stub.run(detach=ctx.obj["detach"]) as app:
            function_handle = getattr(app, function_tag)
            function_handle.call(*args, **kwargs)

    with_click_options = _add_click_options(f, inspect.signature(raw_func))
    return click.command(with_click_options)


def _get_click_command_for_local_entrypoint(_stub, entrypoint_name: str):
    blocking_stub = synchronizer._translate_out(_stub, Interface.BLOCKING)
    func = _stub._local_entrypoints[entrypoint_name]
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


class RunGroup(click.Group):
    def get_command(self, ctx, stub_ref):
        parsed_stub_ref = parse_stub_ref(stub_ref)
        try:
            stub = import_stub(parsed_stub_ref)
        except NoSuchStub:
            _show_stub_ref_failure_help(parsed_stub_ref)
            sys.exit(1)
        except Exception:
            traceback.print_exc()
            sys.exit(1)

        _stub = synchronizer._translate_in(stub)

        function_choices = list(
            (set(_stub.registered_functions) - set(_stub.registered_web_endpoints))
            | set(_stub.registered_entrypoints.keys())
        )
        registered_functions_str = "\n".join(sorted(function_choices))
        function_name = parsed_stub_ref.entrypoint_name
        if not function_name:
            if len(function_choices) == 1:
                function_name = function_choices[0]
            elif len(_stub.registered_entrypoints) == 1:
                function_name = list(_stub.registered_entrypoints.keys())[0]
            else:
                # TODO(erikbern): better error message if there's *zero* functions / entrypoints
                print(
                    f"""You need to specify an entrypoint Modal function to run, e.g.

modal run app.py::stub.my_function [...args]

Registered functions and local entrypoints on the selected stub are:
{registered_functions_str}
    """
                )
                exit(1)
        elif function_name not in function_choices:
            print(
                f"""No function `{function_name}` could be found in the specified stub. Registered functions and entrypoints are:

{registered_functions_str}"""
            )
            exit(1)

        if function_name in _stub.registered_functions:
            click_command = _get_click_command_for_function_handle(_stub, function_name)
        else:
            click_command = _get_click_command_for_local_entrypoint(_stub, function_name)

        return click_command


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
    parsed_stub_ref = parse_stub_ref(stub_ref)
    try:
        stub = import_stub(parsed_stub_ref)
    except NoSuchStub:
        _show_stub_ref_failure_help(parsed_stub_ref)
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


def choose_function(stub: _Stub, functions: List[Tuple[str, _Function]], console: Console):
    if len(functions) == 0:
        return None
    elif len(functions) == 1:
        return functions[0][1]

    function_panels = [make_function_panel(idx, tag, obj, stub) for idx, (tag, obj) in enumerate(functions)]

    renderable = Panel(Group(*function_panels))
    console.print(renderable)

    choice = Prompt.ask(
        "[yellow] Pick a function definition to create a corresponding shell: [/yellow]",
        choices=[str(i) for i in range(len(functions))],
        default="0",
        show_default=False,
    )

    return functions[int(choice)][1]


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
    parsed_stub_ref = parse_stub_ref(stub_ref)
    try:
        stub = import_stub(parsed_stub_ref)
    except NoSuchStub:
        _show_stub_ref_failure_help(parsed_stub_ref)
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
    parsed_stub_ref = parse_stub_ref(stub_ref)
    try:
        stub = import_stub(parsed_stub_ref)
    except NoSuchStub:
        _show_stub_ref_failure_help(parsed_stub_ref)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    console = Console()

    if not console.is_terminal:
        print("`modal shell` can only be run from a terminal.")
        sys.exit(1)

    _stub = synchronizer._translate_in(stub)
    functions = {tag: obj for tag, obj in _stub._blueprint.items() if isinstance(obj, _Function)}
    function_name = parsed_stub_ref.entrypoint_name
    if function_name is not None:
        if function_name not in functions:
            print(f"Function {function_name} not found in stub.")
            sys.exit(1)
        function = functions[function_name]
    else:
        function = choose_function(_stub, list(functions.items()), console)

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
