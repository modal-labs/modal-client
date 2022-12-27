# Copyright Modal Labs 2022
import asyncio
import functools
import inspect
import sys
import traceback
from typing import List, Tuple

import typer
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from synchronicity import Interface

from modal.cli.app import _show_stub_ref_failure_help
from modal.functions import _Function
from modal.stub import _Stub
from modal_utils.async_utils import synchronizer
from modal_utils.package_utils import NoSuchStub, import_stub, parse_stub_ref

run_cli = typer.Typer(name="run")


def _get_run_wrapper_function_handle(_stub, function_tag: str, detach: bool):
    blocking_stub = synchronizer._translate_out(_stub, Interface.BLOCKING)

    @functools.wraps(_stub._blueprint[function_tag]._info.raw_f)
    def f(*args, **kwargs):
        with blocking_stub.run(detach=detach) as app:
            function_handle = getattr(app, function_tag)
            function_handle.call(*args, **kwargs)

    return f


def _get_run_wrapper_local_entrypoint(_stub, entrypoint_name: str, detach: bool):
    stub = synchronizer._translate_out(_stub, Interface.BLOCKING)
    func = _stub._local_entrypoints[entrypoint_name]

    isasync = inspect.iscoroutinefunction(func)

    @functools.wraps(func)
    def f(*args, **kwargs):
        with stub.run(detach=detach):
            if isasync:
                asyncio.run(func(*args, **kwargs))
            else:
                func(*args, **kwargs)

    return f


def run(
    ctx: typer.Context,
    stub_ref: str = typer.Argument(
        ..., help="Path to a Python file or module, optionally identifying the name of your stub: `./main.py:mystub`."
    ),
    detach: bool = typer.Option(default=False, help="Allows app to continue running if local terminal disconnects."),
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

    _stub = synchronizer._translate_in(stub)
    function_choices = list(set(_stub.registered_functions) | set(_stub.registered_entrypoints.keys()))
    registered_functions_str = "\n".join(function_choices)
    function_name = parsed_stub_ref.entrypoint_name
    if not function_name:
        if len(function_choices) == 1:
            function_name = function_choices[0]
        elif len(_stub.registered_entrypoints) == 1:
            function_name = list(_stub.registered_entrypoints.keys())[0]
        else:
            print(
                f"""You need to specify an entrypoint Modal function to run, e.g. `modal run app.py my_function [...args]`.
Registered functions and entrypoints on the selected stub are:
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

    func_typer = typer.Typer()
    if function_name in _stub.registered_functions:
        func_typer.command(name=function_name)(_get_run_wrapper_function_handle(_stub, function_name, detach))
    else:
        func_typer.command(name=function_name)(_get_run_wrapper_local_entrypoint(_stub, function_name, detach))

    # TODO: propagate help to sub-invocation if enough arguments are available
    func_typer(args=ctx.args)


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
    items = [
        f"- {i}"
        for i in [*function._mounts, function._image, *function._secrets, *function._shared_volumes.values()]
        if i not in [stub._client_mount, *stub._function_mounts.values()]
    ]
    if function._gpu:
        items.append("- GPU")
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
