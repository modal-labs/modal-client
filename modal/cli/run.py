# Copyright Modal Labs 2022
import asyncio
import datetime
import inspect
from typing import Optional

import click
import typer
from rich.console import Console
from synchronicity import Interface

from modal._live_reload import run_serve_loop
from modal.exception import InvalidError
from modal.stub import LocalEntrypoint
from modal_utils.async_utils import synchronizer

from .import_refs import import_function, import_stub
from ..functions import _FunctionHandle

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


class RunGroup(click.Group):
    def get_command(self, ctx, func_ref):
        _function_handle_or_entrypoint = import_function(
            func_ref, accept_local_entrypoint=True, interactive=False, base_cmd="modal run"
        )
        _stub = _function_handle_or_entrypoint._stub
        if isinstance(_function_handle_or_entrypoint, LocalEntrypoint):
            click_command = _get_click_command_for_local_entrypoint(_stub, _function_handle_or_entrypoint)
        else:
            tag = _function_handle_or_entrypoint._info.get_tag()
            click_command = _get_click_command_for_function(_stub, tag)

        return click_command


@click.group(
    cls=RunGroup,
    subcommand_metavar="FUNC_REF",
    help="""Run a Modal function or local entrypoint

FUNC_REF should be of the format:

`{file or module}::{function name}`

Alternatively you can refer to the function via the stub:

`{file or module}::{stub variable name}.{function name}`

Examples:
To run the hello_world function (or local entrypoint) in my_app.py:

 > modal run my_app.py::hello_world

If your module only has a single stub called `stub` and your stub has a single local entrypoint (or single function), you can omit the stub/function part:

 > modal run my_app.py

Instead of pointing to a file, you can also use the Python module path to a a file:

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
    _stub = import_stub(stub_ref)

    if name is None:
        name = _stub.name

    blocking_stub = synchronizer._translate_out(_stub, interface=Interface.BLOCKING)
    blocking_stub.deploy(name=name)


def serve(
    stub_ref: str = typer.Argument(..., help="Path to a Python file with a stub."),
    timeout: Optional[float] = None,
):
    """Run a web endpoint(s) associated with a Modal stub and hot-reload code.
    **Examples:**\n
    \n
    ```bash\n
    modal serve hello_world.py
    ```\n
    """
    run_serve_loop(stub_ref, timeout)


def shell(
    func_ref: str = typer.Argument(
        ..., help="Path to a Python file with a Stub or Modal function whose container to run.", metavar="FUNC_REF"
    ),
    cmd: str = typer.Option(default="/bin/bash", help="Command to run inside the Modal image."),
):
    """Run an interactive shell inside a Modal image.\n
    **Examples:**\n
    \n
    - Start a bash shell using the spec for `my_function` in your stub:\n
    ```bash\n
    modal shell hello_world.py::my_function \n
    ```\n
    Note that you can select the function interactively if you omit the function name.\n
    \n
    - Start a `python` shell: \n
    ```bash\n
    modal shell hello_world.py --cmd=python \n
    ```\n
    """
    console = Console()
    if not console.is_terminal:
        raise click.UsageError("`modal shell` can only be run from a terminal.")

    _function_handle = import_function(
        func_ref, accept_local_entrypoint=False, interactive=True, base_cmd="modal shell"
    )
    assert isinstance(_function_handle, _FunctionHandle)  # ensured by accept_local_entrypoint=False
    _stub = _function_handle._stub
    _function = _function_handle._get_function()
    blocking_stub = synchronizer._translate_out(_stub, Interface.BLOCKING)

    if _function_handle is None:
        blocking_stub.interactive_shell(cmd)
    else:
        blocking_stub.interactive_shell(
            cmd,
            mounts=_function._mounts,
            shared_volumes=_function._shared_volumes,
            image=_function._image,
            secrets=_function._secrets,
            gpu=_function._gpu,
            cloud=_function._cloud,
        )
