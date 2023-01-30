# Copyright Modal Labs 2022
import asyncio
import datetime
import inspect
import sys
from typing import Optional

import click
import typer
from rich.console import Console
from synchronicity import Interface

from modal.exception import InvalidError
from modal.functions import _FunctionHandle
from modal.stub import LocalEntrypoint
from modal_utils.async_utils import synchronizer

from .import_refs import (
    DEFAULT_STUB_NAME,
    NoSuchObject,
    get_by_object_path,
    import_file_or_module,
    import_function,
    import_stub,
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


class RunGroup(click.Group):
    def get_command(self, ctx, stub_ref):
        _function = import_function(stub_ref, interactive=False)
        _stub = _function._stub
        if isinstance(_function, LocalEntrypoint):
            click_command = _get_click_command_for_local_entrypoint(_stub, _function)
        else:
            if isinstance(_function, _FunctionHandle):
                _function = _function._function
            tag = _function._info.get_tag()
            click_command = _get_click_command_for_function(_stub, tag)

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
    import_ref = parse_import_ref(stub_ref)
    module = import_file_or_module(import_ref.file_or_module)
    try:
        object_path = import_ref.object_path or DEFAULT_STUB_NAME
        stub = get_by_object_path(module, object_path)
    except NoSuchObject:
        _show_no_auto_detectable_stub(import_ref)
        sys.exit(1)

    if name is None:
        name = stub.name

    res = stub.deploy(name=name)
    if inspect.iscoroutine(res):
        asyncio.run(res)


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
    _stub = import_stub(stub_ref)
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
    console = Console()
    if not console.is_terminal:
        raise click.UsageError("`modal shell` can only be run from a terminal.")

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
