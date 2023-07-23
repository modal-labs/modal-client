# Copyright Modal Labs 2022
import asyncio
import datetime
import inspect
import sys
import time
import warnings
from typing import Optional

import click
import typer
from rich.console import Console
from synchronicity import Interface

from modal.config import config
from modal.exception import InvalidError
from modal.runner import deploy_stub, interactive_shell, run_stub
from modal.serving import serve_stub
from modal.stub import LocalEntrypoint
from modal_utils.async_utils import synchronizer

from ..environments import ensure_env
from ..functions import _FunctionHandle
from .import_refs import import_function, import_stub
from .utils import ENV_OPTION, ENV_OPTION_HELP

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
        param_name = param.name.replace("_", "-")
        cli_name = "--" + param_name
        if param_type in (bool, "bool"):
            cli_name += "/--no-" + param_name
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


def _get_clean_stub_description(func_ref: str) -> str:
    # If possible, consider the 'ref' argument the start of the app's args. Everything
    # before it Modal CLI cruft (eg. `modal run --detach`).
    try:
        func_ref_arg_idx = sys.argv.index(func_ref)
        return " ".join(sys.argv[func_ref_arg_idx:])
    except ValueError:
        return " ".join(sys.argv)


def _get_click_command_for_function(_stub, function_tag):
    blocking_stub = synchronizer._translate_out(_stub, Interface.BLOCKING)

    _function = _stub[function_tag]
    raw_func = _function._info.raw_f

    @click.pass_context
    def f(ctx, *args, **kwargs):
        with run_stub(
            blocking_stub,
            detach=ctx.obj["detach"],
            show_progress=ctx.obj["show_progress"],
            environment_name=ctx.obj["env"],
        ) as app:
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

        with run_stub(
            blocking_stub,
            detach=ctx.obj["detach"],
            show_progress=ctx.obj["show_progress"],
            environment_name=ctx.obj["env"],
        ) as app:
            if isasync:
                asyncio.run(func(*args, **kwargs))
            else:
                func(*args, **kwargs)
            if app.client.function_invocations == 0:
                # TODO: better formatting for the warning message
                warnings.warn(
                    "Warning: no remote function calls were made.\n"
                    "Note that Modal functions run locally when called directly (e.g. `f()`).\n"
                    "In order to run a function remotely, you may use `f.call()`. (See https://modal.com/docs/reference/modal.Function for other options)."
                )

    with_click_options = _add_click_options(f, inspect.signature(func))
    return click.command(with_click_options)


class RunGroup(click.Group):
    def get_command(self, ctx, func_ref):
        _function_handle_or_entrypoint = import_function(
            func_ref, accept_local_entrypoint=True, interactive=False, base_cmd="modal run"
        )
        _stub = _function_handle_or_entrypoint._stub
        if _stub._description is None:
            _stub._description = _get_clean_stub_description(func_ref)
        if isinstance(_function_handle_or_entrypoint, LocalEntrypoint):
            click_command = _get_click_command_for_local_entrypoint(_stub, _function_handle_or_entrypoint)
        else:
            tag = _function_handle_or_entrypoint._info.get_tag()
            click_command = _get_click_command_for_function(_stub, tag)

        return click_command


@click.group(
    cls=RunGroup,
    subcommand_metavar="FUNC_REF",
)
@click.option("-q", "--quiet", is_flag=True, help="Don't show Modal progress indicators.")
@click.option("-d", "--detach", is_flag=True, help="Don't stop the app if the local process dies or disconnects.")
@click.option("-e", "--env", help=ENV_OPTION_HELP, default=None, hidden=True)
@click.pass_context
def run(ctx, detach, quiet, env):
    """Run a Modal function or local entrypoint

    `FUNC_REF` should be of the format `{file or module}::{function name}`.
    Alternatively, you can refer to the function via the stub:

    `{file or module}::{stub variable name}.{function name}`

    **Examples:**

    To run the hello_world function (or local entrypoint) in my_app.py:

    ```bash
    modal run my_app.py::hello_world
    ```

    If your module only has a single stub called `stub` and your stub has a
    single local entrypoint (or single function), you can omit the stub and
    function parts:

    ```bash
    modal run my_app.py
    ```

    Instead of pointing to a file, you can also use the Python module path:

    ```bash
    modal run my_project.my_app
    ```
    """
    ctx.ensure_object(dict)
    ctx.obj["detach"] = detach  # if subcommand would be a click command...
    ctx.obj["show_progress"] = False if quiet else True
    ctx.obj["env"] = env


def deploy(
    stub_ref: str = typer.Argument(..., help="Path to a Python file with a stub."),
    name: str = typer.Option(None, help="Name of the deployment."),
    env: str = ENV_OPTION,
):
    env = ensure_env(
        env
    )  # this ensures that `modal.lookup()` without environment specification uses the same env as specified

    _stub = import_stub(stub_ref)

    if name is None:
        name = _stub.name

    blocking_stub = synchronizer._translate_out(_stub, interface=Interface.BLOCKING)
    deploy_stub(blocking_stub, name=name, environment_name=env)


def serve(
    stub_ref: str = typer.Argument(..., help="Path to a Python file with a stub."),
    timeout: Optional[float] = None,
    env: str = ENV_OPTION,
):
    """Run a web endpoint(s) associated with a Modal stub and hot-reload code.

    **Examples:**

    ```bash
    modal serve hello_world.py
    ```
    """
    env = ensure_env(env)

    with serve_stub(stub_ref, environment_name=env):
        if timeout is None:
            timeout = config["serve_timeout"]
        if timeout is None:
            timeout = float("inf")
        while timeout > 0:
            t = min(timeout, 3600)
            time.sleep(t)
            timeout -= t


def shell(
    func_ref: str = typer.Argument(
        ..., help="Path to a Python file with a Stub or Modal function whose container to run.", metavar="FUNC_REF"
    ),
    cmd: str = typer.Option(default="/bin/bash", help="Command to run inside the Modal image."),
    env: str = ENV_OPTION,
):
    """Run an interactive shell inside a Modal image.

    **Examples:**

    Start a bash shell using the spec for `my_function` in your stub:

    ```bash
    modal shell hello_world.py::my_function
    ```

    Note that you can select the function interactively if you omit the function name.

    Start a `python` shell:

    ```bash
    modal shell hello_world.py --cmd=python
    ```
    """
    env = ensure_env(env)

    console = Console()
    if not console.is_terminal:
        raise click.UsageError("`modal shell` can only be run from a terminal.")

    _function_handle = import_function(
        func_ref, accept_local_entrypoint=False, accept_webhook=True, interactive=True, base_cmd="modal shell"
    )
    assert isinstance(_function_handle, _FunctionHandle)  # ensured by accept_local_entrypoint=False
    _stub = _function_handle._stub
    _function = _function_handle._get_function()
    blocking_stub = synchronizer._translate_out(_stub, Interface.BLOCKING)
    blocking_function = synchronizer._translate_out(_function, Interface.BLOCKING)

    interactive_shell(
        blocking_stub,
        cmd,
        blocking_function,
        environment_name=env,
    )
