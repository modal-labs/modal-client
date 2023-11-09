# Copyright Modal Labs 2022
import asyncio
import datetime
import functools
import inspect
import sys
import time
from typing import Any, Callable, Dict, Optional

import click
import typer
from rich.console import Console

from ..config import config
from ..environments import ensure_env
from ..exception import InvalidError
from ..functions import Function
from ..image import Image
from ..runner import deploy_stub, interactive_shell, run_stub
from ..serving import serve_stub
from ..stub import LocalEntrypoint, Stub
from .import_refs import import_function, import_stub
from .utils import ENV_OPTION, ENV_OPTION_HELP


class AnyParamType(click.ParamType):
    name = "any"

    def convert(self, value, param, ctx):
        return value


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
    Any: AnyParamType(),
}


class NoParserAvailable(InvalidError):
    pass


def _get_signature(f: Callable, is_method: bool = False) -> Dict[str, inspect.Parameter]:
    if is_method:
        self = None  # Dummy, doesn't matter
        f = functools.partial(f, self)
    return {param.name: param for param in inspect.signature(f).parameters.values()}


def _add_click_options(func, signature: Dict[str, inspect.Parameter]):
    """Adds @click.option based on function signature

    Kind of like typer, but using options instead of positional arguments
    """
    for param in signature.values():
        param_type = Any if param.annotation is inspect.Signature.empty else param.annotation
        param_name = param.name.replace("_", "-")
        cli_name = "--" + param_name
        if param_type in (bool, "bool"):
            cli_name += "/--no-" + param_name
        parser = option_parsers.get(param_type)
        if parser is None:
            raise NoParserAvailable(repr(param_type))
        kwargs: Any = {
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


def _get_click_command_for_function(stub: Stub, function_tag):
    function = stub[function_tag]

    if function.is_generator:
        raise InvalidError("`modal run` is not supported for generator functions")

    signature: Dict[str, inspect.Parameter]
    if function.info.cls is not None:
        cls_signature = _get_signature(function.info.cls)
        fun_signature = _get_signature(function.info.raw_f, is_method=True)
        signature = dict(**cls_signature, **fun_signature)  # Pool all arguments
        # TODO(erikbern): assert there's no overlap?
    else:
        signature = _get_signature(function.info.raw_f)

    @click.pass_context
    def f(ctx, **kwargs):
        with run_stub(
            stub,
            detach=ctx.obj["detach"],
            show_progress=ctx.obj["show_progress"],
            environment_name=ctx.obj["env"],
        ):
            if function.info.cls is None:
                function.remote(**kwargs)
            else:
                # unpool class and method arguments
                # TODO(erikbern): this code is a bit hacky
                cls_kwargs = {k: kwargs[k] for k in cls_signature}
                fun_kwargs = {k: kwargs[k] for k in fun_signature}
                method = function.from_parametrized(None, tuple(), cls_kwargs)
                method.remote(**fun_kwargs)

    with_click_options = _add_click_options(f, signature)
    return click.command(with_click_options)


def _get_click_command_for_local_entrypoint(stub: Stub, entrypoint: LocalEntrypoint):
    func = entrypoint.info.raw_f
    isasync = inspect.iscoroutinefunction(func)

    @click.pass_context
    def f(ctx, *args, **kwargs):
        if ctx.obj["detach"]:
            print(
                "Note that running a local entrypoint in detached mode only keeps the last triggered Modal function alive after the parent process has been killed or disconnected."
            )

        with run_stub(
            stub,
            detach=ctx.obj["detach"],
            show_progress=ctx.obj["show_progress"],
            environment_name=ctx.obj["env"],
        ):
            if isasync:
                asyncio.run(func(*args, **kwargs))
            else:
                func(*args, **kwargs)

    with_click_options = _add_click_options(f, _get_signature(func))
    return click.command(with_click_options)


class RunGroup(click.Group):
    def get_command(self, ctx, func_ref):
        function_or_entrypoint = import_function(func_ref, accept_local_entrypoint=True, base_cmd="modal run")
        stub: Stub = function_or_entrypoint.stub
        if stub.description is None:
            stub.set_description(_get_clean_stub_description(func_ref))
        if isinstance(function_or_entrypoint, LocalEntrypoint):
            click_command = _get_click_command_for_local_entrypoint(stub, function_or_entrypoint)
        else:
            tag = function_or_entrypoint.info.get_tag()
            click_command = _get_click_command_for_function(stub, tag)

        return click_command


@click.group(
    cls=RunGroup,
    subcommand_metavar="FUNC_REF",
)
@click.option("-q", "--quiet", is_flag=True, help="Don't show Modal progress indicators.")
@click.option("-d", "--detach", is_flag=True, help="Don't stop the app if the local process dies or disconnects.")
@click.option("-e", "--env", help=ENV_OPTION_HELP, default=None)
@click.pass_context
def run(ctx, detach, quiet, env):
    """Run a Modal function or local entrypoint.

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
    # this ensures that `modal.lookup()` without environment specification uses the same env as specified
    env = ensure_env(env)

    stub = import_stub(stub_ref)

    if name is None:
        name = stub.name

    deploy_stub(stub, name=name, environment_name=env)


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

    stub = import_stub(stub_ref)
    if stub.description is None:
        stub.set_description(_get_clean_stub_description(stub_ref))

    with serve_stub(stub, stub_ref, environment_name=env):
        if timeout is None:
            timeout = config["serve_timeout"]
        if timeout is None:
            timeout = float("inf")
        while timeout > 0:
            t = min(timeout, 3600)
            time.sleep(t)
            timeout -= t


def shell(
    func_ref: Optional[str] = typer.Argument(
        default=None,
        help="Path to a Python file with a Stub or Modal function whose container to run.",
        metavar="FUNC_REF",
    ),
    cmd: str = typer.Option(default="/bin/bash", help="Command to run inside the Modal image."),
    env: str = ENV_OPTION,
    image: Optional[str] = typer.Option(
        default=None, help="Container image tag for inside the shell (if not using FUNC_REF)."
    ),
    add_python: Optional[str] = typer.Option(default=None, help="Add Python to the image (if not using FUNC_REF)."),
    cpu: Optional[int] = typer.Option(
        default=None, help="Number of CPUs to allocate to the shell (if not using FUNC_REF)."
    ),
    memory: Optional[int] = typer.Option(
        default=None, help="Memory to allocate for the shell, in MiB (if not using FUNC_REF)."
    ),
    gpu: Optional[str] = typer.Option(
        default=None,
        help="GPUs to request for the shell, if any. Examples are `any`, `a10g`, `a100:4` (if not using FUNC_REF).",
    ),
):
    """Run an interactive shell inside a Modal image.

    **Examples:**

    Start a shell inside the default Debian-based image:

    ```bash
    modal shell
    ```

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

    if func_ref is not None:
        function = import_function(func_ref, accept_local_entrypoint=False, accept_webhook=True, base_cmd="modal shell")
    else:
        image_obj = Image.from_registry(image, add_python=add_python) if image else None
        stub = Stub("modal shell", image=image_obj)
        function = stub.function(
            serialized=True,
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            timeout=3600,
        )(lambda: None)

    assert isinstance(function, Function)  # ensured by accept_local_entrypoint=False

    interactive_shell(
        function,
        cmd,
        environment_name=env,
    )
