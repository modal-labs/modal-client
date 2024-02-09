# Copyright Modal Labs 2022
import asyncio
import functools
import inspect
import re
import sys
import time
from functools import partial
from typing import Any, Callable, Dict, Optional, get_type_hints

import click
import typer
from rich.console import Console
from typing_extensions import TypedDict

from ..config import config
from ..environments import ensure_env
from ..exception import ExecutionError, InvalidError
from ..functions import Function, FunctionEnv
from ..image import Image
from ..runner import deploy_stub, interactive_shell, run_stub
from ..s3mount import _S3Mount
from ..serving import serve_stub
from ..stub import LocalEntrypoint, Stub
from .import_refs import import_function, import_stub
from .utils import ENV_OPTION, ENV_OPTION_HELP


class ParameterMetadata(TypedDict):
    name: str
    default: Any
    annotation: Any
    type_hint: Any


class AnyParamType(click.ParamType):
    name = "any"

    def convert(self, value, param, ctx):
        return value


option_parsers = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "datetime.datetime": click.DateTime(),
    "Any": AnyParamType(),
}


class NoParserAvailable(InvalidError):
    pass


def _get_signature(f: Callable, is_method: bool = False) -> Dict[str, ParameterMetadata]:
    try:
        type_hints = get_type_hints(f)
    except Exception as exc:
        # E.g., if entrypoint type hints cannot be evaluated by local Python runtime
        msg = "Unable to generate command line interface for app entrypoint. See traceback above for details."
        raise ExecutionError(msg) from exc

    if is_method:
        self = None  # Dummy, doesn't matter
        f = functools.partial(f, self)
    signature: Dict[str, ParameterMetadata] = {}
    for param in inspect.signature(f).parameters.values():
        signature[param.name] = {
            "name": param.name,
            "default": param.default,
            "annotation": param.annotation,
            "type_hint": type_hints.get(param.name, "Any"),
        }
    return signature


def _get_param_type_as_str(annot: Any) -> str:
    """Return annotation as a string, handling various spellings for optional types."""
    annot_str = str(annot)
    annot_patterns = [
        r"typing\.Optional\[([\w.]+)\]",
        r"typing\.Union\[([\w.]+), NoneType\]",
        r"([\w.]+) \| None",
        r"<class '([\w\.]+)'>",
    ]
    for pat in annot_patterns:
        m = re.match(pat, annot_str)
        if m is not None:
            return m.group(1)
    return annot_str


def _add_click_options(func, signature: Dict[str, ParameterMetadata]):
    """Adds @click.option based on function signature

    Kind of like typer, but using options instead of positional arguments
    """
    for param in signature.values():
        param_type_str = _get_param_type_as_str(param["type_hint"])
        param_name = param["name"].replace("_", "-")
        cli_name = "--" + param_name
        if param_type_str == "bool":
            cli_name += "/--no-" + param_name
        parser = option_parsers.get(param_type_str)
        if parser is None:
            msg = f"Parameter `{param_name}` has unparseable annotation: {param['annotation']!r}"
            raise NoParserAvailable(msg)
        kwargs: Any = {
            "type": parser,
        }
        if param["default"] is not inspect.Signature.empty:
            kwargs["default"] = param["default"]
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

    signature: Dict[str, ParameterMetadata]
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
                method = function.from_parametrized(None, False, None, tuple(), cls_kwargs)
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
        # note: get_command here is run before the "group logic" in the `run` logic below
        # so to ensure that `env` has been globally populated before user code is loaded, it
        # needs to be handled here, and not in the `run` logic below
        ctx.ensure_object(dict)
        ctx.obj["env"] = ensure_env(ctx.params["env"])
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


def deploy(
    stub_ref: str = typer.Argument(..., help="Path to a Python file with a stub."),
    name: str = typer.Option(None, help="Name of the deployment."),
    env: str = ENV_OPTION,
    public: bool = typer.Option(
        False, help="[beta] Publicize the deployment so other workspaces can lookup the function."
    ),
    skip_confirm: bool = typer.Option(False, help="Skip public app confirmation dialog."),
):
    # this ensures that `modal.lookup()` without environment specification uses the same env as specified
    env = ensure_env(env)

    stub = import_stub(stub_ref)

    if name is None:
        name = stub.name

    if public and not skip_confirm:
        if not click.confirm(
            "⚠️ Public apps are a beta feature. ⚠️\n"
            "Making an app public will allow any user (including from outside your workspace) to look up and use your functions.\n"
            "Are you sure you want your app to be public?"
        ):
            return

    deploy_stub(stub, name=name, environment_name=env, public=public)


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
    cloud: Optional[str] = typer.Option(
        default=None,
        help="Cloud provider to run the function on. Possible values are `aws`, `gcp`, `oci`, `auto` (if not using FUNC_REF).",
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

    stub = Stub("modal shell")

    if func_ref is not None:
        function = import_function(func_ref, accept_local_entrypoint=False, accept_webhook=True, base_cmd="modal shell")
        assert isinstance(function, Function)
        function_env: FunctionEnv = function.env
        s3_mounts = {k: v for k, v in function_env.volumes.items() if isinstance(v, _S3Mount)}
        start_shell = partial(
            interactive_shell,
            image=function_env.image,
            mounts=function_env.mounts,
            secrets=function_env.secrets,
            network_file_systems=function_env.network_file_systems,
            gpu=function_env.gpu,
            cloud=function_env.cloud,
            cpu=function_env.cpu,
            memory=function_env.memory,
            volumes=s3_mounts,  # currently, sandboxes only support s3 mounts
        )
    else:
        modal_image = Image.from_registry(image, add_python=add_python) if image else None
        start_shell = partial(interactive_shell, image=modal_image, cpu=cpu, memory=memory, gpu=gpu, cloud=cloud)

    start_shell(
        stub,
        cmd=[cmd],
        environment_name=env,
        timeout=3600,
    )
