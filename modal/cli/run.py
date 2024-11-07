# Copyright Modal Labs 2022
import asyncio
import functools
import inspect
import platform
import re
import shlex
import sys
import time
import typing
from functools import partial
from typing import Any, Callable, Dict, Optional, get_type_hints

import click
import typer
from rich.console import Console
from rich.panel import Panel
from typing_extensions import TypedDict

from .. import Cls
from ..app import App, LocalEntrypoint
from ..config import config
from ..environments import ensure_env
from ..exception import ExecutionError, InvalidError, _CliUserExecutionError
from ..functions import Function, _FunctionSpec
from ..image import Image
from ..output import enable_output
from ..runner import deploy_app, interactive_shell, run_app
from ..serving import serve_app
from ..volume import Volume
from .import_refs import import_app, import_function
from .utils import ENV_OPTION, ENV_OPTION_HELP, stream_app_logs


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


def _get_signature(f: Callable[..., Any], is_method: bool = False) -> Dict[str, ParameterMetadata]:
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


def _get_clean_app_description(func_ref: str) -> str:
    # If possible, consider the 'ref' argument the start of the app's args. Everything
    # before it Modal CLI cruft (eg. `modal run --detach`).
    try:
        func_ref_arg_idx = sys.argv.index(func_ref)
        return " ".join(sys.argv[func_ref_arg_idx:])
    except ValueError:
        return " ".join(sys.argv)


def _get_click_command_for_function(app: App, function_tag):
    function = app.indexed_objects[function_tag]
    assert isinstance(function, Function)
    function = typing.cast(Function, function)
    if function.is_generator:
        raise InvalidError("`modal run` is not supported for generator functions")

    signature: Dict[str, ParameterMetadata]
    cls: Optional[Cls] = None
    method_name: Optional[str] = None
    if function.info.user_cls is not None:
        class_name, method_name = function_tag.rsplit(".", 1)
        cls = typing.cast(Cls, app.indexed_objects[class_name])
        cls_signature = _get_signature(function.info.user_cls)
        fun_signature = _get_signature(function.info.raw_f, is_method=True)
        signature = dict(**cls_signature, **fun_signature)  # Pool all arguments
        # TODO(erikbern): assert there's no overlap?
    else:
        signature = _get_signature(function.info.raw_f)

    @click.pass_context
    def f(ctx, **kwargs):
        show_progress: bool = ctx.obj["show_progress"]
        with enable_output(show_progress):
            with run_app(
                app,
                detach=ctx.obj["detach"],
                environment_name=ctx.obj["env"],
                interactive=ctx.obj["interactive"],
            ):
                if cls is None:
                    function.remote(**kwargs)
                else:
                    # unpool class and method arguments
                    # TODO(erikbern): this code is a bit hacky
                    cls_kwargs = {k: kwargs[k] for k in cls_signature}
                    fun_kwargs = {k: kwargs[k] for k in fun_signature}

                    instance = cls(**cls_kwargs)
                    method: Function = getattr(instance, method_name)
                    method.remote(**fun_kwargs)

    with_click_options = _add_click_options(f, signature)
    return click.command(with_click_options)


def _get_click_command_for_local_entrypoint(app: App, entrypoint: LocalEntrypoint):
    func = entrypoint.info.raw_f
    isasync = inspect.iscoroutinefunction(func)

    @click.pass_context
    def f(ctx, *args, **kwargs):
        if ctx.obj["detach"]:
            print(
                "Note that running a local entrypoint in detached mode only keeps the last "
                "triggered Modal function alive after the parent process has been killed or disconnected."
            )

        show_progress: bool = ctx.obj["show_progress"]
        with enable_output(show_progress):
            with run_app(
                app,
                detach=ctx.obj["detach"],
                environment_name=ctx.obj["env"],
                interactive=ctx.obj["interactive"],
            ):
                try:
                    if isasync:
                        asyncio.run(func(*args, **kwargs))
                    else:
                        func(*args, **kwargs)
                except Exception as exc:
                    raise _CliUserExecutionError(inspect.getsourcefile(func)) from exc

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
        app: App = function_or_entrypoint.app
        if app.description is None:
            app.set_description(_get_clean_app_description(func_ref))
        if isinstance(function_or_entrypoint, LocalEntrypoint):
            click_command = _get_click_command_for_local_entrypoint(app, function_or_entrypoint)
        else:
            tag = function_or_entrypoint.info.get_tag()
            click_command = _get_click_command_for_function(app, tag)

        return click_command


@click.group(
    cls=RunGroup,
    subcommand_metavar="FUNC_REF",
)
@click.option("-q", "--quiet", is_flag=True, help="Don't show Modal progress indicators.")
@click.option("-d", "--detach", is_flag=True, help="Don't stop the app if the local process dies or disconnects.")
@click.option("-i", "--interactive", is_flag=True, help="Run the app in interactive mode.")
@click.option("-e", "--env", help=ENV_OPTION_HELP, default=None)
@click.pass_context
def run(ctx, detach, quiet, interactive, env):
    """Run a Modal function or local entrypoint.

    `FUNC_REF` should be of the format `{file or module}::{function name}`.
    Alternatively, you can refer to the function via the app:

    `{file or module}::{app variable name}.{function name}`

    **Examples:**

    To run the hello_world function (or local entrypoint) in my_app.py:

    ```
    modal run my_app.py::hello_world
    ```

    If your module only has a single app called `app` and your app has a
    single local entrypoint (or single function), you can omit the app and
    function parts:

    ```
    modal run my_app.py
    ```

    Instead of pointing to a file, you can also use the Python module path:

    ```
    modal run my_project.my_app
    ```
    """
    ctx.ensure_object(dict)
    ctx.obj["detach"] = detach  # if subcommand would be a click command...
    ctx.obj["show_progress"] = False if quiet else True
    ctx.obj["interactive"] = interactive


def deploy(
    app_ref: str = typer.Argument(..., help="Path to a Python file with an app."),
    name: str = typer.Option("", help="Name of the deployment."),
    env: str = ENV_OPTION,
    stream_logs: bool = typer.Option(False, help="Stream logs from the app upon deployment."),
    tag: str = typer.Option("", help="Tag the deployment with a version."),
):
    # this ensures that `modal.lookup()` without environment specification uses the same env as specified
    env = ensure_env(env)

    app = import_app(app_ref)

    if name is None:
        name = app.name

    with enable_output():
        res = deploy_app(app, name=name, environment_name=env or "", tag=tag)
        if res.warnings:
            console = Console()
            for warning in res.warnings:
                panel = Panel(warning, title="Warning", title_align="left", border_style="yellow")
                console.print(panel, highlight=False)

    if stream_logs:
        stream_app_logs(app_id=res.app_id, app_logs_url=res.app_logs_url)


def serve(
    app_ref: str = typer.Argument(..., help="Path to a Python file with an app."),
    timeout: Optional[float] = None,
    env: str = ENV_OPTION,
):
    """Run a web endpoint(s) associated with a Modal app and hot-reload code.

    **Examples:**

    ```
    modal serve hello_world.py
    ```
    """
    env = ensure_env(env)

    app = import_app(app_ref)
    if app.description is None:
        app.set_description(_get_clean_app_description(app_ref))

    with enable_output():
        with serve_app(app, app_ref, environment_name=env):
            if timeout is None:
                timeout = config["serve_timeout"]
            if timeout is None:
                timeout = float("inf")
            while timeout > 0:
                t = min(timeout, 3600)
                time.sleep(t)
                timeout -= t


def shell(
    container_or_function: Optional[str] = typer.Argument(
        default=None,
        help=(
            "ID of running container, or path to a Python file containing a Modal App."
            " Can also include a function specifier, like `module.py::func`, if the file defines multiple functions."
        ),
        metavar="REF",
    ),
    cmd: str = typer.Option("/bin/bash", "-c", "--cmd", help="Command to run inside the Modal image."),
    env: str = ENV_OPTION,
    image: Optional[str] = typer.Option(
        default=None, help="Container image tag for inside the shell (if not using REF)."
    ),
    add_python: Optional[str] = typer.Option(default=None, help="Add Python to the image (if not using REF)."),
    volume: Optional[typing.List[str]] = typer.Option(
        default=None,
        help=(
            "Name of a `modal.Volume` to mount inside the shell at `/mnt/{name}` (if not using REF)."
            " Can be used multiple times."
        ),
    ),
    cpu: Optional[int] = typer.Option(default=None, help="Number of CPUs to allocate to the shell (if not using REF)."),
    memory: Optional[int] = typer.Option(
        default=None, help="Memory to allocate for the shell, in MiB (if not using REF)."
    ),
    gpu: Optional[str] = typer.Option(
        default=None,
        help="GPUs to request for the shell, if any. Examples are `any`, `a10g`, `a100:4` (if not using REF).",
    ),
    cloud: Optional[str] = typer.Option(
        default=None,
        help=(
            "Cloud provider to run the shell on. " "Possible values are `aws`, `gcp`, `oci`, `auto` (if not using REF)."
        ),
    ),
    region: Optional[str] = typer.Option(
        default=None,
        help=(
            "Region(s) to run the container on. "
            "Can be a single region or a comma-separated list to choose from (if not using REF)."
        ),
    ),
):
    """Run an interactive shell inside a Modal container.

    **Examples:**

    Start a shell inside the default Debian-based image:

    ```
    modal shell
    ```

    Start a bash shell using the spec for `my_function` in your App:

    ```
    modal shell hello_world.py::my_function
    ```

    Or, if you're using a [modal.Cls](/docs/reference/modal.Cls), you can refer to a `@modal.method` directly:

    ```
    modal shell hello_world.py::MyClass.my_method
    ```

    Start a `python` shell:

    ```
    modal shell hello_world.py --cmd=python
    ```
    """
    env = ensure_env(env)

    console = Console()
    if not console.is_terminal:
        raise click.UsageError("`modal shell` can only be run from a terminal.")

    if platform.system() == "Windows":
        raise InvalidError("`modal shell` is currently not supported on Windows")

    app = App("modal shell")

    if container_or_function is not None:
        # `modal shell` with a container ID is a special case, alias for `modal container exec`.
        if (
            container_or_function.startswith("ta-")
            and len(container_or_function[3:]) > 0
            and container_or_function[3:].isalnum()
        ):
            from .container import exec

            exec(container_id=container_or_function, command=shlex.split(cmd), pty=True)
            return

        function = import_function(
            container_or_function, accept_local_entrypoint=False, accept_webhook=True, base_cmd="modal shell"
        )
        assert isinstance(function, Function)
        function_spec: _FunctionSpec = function.spec
        start_shell = partial(
            interactive_shell,
            image=function_spec.image,
            mounts=function_spec.mounts,
            secrets=function_spec.secrets,
            network_file_systems=function_spec.network_file_systems,
            gpu=function_spec.gpus,
            cloud=function_spec.cloud,
            cpu=function_spec.cpu,
            memory=function_spec.memory,
            volumes=function_spec.volumes,
            region=function_spec.scheduler_placement.proto.regions if function_spec.scheduler_placement else None,
        )
    else:
        modal_image = Image.from_registry(image, add_python=add_python) if image else None
        volumes = {} if volume is None else {f"/mnt/{vol}": Volume.from_name(vol) for vol in volume}
        start_shell = partial(
            interactive_shell,
            image=modal_image,
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            cloud=cloud,
            volumes=volumes,
            region=region.split(",") if region else [],
        )

    # NB: invoking under bash makes --cmd a lot more flexible.
    cmds = shlex.split(f'/bin/bash -c "{cmd}"')
    start_shell(app, cmds=cmds, environment_name=env, timeout=3600)
