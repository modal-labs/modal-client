# Copyright Modal Labs 2025
import dataclasses
import platform
import shlex
from collections.abc import Collection, Sequence
from pathlib import PurePosixPath
from typing import Any, Optional, Union, cast

import typer
from click import ClickException

from .._functions import _FunctionSpec
from ..app import App
from ..cloud_bucket_mount import _CloudBucketMount
from ..exception import InvalidError, NotFoundError
from ..functions import Function
from ..gpu import GPU_T
from ..image import Image, _Image
from ..mount import _Mount
from ..network_file_system import _NetworkFileSystem
from ..proxy import _Proxy
from ..runner import interactive_shell
from ..secret import Secret, _Secret
from ..volume import Volume, _Volume
from .import_refs import MethodReference, import_and_filter, parse_import_ref
from .utils import ENV_OPTION, is_tty, is_valid_modal_id


@dataclasses.dataclass
class _ShellKwargs:
    """Container for shell keyword arguments that can be passed to Sandbox._create."""

    image: Optional[_Image] = None
    mounts: Optional[Sequence[_Mount]] = None
    secrets: Optional[Collection[_Secret]] = None
    network_file_systems: Optional[dict[Union[str, PurePosixPath], _NetworkFileSystem]] = None
    gpu: Optional[Union[GPU_T, list[GPU_T]]] = None
    cloud: Optional[str] = None
    cpu: Optional[float | tuple[float, float]] = None
    memory: Optional[int | tuple[int, int]] = None
    volumes: Optional[dict[Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]]] = None
    region: Optional[Union[str, Sequence[str]]] = None
    proxy: Optional[_Proxy] = None
    pty: Optional[bool] = None

    def to_kwargs(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}

    @staticmethod
    def from_function_spec(function_spec: _FunctionSpec) -> "_ShellKwargs":
        """Create _ShellKwargs from a function spec."""
        return _ShellKwargs(
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
            proxy=function_spec.proxy,
        )


def _get_runnable_list(all_usable_commands) -> str:
    usable_command_lines = []
    for cmd in all_usable_commands:
        cmd_names = " / ".join(cmd.names)
        usable_command_lines.append(cmd_names)

    return "\n".join(usable_command_lines)


def _shell_in_container(task_id: str, cmd: str, pty: bool, sandbox_id: Optional[str] = None) -> None:
    from .container import exec

    sandbox_str = f" (Sandbox '{sandbox_id}')" if sandbox_id else ""
    try:
        exec(container_id=task_id, command=shlex.split(cmd), pty=pty)
    except NotFoundError:
        raise ClickException(f"Container '{task_id}'{sandbox_str} not found. Is it running?")
    except Exception as e:
        raise ClickException(f"Error connecting to container '{task_id}'{sandbox_str}: {str(e)}")


def _shell_in_sandbox(sandbox_id: str, cmd: str, pty: bool) -> None:
    from ..sandbox import Sandbox

    try:
        sandbox = Sandbox.from_id(sandbox_id)
        task_id = sandbox._get_task_id()
    except NotFoundError:
        raise ClickException(f"Sandbox '{sandbox_id}' not found")
    except Exception as e:
        raise ClickException(f"Error identifying container for Sandbox '{sandbox_id}': {str(e)}")

    _shell_in_container(task_id, cmd, pty)


def _ref_to_function_spec(ref: str, use_module_mode: bool) -> _FunctionSpec:
    import_ref = parse_import_ref(ref, use_module_mode=use_module_mode)
    runnable, all_usable_commands = import_and_filter(
        import_ref, base_cmd="modal shell", accept_local_entrypoint=False, accept_webhook=True
    )
    if not runnable:
        help_header = (
            "Specify a Modal function to start a shell session for. E.g.\n"
            f"> modal shell {import_ref.file_or_module}::my_function"
        )

        if all_usable_commands:
            help_footer = f"The selected module '{import_ref.file_or_module}' has the following choices:\n\n"
            help_footer += _get_runnable_list(all_usable_commands)
        else:
            help_footer = f"The selected module '{import_ref.file_or_module}' has no Modal functions or classes."

        raise ClickException(f"{help_header}\n\n{help_footer}")

    function_spec: _FunctionSpec
    if isinstance(runnable, MethodReference):
        # TODO: let users specify a class instead of a method, since they use the same environment
        class_service_function = runnable.cls._get_class_service_function()
        function_spec = class_service_function.spec
    elif isinstance(runnable, Function):
        function_spec = runnable.spec
    else:
        raise ValueError("Referenced entity is not a Modal function or class")

    return function_spec


def shell(
    ref: Optional[str] = typer.Argument(
        default=None,
        help=(
            "ID of running container or Sandbox, or path to a Python file containing an App."
            " Can also include a Function specifier, like `module.py::func`, if the file defines multiple Functions."
        ),
    ),
    cmd: str = typer.Option("/bin/bash", "-c", "--cmd", help="Command to run inside the Modal image."),
    env: str = ENV_OPTION,
    image: Optional[str] = typer.Option(
        default=None, help="Container image tag for inside the shell (if not using REF)."
    ),
    add_python: Optional[str] = typer.Option(default=None, help="Add Python to the image (if not using REF)."),
    volume: Optional[list[str]] = typer.Option(
        default=None,
        help=(
            "Name of a `modal.Volume` to mount inside the shell at `/mnt/{name}` (if not using REF)."
            " Can be used multiple times."
        ),
    ),
    secret: Optional[list[str]] = typer.Option(
        default=None,
        help=("Name of a `modal.Secret` to mount inside the shell (if not using REF). Can be used multiple times."),
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
            "Cloud provider to run the shell on. Possible values are `aws`, `gcp`, `oci`, `auto` (if not using REF)."
        ),
    ),
    region: Optional[str] = typer.Option(
        default=None,
        help=(
            "Region(s) to run the container on. "
            "Can be a single region or a comma-separated list to choose from (if not using REF)."
        ),
    ),
    pty: Optional[bool] = typer.Option(default=None, help="Run the command using a PTY."),
    use_module_mode: bool = typer.Option(
        False, "-m", help="Interpret argument as a Python module path instead of a file/script path"
    ),
):
    """Run a command or interactive shell inside a Modal container.

    **Examples:**

    Start an interactive shell inside the default Debian-based image:

    ```
    modal shell
    ```

    Start an interactive shell with the spec for `my_function` in your App
    (uses the same image, volumes, mounts, etc.):

    ```
    modal shell hello_world.py::my_function
    ```

    Or, if you're using a [modal.Cls](https://modal.com/docs/reference/modal.Cls)
    you can refer to a `@modal.method` directly:

    ```
    modal shell hello_world.py::MyClass.my_method
    ```

    Start a `python` shell:

    ```
    modal shell hello_world.py --cmd=python
    ```

    Run a command with your function's spec and pipe the output to a file:

    ```
    modal shell hello_world.py -c 'uv pip list' > env.txt
    ```

    Connect to a running Sandbox by ID:

    ```
    modal shell sb-abc123xyz
    ```
    """
    from ..environments import ensure_env

    env = ensure_env(env)

    if pty is None:
        pty = is_tty()

    if platform.system() == "Windows":
        raise InvalidError("`modal shell` is currently not supported on Windows")

    if ref and (is_valid_modal_id(ref, "sb") or is_valid_modal_id(ref, "ta")):
        mutually_exclusive = {
            "--image": image,
            "--add-python": add_python,
            "--volume": volume,
            "--secret": secret,
            "--cpu": cpu,
            "--memory": memory,
            "--gpu": gpu,
            "--cloud": cloud,
            "--region": region,
        }
        if provided := [k for k, v in mutually_exclusive.items() if v]:
            raise InvalidError(f"Cannot use {', '.join(provided)} with a Modal container/sandbox ID")

    if ref and is_valid_modal_id(ref, "sb"):
        _shell_in_sandbox(ref, cmd, pty)
        return

    if ref and is_valid_modal_id(ref, "ta"):
        _shell_in_container(ref, cmd, pty)
        return

    function_spec = _ref_to_function_spec(ref, use_module_mode) if ref else None

    # Start with function spec kwargs if available
    shell_kwargs = _ShellKwargs.from_function_spec(function_spec) if function_spec else _ShellKwargs()

    # Override with CLI arguments
    if image is not None:
        shell_kwargs.image = cast(_Image, Image.from_registry(image, add_python=add_python))
    if volume is not None:
        shell_kwargs.volumes = {f"/mnt/{vol}": cast(_Volume, Volume.from_name(vol)) for vol in volume}
    if secret is not None:
        shell_kwargs.secrets = [cast(_Secret, Secret.from_name(s)) for s in secret]
    if cpu is not None:
        shell_kwargs.cpu = cpu
    if memory is not None:
        shell_kwargs.memory = memory
    if gpu is not None:
        shell_kwargs.gpu = gpu
    if cloud is not None:
        shell_kwargs.cloud = cloud
    if region is not None:
        shell_kwargs.region = region.split(",")
    shell_kwargs.pty = pty

    # NB: invoking under bash makes --cmd a lot more flexible.
    cmds = shlex.split(f'/bin/bash -c "{cmd}"')

    app = App("modal shell")
    interactive_shell(
        app,
        cmds=cmds,
        environment_name=env,
        timeout=3600,
        **shell_kwargs.to_kwargs(),
    )
