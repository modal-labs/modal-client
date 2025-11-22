# Copyright Modal Labs 2022
import platform
import shlex
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Optional

import typer
from click import ClickException

from .._functions import _FunctionSpec
from ..app import App
from ..environments import ensure_env
from ..exception import InvalidError, NotFoundError
from ..functions import Function
from ..image import Image
from ..mount import _Mount
from ..runner import interactive_shell
from ..secret import Secret
from ..volume import Volume
from .import_refs import (
    MethodReference,
    import_and_filter,
    parse_import_ref,
)
from .run import _get_runnable_list
from .utils import ENV_OPTION, is_tty


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
    add_local: Optional[list[str]] = typer.Option(
        default=None,
        help=(
            "Local file or directory to mount inside the shell at `/mnt/{basename}` (if not using REF)."
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
    env = ensure_env(env)

    if pty is None:
        pty = is_tty()

    if platform.system() == "Windows":
        raise InvalidError("`modal shell` is currently not supported on Windows")

    app = App("modal shell")

    if ref is not None:
        # `modal shell` with a sandbox ID gets the task_id, that's then handled by the `ta-*` flow below.
        if ref.startswith("sb-") and len(ref[3:]) > 0 and ref[3:].isalnum():
            from ..sandbox import Sandbox

            try:
                sandbox = Sandbox.from_id(ref)
                task_id = sandbox._get_task_id()
                ref = task_id
            except NotFoundError as e:
                raise ClickException(f"Sandbox '{ref}' not found")
            except Exception as e:
                raise ClickException(f"Error connecting to sandbox '{ref}': {str(e)}")

        # `modal shell` with a container ID is a special case, alias for `modal container exec`.
        if ref.startswith("ta-") and len(ref[3:]) > 0 and ref[3:].isalnum():
            from .container import exec

            exec(container_id=ref, command=shlex.split(cmd), pty=pty)
            return

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
            region=function_spec.scheduler_placement.regions if function_spec.scheduler_placement else None,
            pty=pty,
            proxy=function_spec.proxy,
        )
    else:
        modal_image = Image.from_registry(image, add_python=add_python) if image else None
        volumes = {} if volume is None else {f"/mnt/{vol}": Volume.from_name(vol) for vol in volume}
        secrets = [] if secret is None else [Secret.from_name(s) for s in secret]

        mounts = []
        if add_local:
            for local_path_str in add_local:
                local_path = Path(local_path_str).expanduser().resolve()
                remote_path = PurePosixPath(f"/mnt/{local_path.name}")

                if local_path.is_dir():
                    m = _Mount._from_local_dir(local_path, remote_path=remote_path)
                else:
                    m = _Mount._from_local_file(local_path, remote_path=remote_path)
                mounts.append(m)

        start_shell = partial(
            interactive_shell,
            image=modal_image,
            mounts=mounts,
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            cloud=cloud,
            volumes=volumes,
            secrets=secrets,
            region=region.split(",") if region else [],
            pty=pty,
        )

    # Invoking under sh makes --cmd a lot more flexible.
    # We use /bin/sh rather than bash because e.g. alpine images don't come with bash.
    cmds = ["/bin/sh", "-c", cmd]
    start_shell(app, cmds=cmds, environment_name=env, timeout=3600)
