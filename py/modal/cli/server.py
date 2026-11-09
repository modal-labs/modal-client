# Copyright Modal Labs 2026
from __future__ import annotations

import dataclasses
import json as json_lib
import math
import os
import shlex
import sys
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import AsyncGenerator, Sequence, cast

import click
from click import UsageError
from rich.box import SIMPLE, SIMPLE_HEAD
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text

from modal._environments import ensure_env
from modal._image import _Image, _upgrade_image_name
from modal._logs import _fetch_time_range, _seconds_to_timestamp, _timestamp_to_seconds
from modal._object import _get_environment_name
from modal._partial_function import _sessioned
from modal._server import _DirectServerCls, _Server
from modal._utils.async_utils import async_map_ordered, synchronizer
from modal._utils.name_utils import is_valid_object_name
from modal._utils.time_utils import parse_duration, timestamp_to_localized_str
from modal.app import _App
from modal.cli.utils import humanize_filesize
from modal.client import _Client
from modal.cloud_bucket_mount import _CloudBucketMount
from modal.exception import InvalidError, LogsFetchError, NotFoundError
from modal.output import OutputManager
from modal.runner import _deploy_app
from modal.secret import _Secret
from modal.types import ServerAutoscalerSettings
from modal.volume import _Volume
from modal_proto import api_pb2

from ._help import ModalGroup
from ._logs import _parse_fetch_range, _parse_time_arg, _run_logs_command, _validate_logs_args
from ._stats import (
    _DEFAULT_STATS_WINDOW,
    STATS_HEADING_STYLE,
    STATS_METADATA_STYLE,
    STATS_PROBLEM_STYLE,
    STATS_SECTION_STYLE,
    _distribution_map_json,
    _export_metrics,
    _metric_rows,
    _percentile_table,
    _timestamp,
    percentile_style,
    problem_style,
    progress_style,
    stats_style,
    success_style,
)
from .utils import (
    HEADER_ONLY,
    _resolve_function_id,
    _stream_app_logs,
    confirm_or_suggest_yes,
    display_table,
    env_option,
    grouped_utc_timestamp,
    parse_int,
    yes_option,
)

server_cli = ModalGroup(name="server", help="Deploy and inspect Modal Servers.")

_DEFAULT_REQUEST_TAIL = 10
_MAX_REQUEST_TAIL = 1000
_MAX_REQUEST_FETCH = 1_000_000


def _server_request_status_cell(status: int, no_color: bool = False) -> Text:
    return Text(str(status), style="red") if status >= 400 and not no_color else Text(str(status))


# Deploying from an image declares exactly one Server per App, so it always gets this name.
_SERVER_NAME = "Server"


async def _resolve_image(
    image_ref: str,
    environment_name: str,
    registry_secret: _Secret | None,
    registry_secret_flag: str,
    force_build: bool,
) -> _Image:
    """A published Modal Image with this name, or else the registry image it references."""
    client = await _Client.from_env()
    try:
        tag = _upgrade_image_name(image_ref, fallback_environment_name=environment_name)
        response = await client._stub.ImageGetByTag(api_pb2.ImageGetByTagRequest(tag=tag))
    except (NotFoundError, InvalidError):
        pass
    else:
        if registry_secret is not None:
            raise click.BadParameter(
                f"only applies to registry images, but '{image_ref}' is a published Modal Image.",
                param_hint=registry_secret_flag,
            )
        if force_build:
            raise click.BadParameter(
                f"only applies to registry images, but '{image_ref}' is a published Modal Image.",
                param_hint="--force-build",
            )
        # `from_id` returns the public `Image`, so translate it back for use in async code.
        return cast(_Image, synchronizer._translate_in(_Image.from_id(response.image_id)))

    host, separator, repository = image_ref.partition("/")
    # A first component counts as a registry host only when it looks like one, which is how
    # container tooling tells `docker.io/nginx` apart from the `library/nginx` shorthand.
    if not separator or not repository or not ("." in host or ":" in host or host == "localhost"):
        raise click.BadParameter(
            f"'{image_ref}' is not a published Modal Image or a registry reference such as 'docker.io/nginx'.",
            param_hint="--image",
        )
    return _Image.from_registry(image_ref, secret=registry_secret, force_build=force_build)


def _infer_app_name(image_ref: str) -> str:
    """The image's repository name without its registry, namespace, tag, or digest."""
    app_name = image_ref.partition("@")[0].rpartition("/")[2].partition(":")[0]
    if not is_valid_object_name(app_name):
        raise UsageError(f"Cannot infer an App name from '{image_ref}'; pass one with --app-name.")
    return app_name


def _parse_env_vars(env_vars: tuple[str, ...]) -> dict[str, str | None]:
    parsed: dict[str, str | None] = {}
    for env_var in env_vars:
        key, separator, value = env_var.partition("=")
        if not key or not separator:
            raise click.BadParameter("must use KEY=VALUE", param_hint="--env-var")
        parsed[key] = value
    return parsed


def _secret_from_env(keys_csv: str | None, flag: str) -> _Secret | None:
    if keys_csv is None:
        return None
    keys = [key.strip() for key in keys_csv.split(",") if key.strip()]
    if not keys:
        raise click.BadParameter("must name at least one environment variable", param_hint=flag)
    if missing := [key for key in keys if key not in os.environ]:
        raise click.BadParameter(f"not set in the local environment: {', '.join(missing)}", param_hint=flag)
    return _Secret.from_local_environ(keys)


class _NumberOrNone(click.ParamType):
    """A number in a range, or the string "none" for a setting's no-limit value."""

    def __init__(self, number_type: click.FloatRange | click.IntRange, none_value: float | None, description: str):
        # Typed as the base ParamType: convert() parses the raw CLI string, which the range types' stubs reject.
        self.number_type: click.ParamType = number_type
        self.none_value = none_value
        self.description = description
        self.name = f"{number_type.name} or none"

    def convert(self, value, param, ctx):
        if not isinstance(value, str):
            return value
        if value.strip().lower() == "none":
            return self.none_value
        try:
            number = self.number_type.convert(value, param, ctx)
        except click.BadParameter:
            number = math.nan
        if not math.isfinite(number):
            self.fail(f"{value!r} is not {self.description}.", param, ctx)
        return number


class _RequestLimit(click.ParamType):
    """A resource request, optionally followed by a comma and a hard limit."""

    def __init__(self, number_type: click.FloatRange | click.IntRange):
        # Typed as the base ParamType: convert() parses the raw CLI string, which the range types' stubs reject.
        self.number_type: click.ParamType = number_type
        self.name = f"{number_type.name}[,limit]"

    def convert(self, value, param, ctx):
        if not isinstance(value, str):
            return value
        parts = [part.strip() for part in value.split(",")]
        if len(parts) > 2:
            self.fail(f"{value!r} must be REQUEST or REQUEST,LIMIT.", param, ctx)
        numbers = [self.number_type.convert(part, param, ctx) for part in parts]
        if not all(math.isfinite(number) for number in numbers):
            self.fail(f"{value!r} must be finite.", param, ctx)
        if len(numbers) == 1:
            return numbers[0]
        request, limit = numbers
        if limit < request:
            self.fail(f"limit {limit} is lower than request {request}.", param, ctx)
        return (request, limit)


class _GpuRequest(click.ParamType):
    """A GPU type, or a comma-separated list of GPU types to fall back through in order."""

    name = "gpu"

    def convert(self, value, param, ctx):
        if not isinstance(value, str):
            return value
        gpus = [gpu.strip() for gpu in value.split(",")]
        if not all(gpus):
            self.fail(f"{value!r} must be GPU or GPU,FALLBACK,... with no empty entries.", param, ctx)
        return gpus[0] if len(gpus) == 1 else gpus


def _parse_volume_specs(volumes: tuple[str, ...]) -> dict[str, str]:
    """The Volume names by mount path."""
    parsed: dict[str, str] = {}
    for volume_spec in volumes:
        name, separator, mount_path = volume_spec.partition(":")
        if not name or not separator or not mount_path.startswith("/"):
            raise click.BadParameter("must use NAME:/absolute/path", param_hint="--volume")
        path = str(PurePosixPath(mount_path))
        if path in parsed:
            raise click.BadParameter(f"more than one Volume is mounted at {path}", param_hint="--volume")
        parsed[path] = name
    return parsed


async def _resolve_volumes(
    volume_names: dict[str, str], environment_name: str
) -> tuple[dict[str | PurePosixPath, _Volume | _CloudBucketMount], list[str]]:
    """The Volume mounts by path, and the names of Volumes that deploying will create."""
    resolved: dict[str | PurePosixPath, _Volume | _CloudBucketMount] = {}
    missing: list[str] = []
    for mount_path, name in volume_names.items():
        volume = _Volume.from_name(name, environment_name=environment_name)
        try:
            await volume.hydrate()
        except NotFoundError:
            volume = _Volume.from_name(name, environment_name=environment_name, create_if_missing=True)
            missing.append(name)
        resolved[mount_path] = volume
    return resolved, missing


@server_cli.command(
    "deploy",
    no_args_is_help=True,
)
@click.option(
    "--image",
    "image_ref",
    required=True,
    help=(
        "Container image including its registry, e.g. 'docker.io/nginx', "
        "or the name of a published Modal Image, e.g. 'my-image:v1'."
    ),
)
@click.option("--port", required=True, type=int, help="Port the server listens on inside the container.")
@click.option(
    "--registry-secret",
    "registry_secret_name",
    default=None,
    help=(
        "Name of a Modal Secret with credentials for pulling a private registry image. Its keys select the kind "
        "of authentication: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_REGION (optionally "
        "AWS_SESSION_TOKEN), or AWS_ROLE_ARN and AWS_REGION, for AWS ECR; SERVICE_ACCOUNT_JSON for GCP; or "
        "REGISTRY_USERNAME and REGISTRY_PASSWORD for any other registry."
    ),
)
@click.option(
    "--registry-secret-from-env",
    "registry_secret_env_keys",
    default=None,
    metavar="KEYS",
    help=(
        "Comma-separated names of local environment variables whose values are used to pull a private "
        "registry image, e.g. REGISTRY_USERNAME,REGISTRY_PASSWORD. Accepts the same keys as --registry-secret."
    ),
)
@click.option(
    "--force-build",
    is_flag=True,
    default=False,
    help=(
        "Pull the registry image again instead of using a cached pull, e.g. after its tag is updated. "
        "Later builds from the same image reference also use the new pull."
    ),
)
@click.option(
    "--entrypoint",
    default=None,
    help=(
        "Override the image's ENTRYPOINT, which also clears its CMD unless --cmd is given. "
        "--cmd is appended as arguments, e.g. --entrypoint 'python -m' --cmd 'http.server 8000'."
    ),
)
@click.option(
    "--cmd",
    default=None,
    help=(
        "Command to start the server, overriding the image's CMD, e.g. 'python -m http.server 8000'. "
        "Split into arguments with shell quoting rules, so quote an argument that contains spaces, "
        "e.g. --cmd 'sh -c \"python -m http.server 8000\"'."
    ),
)
@click.option(
    "--env-var",
    "env_vars",
    multiple=True,
    metavar="KEY=VALUE",
    help="Set an environment variable in the container, e.g. LOG_LEVEL=debug. Repeatable.",
)
@click.option(
    "--secret",
    "secret_names",
    multiple=True,
    help=(
        "Name of a Modal Secret whose keys are set as environment variables in the running container. "
        "Repeatable. Unlike --registry-secret, it is not used to pull the image."
    ),
)
@click.option(
    "--secret-from-env",
    "secret_env_keys",
    default=None,
    metavar="KEYS",
    help=(
        "Comma-separated names of local environment variables whose values are set as environment variables "
        "in the running container, e.g. API_KEY,API_TOKEN. Unlike --registry-secret-from-env, they are not "
        "used to pull the image."
    ),
)
@click.option(
    "--volume",
    "volume_specs",
    multiple=True,
    metavar="NAME:PATH",
    help="Mount a Modal Volume at an absolute path in the container, creating it if needed. Repeatable.",
)
@click.option(
    "--cpu",
    default=None,
    type=_RequestLimit(click.FloatRange(min=0, min_open=True)),
    metavar="CORES[,LIMIT]",
    help=(
        "CPU cores to request, which may be fractional, e.g. 2. Add a comma and a hard limit to also cap usage, "
        "e.g. 2,4."
    ),
)
@click.option(
    "--memory",
    default=None,
    type=_RequestLimit(click.IntRange(min=1)),
    metavar="MIB[,LIMIT]",
    help="Memory in MiB to request, e.g. 4096. Add a comma and a hard limit to also cap usage, e.g. 4096,8192.",
)
@click.option(
    "--gpu",
    default=None,
    type=_GpuRequest(),
    metavar="GPU[,FALLBACK...]",
    help=(
        "GPU request, e.g. 'A100' or 'H100:8'. Pass a comma-separated list to fall back to the next GPU type "
        "when the earlier ones are unavailable, e.g. 'H100,A100'."
    ),
)
@click.option(
    "--region",
    "routing_region",
    default="us-east",
    show_default=True,
    help="Region where Server requests enter Modal before reaching a container.",
)
@click.option(
    "--unauthenticated",
    is_flag=True,
    default=False,
    help=(
        "Accept requests without credentials. By default, requests must carry a workspace proxy token in "
        "Modal-Key and Modal-Secret headers; see 'modal workspace proxy-tokens'."
    ),
)
@click.option(
    "--h2",
    "h2_enabled",
    is_flag=True,
    default=False,
    help="Forward requests to the container over HTTP/2. The server must accept cleartext HTTP/2 (h2c) on --port.",
)
@click.option(
    "--sessioned",
    is_flag=True,
    default=False,
    help=(
        "Enable sticky sessions. Clients start a session with 'Server.sessions.start()' and send its token in "
        "a 'Modal-Authorization: Bearer <token>' header; requests with the same token reach the same container."
    ),
)
@click.option(
    "--target-concurrency",
    default=None,
    type=_NumberOrNone(click.FloatRange(min=0, min_open=True), 0.0, "a positive number or 'none'"),
    metavar="FLOAT|none",
    help=(
        "Concurrent requests each container aims to handle before scaling up. May be fractional. "
        "'none' keeps the server on a single container."
    ),
)
@click.option(
    "--max-concurrency",
    default=None,
    type=_NumberOrNone(click.IntRange(min=1), 0, "a positive integer or 'none'"),
    metavar="INTEGER|none",
    help="Maximum concurrent requests per container; requests above it receive a 429. 'none' means unlimited.",
)
@click.option("--min-containers", default=None, type=int, help="Minimum number of containers to keep warm.")
@click.option(
    "--max-containers",
    default=None,
    type=_NumberOrNone(click.IntRange(min=1), None, "a positive integer or 'none'"),
    metavar="INTEGER|none",
    help="Maximum number of containers. 'none' means bounded only by your workspace's container limit.",
)
@click.option("--buffer-containers", default=None, type=int, help="Additional idle containers under active load.")
@click.option(
    "--scaleup-window", default=None, type=int, help="Seconds of sustained demand before scaling up containers."
)
@click.option(
    "--scaledown-window", default=None, type=int, help="Maximum seconds idle containers wait before scaling down."
)
@click.option(
    "--startup-timeout",
    default=30,
    type=int,
    show_default=True,
    help="Maximum container startup time in seconds.",
)
@click.option(
    "--exit-grace-period",
    default=0,
    type=int,
    show_default=True,
    help="Seconds to let in-flight requests finish on container shutdown.",
)
@click.option(
    "--app-name",
    default=None,
    help="App to deploy under. Defaults to the image's repository name, e.g. 'api' for 'ghcr.io/org/api:v2'.",
)
@click.option("--stream-logs", is_flag=True, default=False, help="Stream logs from the Server upon deployment.")
@yes_option
@env_option
@synchronizer.create_blocking
async def deploy(
    image_ref: str,
    port: int,
    app_name: str | None = None,
    registry_secret_name: str | None = None,
    registry_secret_env_keys: str | None = None,
    force_build: bool = False,
    entrypoint: str | None = None,
    cmd: str | None = None,
    routing_region: str = "us-east",
    startup_timeout: int = 30,
    exit_grace_period: int = 0,
    unauthenticated: bool = False,
    h2_enabled: bool = False,
    cpu: float | tuple[float, float] | None = None,
    memory: int | tuple[int, int] | None = None,
    gpu: str | list[str] | None = None,
    secret_names: tuple[str, ...] = (),
    secret_env_keys: str | None = None,
    env_vars: tuple[str, ...] = (),
    volume_specs: tuple[str, ...] = (),
    target_concurrency: float | None = None,
    max_concurrency: int | None = None,
    min_containers: int | None = None,
    max_containers: int | None = None,
    buffer_containers: int | None = None,
    scaleup_window: int | None = None,
    scaledown_window: int | None = None,
    sessioned: bool = False,
    stream_logs: bool = False,
    *,
    yes: bool = False,
    env: str | None = None,
) -> None:
    """Deploy a container image as a Server.

    Runs an existing container image on Modal without writing any Python. ``--image`` is either a
    container image reference that includes its registry, such as ``docker.io/nginx``, or the
    name of a Modal Image published with ``modal image build`` or ``Image.publish()``, such as
    ``my-image:v1``. A published Modal Image takes precedence over a registry image with the
    same reference. The image needs no Python and no Modal client; it only has to serve HTTP
    on ``--port``.
    The server is started by the image's own ENTRYPOINT and CMD; ``--entrypoint`` and ``--cmd``
    override them.

    A registry image is pulled once and reused by later deploys of the same reference. If a
    tag such as ``latest`` changes in the registry, pass ``--force-build`` to pull it again,
    or reference the image by digest.

    The Server is named "Server" and deployed under the App given by ``--app-name``, which
    defaults to the image's repository name without its tag, so deploying a new tag of an
    image updates the same App. For example, ``ghcr.io/org/api:v2`` and ``ghcr.io/org/api:v3``
    both deploy to the App ``api``. Other ``modal server`` commands refer to it as
    ``APP_NAME/Server``, e.g. ``modal server logs api/Server``. Redeploying an App that is
    already deployed asks for confirmation first unless ``--yes`` is given. Without an
    interactive terminal, such as in CI, redeploying aborts unless ``--yes`` is given.

    Examples:

    Deploy a public image that starts its server through its own ENTRYPOINT and CMD:

    ```
    modal server deploy --image docker.io/nginx --port 80 --unauthenticated
    ```

    Deploy a private image with GPU, a Secret, a Volume, and an environment variable:

    ```
    modal server deploy --image my-registry.com/api:v2 --port 8000 \\
        --registry-secret my-registry-creds --gpu A100 --secret my-api-keys \\
        --volume my-models:/models --env-var LOG_LEVEL=debug
    ```

    Deploy an AWS ECR image with registry credentials and an API key taken from the local
    environment:

    ```
    modal server deploy --image 123456789012.dkr.ecr.us-east-1.amazonaws.com/api:v2 --port 8000 \\
        --registry-secret-from-env AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,AWS_REGION \\
        --secret-from-env OPENAI_API_KEY
    ```

    Deploy an image with a custom entrypoint and command:

    ```
    modal server deploy --image docker.io/python:3.12 --port 8000 \\
        --entrypoint "python -m" --cmd "http.server 8000"
    ```

    Build a Dockerfile in the current directory into a named Modal Image, then deploy it:

    ```
    modal image build -t my-image:v1 .
    modal server deploy --image my-image:v1 --port 8000
    ```
    """
    env = ensure_env(env)

    # Validate local input before any request to Modal.
    parsed_env_vars = _parse_env_vars(env_vars)
    volume_names = _parse_volume_specs(volume_specs)
    env_secret = _secret_from_env(secret_env_keys, "--secret-from-env")
    if app_name is not None and not is_valid_object_name(app_name):
        raise click.BadParameter(f"'{app_name}' is not a valid App name.", param_hint="--app-name")
    if registry_secret_name and registry_secret_env_keys:
        raise UsageError("--registry-secret and --registry-secret-from-env cannot be used together.")
    if registry_secret_name:
        registry_secret = _Secret.from_name(registry_secret_name, environment_name=env)
        registry_secret_flag = "--registry-secret"
    else:
        registry_secret_flag = "--registry-secret-from-env"
        registry_secret = _secret_from_env(registry_secret_env_keys, registry_secret_flag)
    image = await _resolve_image(image_ref, env, registry_secret, registry_secret_flag, force_build)
    inferred_app_name = app_name is None
    if app_name is None:
        app_name = _infer_app_name(image_ref)
    if entrypoint is not None:
        image = image.entrypoint(shlex.split(entrypoint))
    if cmd is not None:
        image = image.cmd(shlex.split(cmd))
    secrets = [_Secret.from_name(secret_name, environment_name=env) for secret_name in secret_names]
    if env_secret:
        secrets.append(env_secret)
    volumes, missing_volumes = await _resolve_volumes(volume_names, env)

    client = await _Client.from_env()
    existing = await client._stub.AppGetByDeploymentName(
        api_pb2.AppGetByDeploymentNameRequest(name=app_name, environment_name=env)
    )
    if existing.app_id and not yes:
        msg = f"App '{app_name}' is already deployed. Redeploying replaces everything in it with this Server."
        if inferred_app_name:
            msg += " Pass --app-name to deploy under a different name."
        confirm_or_suggest_yes(f"{msg} Do you want to redeploy it?")
    for volume_name in missing_volumes:
        OutputManager.get().print(f"Creating Volume '{volume_name}', which does not exist yet.")

    app = _App(app_name)
    app.server(
        image=image,
        name=_SERVER_NAME,
        port=port,
        routing_region=routing_region,
        startup_timeout=startup_timeout,
        exit_grace_period=exit_grace_period,
        unauthenticated=unauthenticated,
        h2_enabled=h2_enabled,
        cpu=cpu,
        memory=memory,
        gpu=gpu,
        secrets=secrets,
        env=parsed_env_vars,
        volumes=volumes,
        target_concurrency=target_concurrency,
        max_concurrency=max_concurrency,
        min_containers=min_containers,
        max_containers=max_containers,
        buffer_containers=buffer_containers,
        scaleup_window=scaleup_window,
        scaledown_window=scaledown_window,
        include_source=False,
    )(_sessioned()(_DirectServerCls) if sessioned else _DirectServerCls)

    res = await _deploy_app(app, name=app_name, environment_name=env or "")

    if stream_logs:
        await _stream_app_logs(
            app_id=res.app_id,
            app_logs_url=res.app_logs_url,
            follow=True,
        )


async def _run_server_logs(
    app_id: str,
    function_id: str,
    *,
    follow: bool = False,
    since: str | None = None,
    until: str | None = None,
    tail: int | None = None,
    search: str | None = None,
    container_id: str = "",
    source: str | None = None,
    timestamps: bool = False,
    show_server_id: bool = False,
    show_container_id: bool = False,
) -> None:
    _validate_logs_args(follow=follow, since=since, until=until, tail=tail)

    prefix_fields: list[str] = []
    if show_server_id:
        prefix_fields.append("fu")
    if show_container_id:
        prefix_fields.append("ta")

    await _run_logs_command(
        app_id,
        follow=follow,
        since=since,
        until=until,
        tail=tail,
        search=search,
        function_id=function_id,
        container_id=container_id,
        source=source,
        timestamps=timestamps,
        prefix_fields=prefix_fields,
    )


@server_cli.command("info", no_args_is_help=True)
@click.argument("server_identifier", metavar="SERVER")
@click.option("--json", "json", is_flag=True, default=False, help="Output as JSON.")
@env_option
@synchronizer.create_blocking
async def info(
    server_identifier: str,
    json: bool = False,
    *,
    env: str | None = None,
):
    """Show information about a given Modal Server.

    SERVER can either be a Function ID (`fu-...`) or a deployed Server name in the format
    `APP_NAME/SERVER_NAME`. The output of this command includes information about any resources
    requested by this Server, any scheduling/autoscaling settings, any mounted Volumes or Buckets,
    and any HTTP settings.

    Examples:

    Providing a Server ID directly:

    ```
    modal server info fu-0123456789abcdefghijkl
    ```

    Referring to a Server within a deployed App:

    ```
    modal server info hello-world-app/test_server
    ```
    """
    client = await _Client.from_env()
    environment_name = _get_environment_name(ensure_env(env))
    tty = sys.stdout.isatty()

    server_id, handle_metadata, function_proto = await _resolve_function_id(
        client,
        server_identifier,
        environment_name,
        object_type="Server",
        command="info",
    )

    server = _Server._new_from_function(server_id, client, handle_metadata)
    info = await server.info()

    autoscaler_response = await client._stub.FunctionGetSchedulingParams(
        api_pb2.FunctionGetSchedulingParamsRequest(function_id=server_id)
    )
    autoscaler_settings = ServerAutoscalerSettings._from_proto(autoscaler_response.autoscaler_configuration.settings)

    output_manager = OutputManager.get()

    if json:
        info_dict = {
            "name": function_proto.function_name,
            "object_id": server.object_id,
            "app_id": handle_metadata.app_id,
        }
        info_dict = info_dict | dataclasses.asdict(info)
        info_dict = info_dict | dataclasses.asdict(autoscaler_settings)

        output_manager.print_json(json_lib.dumps(info_dict))
        return

    not_configured = "-"
    enabled = "Enabled"
    disabled = "Disabled"

    rows: list[str | Text | tuple[str | Text, str | Text] | None] = []

    rows.append((Text("Server Name:", style="bold"), handle_metadata.function_name))
    rows.append(None)
    rows.append((Text("Object ID:", style="bold"), server_id))
    rows.append((Text("App ID:", style="bold"), handle_metadata.app_id))
    rows.append((Text("Image ID:", style="bold"), str(info.image_info.image_id)))

    # --- HTTP ---
    rows.append(None)
    rows.append("HTTP Info:")

    url = Text(info.server_url, overflow="fold")
    url.stylize(style=Style(link=info.server_url), end=len(info.server_url))

    rows.append(("  URL:", url))
    rows.append(("  Port:", not_configured if not info.port else str(info.port)))
    rows.append(("  Authentication:", disabled if info.unauthenticated else enabled))
    rows.append(("  HTTP/2:", disabled if not info.h2_enabled else enabled))
    rows.append(("  Routing Region:", not_configured if not info.routing_region else info.routing_region))
    rows.append(("  Sessioned:", not_configured if not info.sessioned else enabled))
    rows.append(("  Startup Timeout:", f"{info.startup_timeout} seconds"))
    rows.append(("  Exit Grace Period:", f"{info.exit_grace_period} seconds"))

    # --- Resources ---
    rows.append(None)
    rows.append("Resources:")
    rows.append(
        (
            "  CPU:",
            not_configured
            if info.cpu is None
            else f"{info.cpu} core(s)"
            if isinstance(info.cpu, (int, float))
            else f"{info.cpu[0]} - {info.cpu[1]} core(s)",
        )
    )
    rows.append(
        (
            "  Memory:",
            not_configured
            if info.memory_mib is None
            else humanize_filesize(info.memory_mib << 20)
            if isinstance(info.memory_mib, int)
            else f"{humanize_filesize(info.memory_mib[0] << 20)} - {humanize_filesize(info.memory_mib[1] << 20)}",
        )
    )
    rows.append(
        (
            "  Ephemeral Disk:",
            not_configured if info.ephemeral_disk_mib is None else (humanize_filesize(info.ephemeral_disk_mib << 20)),
        )
    )
    rows.append(
        (
            "  GPU(s):",
            not_configured
            if len(info.gpus) == 0
            else " | ".join([f"{gpu_type} x {count}" for gpu_type, count in info.gpus]),
        )
    )

    # --- Autoscaling ---
    rows.append(None)
    rows.append("Autoscaling:")
    rows.append(
        (
            "  Min/Max/Buffer Containers:",
            " / ".join(
                [
                    not_configured
                    if autoscaler_settings.min_containers is None
                    else str(autoscaler_settings.min_containers),
                    not_configured
                    if autoscaler_settings.max_containers is None
                    else str(autoscaler_settings.max_containers),
                    not_configured
                    if autoscaler_settings.buffer_containers is None
                    else str(autoscaler_settings.buffer_containers),
                ]
            ),
        )
    )

    rows.append(
        (
            "  Scaleup Window:",
            not_configured
            if autoscaler_settings.scaleup_window is None
            else f"{autoscaler_settings.scaleup_window} seconds",
        )
    )
    rows.append(
        (
            "  Scaledown Window:",
            not_configured
            if autoscaler_settings.scaledown_window is None
            else f"{autoscaler_settings.scaledown_window} seconds",
        )
    )
    rows.append(
        (
            "  Target Concurrency:",
            not_configured
            if not autoscaler_settings.target_concurrency
            else f"{autoscaler_settings.target_concurrency:.1f} requests / container",
        )
    )

    # --- Scheduling ---
    rows.append(None)
    rows.append("Scheduling:")
    rows.append(
        ("  Compute Region(s):", not_configured if not info.compute_regions else " | ".join(info.compute_regions))
    )
    rows.append(("  Nonpreemptible Capacity:", not_configured if not info.nonpreemptible else enabled))
    if info.cloud:
        rows.append(("  Cloud Provider:", info.cloud))

    if info.cluster_info:
        rows.append(None)
        rows.append("Clustering:")
        rows.append(("  Cluster Size:", str(info.cluster_info.size)))
        rows.append(("  RDMA:", enabled if info.cluster_info.rdma else disabled))
        if info.cluster_info.fabric_size is not None:
            rows.append(("  Fabric Size:", str(info.cluster_info.fabric_size)))

    # --- Mounts ---
    if info.volumes:
        rows.append(None)
        rows.append("Volume Mounts:")
        for mount_path, volume in info.volumes.items():
            assert volume.volume_id is not None

            extra = []
            if volume.read_only:
                extra.append("Read-Only")
            if volume.sub_path:
                extra.append(f"Sub-Path: {volume.sub_path}")

            if extra:
                volume_text = f"{volume.volume_id} ({', '.join(extra)})"
            else:
                volume_text = volume.volume_id

            rows.append((f"  {mount_path}", volume_text))

    if info.cloud_bucket_mounts:
        rows.append(None)
        rows.append("Cloud Bucket Mounts:")
        for mount_path, cbm in info.cloud_bucket_mounts.items():
            extra = []
            if cbm.read_only:
                extra.append("Read-Only")
            if cbm.key_prefix:
                extra.append(f"Prefix: {cbm.key_prefix}")

            if extra:
                cbm_text = f"{cbm.bucket_name} ({', '.join(extra)})"
            else:
                cbm_text = cbm.bucket_name

            rows.append((f"  {mount_path}", cbm_text))

    # --- Secrets ---
    if info.secrets:
        rows.append(None)
        rows.append("Secrets:")

        async def _hydrate(secret_id: str) -> tuple[str, _Secret | None]:
            try:
                s = _Secret._from_id(secret_id)
                await s.hydrate()
            except NotFoundError:
                return secret_id, None

            return secret_id, s

        async def _secret_iter() -> AsyncGenerator[str, None]:
            for secret_id in info.secrets:
                yield secret_id

        async for secret_id, secret in async_map_ordered(
            _secret_iter(),
            _hydrate,
            10,
        ):
            if secret is None:
                rows.append((f"  {secret_id}", "[DELETED]"))
                continue

            secret_identifier = secret.object_id
            if secret._name:
                secret_identifier = secret._name
                secret_env = secret._get_metadata().environment_name

                if secret_env != environment_name:
                    secret_identifier = f"{secret._name} ({secret_env})"

            keys = await secret._get_keys()
            display_keys = sorted(keys)[:5]
            if len(display_keys) < len(keys):
                display_keys.append(f"({len(keys) - len(display_keys)} keys omitted)")

            rows.append((f"  {secret_identifier}", ", ".join(display_keys)))

    t = Table(box=SIMPLE_HEAD, show_header=False)
    for row in rows:
        if row is None:
            t.add_row()
            continue

        if not isinstance(row, tuple):
            if isinstance(row, str):
                row = Text(row, style="bold")

            if not tty:
                output_manager.print(row)
            else:
                t.add_row(row)

            continue

        left, right = row

        if not tty:
            output_manager.print(Text(f"{left} {right}"))
            continue

        if isinstance(left, str):
            left = Text(left, overflow="fold")
        if isinstance(right, str):
            right = Text(right, overflow="fold")

        t.add_row(left, right)

    if tty:
        output_manager.print(t)


@server_cli.command("logs", no_args_is_help=True)
@click.argument("server_ref")
@click.option("-f", "--follow", is_flag=True, default=False, help="Stream log output until interrupted")
@click.option(
    "--since",
    default=None,
    help="Start of time range. Accepts ISO 8601 datetime or relative time, e.g. '1d' (1 day ago), '2h', '30m', etc.",
)
@click.option("--until", default=None, help="End of time range; accepts same argument types as --since")
@click.option("-n", "--tail", default=None, type=int, help="Show only the last N log entries")
@click.option("--search", default=None, help="Filter by search text")
@click.option("--container", "container_id", default="", help="Filter by Container ID (ta-*)")
@click.option("-s", "--source", default=None, help="Filter by source: 'stdout', 'stderr', or 'system'")
@click.option("--timestamps", is_flag=True, default=False, help="Prefix each line with its timestamp")
@click.option("--show-server-id", is_flag=True, default=False, help="Prefix each line with its Server ID")
@click.option("--show-container-id", is_flag=True, default=False, help="Prefix each line with its Container ID")
@env_option
@synchronizer.create_blocking
async def logs(
    server_ref: str,
    follow: bool = False,
    since: str | None = None,
    until: str | None = None,
    tail: int | None = None,
    search: str | None = None,
    container_id: str = "",
    source: str | None = None,
    timestamps: bool = False,
    show_server_id: bool = False,
    show_container_id: bool = False,
    *,
    env: str | None = None,
) -> None:
    """Fetch or stream Server logs.

    By default, this command fetches the last 100 log entries and exits. Use ``-f`` to
    live-stream logs from a running server instead. Fetch and follow are mutually exclusive.

    Examples:

    Get recent logs based on a server ID:

    ```
    modal server logs fu-12345
    ```

    Get recent logs for a currently deployed server based on its name:

    ```
    modal server logs my-app/qwen-server
    ```

    Follow (stream) logs from a running server:

    ```
    modal server logs my-app/qwen-server -f
    ```

    Fetch the last 1000 entries:

    ```
    modal server logs my-app/qwen-server --tail 1000
    ```

    Fetch logs from the last 2 hours:

    ```
    modal server logs my-app/qwen-server --since 2h
    ```

    Fetch logs in a specific time range:

    ```
    modal server logs my-app/qwen-server --since 2026-09-01T05:00:00 --until 2026-09-01T08:00:00
    ```

    Filter the logs by source:

    ```
    modal server logs my-app/qwen-server --source stderr
    ```

    Include timestamps along with server and container IDs on each line:

    ```
    modal server logs my-app/qwen-server --timestamps --show-server-id --show-container-id
    ```
    """
    env = ensure_env(env)
    _validate_logs_args(follow=follow, since=since, until=until, tail=tail)
    client = await _Client.from_env()

    function_id, metadata, _ = await _resolve_function_id(client, server_ref, env, object_type="Server", command="logs")

    await _run_server_logs(
        metadata.app_id,
        function_id,
        follow=follow,
        since=since,
        until=until,
        tail=tail,
        search=search,
        container_id=container_id,
        source=source,
        timestamps=timestamps,
        show_server_id=show_server_id,
        show_container_id=show_container_id,
    )


_REQUEST_METRIC_ORDER: tuple[str, ...] = ()
_CONTAINER_METRIC_ORDER = (
    "Startup time (s)",
    "CPU usage (cores)",
    "Memory usage (GiB)",
    "GPU utilization (%)",
)
# display name, scale, precision
_INFERENCE_METRIC_FORMATS = {
    "time_to_first_token": ("Time to first token (ms)", 1000, 0),
    "inter_token_latency": ("Inter-token latency (ms)", 1000, 0),
    "end_to_end_latency": ("End-to-end latency (s)", 1, 2),
}
_INFERENCE_METRIC_ORDER = tuple(_INFERENCE_METRIC_FORMATS)
_DISPLAYED_PERCENTILES = ((5000, "p50"), (9000, "p90"), (9900, "p99"))


def _enum_value_name(value: int, enum_wrapper, prefix: str) -> str:
    # Preserve unknown values so older clients can serialize enums added by newer servers.
    try:
        name = enum_wrapper.Name(value)
    except ValueError:
        return "unrecognized"
    return name.removeprefix(prefix).lower()


def _inference_json(inference: api_pb2.ServerGetTimeRangeStatsResponse.ServerInferenceStats) -> dict[str, object]:
    return {
        "engine": _enum_value_name(inference.engine, api_pb2.LLMEngine, "LLM_ENGINE_"),
        "status": _enum_value_name(
            inference.status,
            api_pb2.ServerInferenceStatsStatus,
            "SERVER_INFERENCE_STATS_STATUS_",
        ),
        "percentile_stats": _distribution_map_json(
            cast(dict[str, api_pb2.StatsPercentileDistribution], inference.percentile_stats)
        ),
        "scalar_stats": dict(inference.scalar_stats),
    }


def _stats_json(
    function_id: str, history: api_pb2.ServerGetTimeRangeStatsResponse, endpoint_id: str | None = None
) -> dict[str, object]:
    id_dict = {"object_id": function_id} if not endpoint_id else {"endpoint_id": endpoint_id}
    return {
        **id_dict,
        "since": history.since.ToDatetime(tzinfo=timezone.utc).isoformat(),
        "until": history.until.ToDatetime(tzinfo=timezone.utc).isoformat(),
        "request_count": history.request_count,
        "request_count_by_status_code": {
            f"{status_count.status_code // 100}xx": status_count.count
            for status_count in history.request_count_by_status_code
        },
        "request_rate_per_second": history.request_rate_per_second,
        "container_started_count": history.container_started_count,
        "container_error_count": history.container_error_count,
        "container_total_count": history.container_total_count,
        "container_creating_at_end_count": history.container_creating_at_end_count,
        "request_percentile_stats": _distribution_map_json(
            cast(dict[str, api_pb2.StatsPercentileDistribution], history.request_percentile_stats)
        ),
        "container_percentile_stats": _distribution_map_json(
            cast(dict[str, api_pb2.StatsPercentileDistribution], history.container_percentile_stats)
        ),
        "inference": _inference_json(history.inference) if history.HasField("inference") else None,
    }


def _inference_engine_label(engine: int) -> str:
    if engine == api_pb2.LLM_ENGINE_SGLANG:
        return "SGLang"
    if engine == api_pb2.LLM_ENGINE_VLLM:
        return "vLLM"
    return "Unknown engine"


def _inference_metric_rows(
    distributions: dict[str, api_pb2.StatsPercentileDistribution],
    use_color: bool,
) -> list[tuple[Text, Text, Text, Text]]:
    rows: list[tuple[Text, Text, Text, Text]] = []
    ordered_names = [name for name in _INFERENCE_METRIC_ORDER if name in distributions]
    ordered_names.extend(sorted(set(distributions) - set(_INFERENCE_METRIC_ORDER)))
    for name in ordered_names:
        distribution = distributions[name]
        display_name, scale, precision = _INFERENCE_METRIC_FORMATS.get(name, (name, 1, 2))
        values_by_percentile = {
            percentile.percentile_basis_points: percentile.value for percentile in distribution.percentiles
        }
        if not any(basis_points in values_by_percentile for basis_points, _ in _DISPLAYED_PERCENTILES):
            continue
        values = [
            (
                f"{values_by_percentile[basis_points] * scale:.{precision}f}"
                if basis_points in values_by_percentile
                else "—"
            )
            for basis_points, _ in _DISPLAYED_PERCENTILES
        ]
        p50 = values_by_percentile.get(5000)
        p99 = values_by_percentile.get(9900)
        rows.append(
            (
                Text(f"  {display_name}", style=stats_style(STATS_METADATA_STYLE, use_color)),
                Text(values[0]),
                Text(values[1]),
                Text(values[2], style=percentile_style(p50, p99, use_color)),
            )
        )
    return rows


def _inference_throughput(inference: api_pb2.ServerGetTimeRangeStatsResponse.ServerInferenceStats) -> str:
    throughput = []
    if "input_tokens_per_second" in inference.scalar_stats:
        throughput.append(f"Input {inference.scalar_stats['input_tokens_per_second']:,.0f} tok/s")
    if "cached_input_tokens_per_second" in inference.scalar_stats:
        throughput.append(f"Cached input {inference.scalar_stats['cached_input_tokens_per_second']:,.0f} tok/s")
    if "output_tokens_per_second" in inference.scalar_stats:
        throughput.append(f"Output {inference.scalar_stats['output_tokens_per_second']:,.0f} tok/s")
    return " · ".join(throughput)


def _render_inference(history: api_pb2.ServerGetTimeRangeStatsResponse, use_color: bool) -> None:
    if not history.HasField("inference"):
        return

    output = OutputManager.get()
    inference = history.inference
    engine_label = _inference_engine_label(inference.engine)
    output.print("")
    heading = f"Inference · {engine_label}"
    output.print(Text(heading, style=stats_style(STATS_SECTION_STYLE, use_color)))
    throughput = _inference_throughput(inference)
    if throughput:
        output.print(Text(f"  {throughput}", style=stats_style(STATS_METADATA_STYLE, use_color)))

    if inference.status == api_pb2.SERVER_INFERENCE_STATS_STATUS_NO_DATA:
        output.print(
            Text(
                "  No inference metrics recorded for this range.",
                style=stats_style(STATS_METADATA_STYLE, use_color),
            )
        )
        return
    if inference.status == api_pb2.SERVER_INFERENCE_STATS_STATUS_UNAVAILABLE:
        output.print(
            Text(
                "  Inference metrics are temporarily unavailable.",
                style=stats_style(STATS_PROBLEM_STYLE, use_color),
            )
        )
        return

    percentile_rows = _inference_metric_rows(
        cast(dict[str, api_pb2.StatsPercentileDistribution], inference.percentile_stats),
        use_color,
    )
    if percentile_rows:
        output.print("")
        output.print(_percentile_table(percentile_rows, use_color))


def _server_stats_time_range(since: str | None, until: str | None) -> tuple[datetime, datetime, timedelta]:
    now = datetime.now(timezone.utc)
    until_dt = _parse_time_arg(until, default=now, now=now)
    since_dt = _parse_time_arg(since, default=until_dt - _DEFAULT_STATS_WINDOW, now=now)
    if since_dt >= until_dt:
        raise UsageError("--since must be before --until.")

    history_duration = until_dt - since_dt
    if since is not None and until is None:
        try:
            history_duration = parse_duration(since)
        except ValueError:
            pass
    return since_dt, until_dt, history_duration


async def _run_server_stats(
    client: _Client,
    function_id: str,
    *,
    since_dt: datetime,
    until_dt: datetime,
    history_duration: timedelta,
    container_id: str | None = None,
    no_color: bool = False,
    json_output: bool = False,
    heading: str | None = None,
    endpoint_id: str | None = None,
) -> None:
    req = api_pb2.ServerGetTimeRangeStatsRequest(
        function_id=function_id,
        since=_timestamp(since_dt),
        until=_timestamp(until_dt),
    )
    if container_id:
        req.container_id = container_id

    history = await client._stub.ServerGetTimeRangeStats(req)

    if json_output:
        OutputManager.get().print_json(json_lib.dumps(_stats_json(function_id, history, endpoint_id)))
        return

    output = OutputManager.get()
    output.print("")
    use_color = not no_color
    output.print(Text(heading or f"Server stats for {function_id}", style=stats_style(STATS_HEADING_STYLE, use_color)))
    output.print("")
    since_label = history.since.ToDatetime(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    until_label = history.until.ToDatetime(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    range_heading = f"Duration {str(history_duration)} · "
    output.print(
        Text(
            f"{range_heading}{since_label} to {until_label} UTC",
            style=stats_style(STATS_METADATA_STYLE, use_color),
        )
    )
    output.print("")

    request_heading = Text("Requests", style=stats_style(STATS_SECTION_STYLE, use_color))
    request_heading.append(" " * (46 - len("Requests")))
    request_heading.append(f"{history.request_count:,} total ({history.request_rate_per_second:,.2f} req/s)")
    output.print(request_heading)
    if history.request_count_by_status_code:
        status_counts = Text("  ")
        for index, item in enumerate(sorted(history.request_count_by_status_code, key=lambda item: item.status_code)):
            if index:
                status_counts.append(" · ")
            status_family = item.status_code // 100
            style = None
            if status_family == 2:
                style = success_style(item.count, history.request_count, use_color)
            elif status_family in (4, 5):
                style = problem_style(item.count, history.request_count, use_color)
            status_counts.append(f"{status_family}xx: {item.count:,}", style=style)
        output.print(status_counts)

    request_rows = _metric_rows(
        cast(dict[str, api_pb2.StatsPercentileDistribution], history.request_percentile_stats),
        _REQUEST_METRIC_ORDER,
        use_color,
    )
    if request_rows:
        output.print("")
        output.print(_percentile_table(request_rows, use_color))

    output.print("")
    container_count = history.container_started_count + history.container_error_count
    container_total_count = history.container_total_count
    container_heading = Text("Containers", style=stats_style(STATS_SECTION_STYLE, use_color))
    container_heading.append(" " * (46 - len("Containers")))
    container_heading.append(f"{container_total_count:,} Total (")
    container_heading.append(
        f"{history.container_creating_at_end_count:,} creating",
        style=progress_style(history.container_creating_at_end_count, use_color),
    )
    container_heading.append(")")
    output.print(container_heading)

    container_counts = Text("  ")
    container_counts.append(
        f"{history.container_started_count:,} started",
        style=success_style(history.container_started_count, container_count, use_color),
    )
    container_counts.append(" · ")
    container_counts.append(
        f"{history.container_error_count:,} errored",
        style=problem_style(history.container_error_count, container_count, use_color),
    )
    output.print(container_counts)
    container_rows = _metric_rows(
        cast(dict[str, api_pb2.StatsPercentileDistribution], history.container_percentile_stats),
        _CONTAINER_METRIC_ORDER,
        use_color,
    )
    if container_rows:
        output.print("")
        output.print(_percentile_table(container_rows, use_color))

    _render_inference(history, use_color)


@server_cli.command("stats", no_args_is_help=True)
@click.argument("server_identifier", metavar="SERVER")
@click.option(
    "--since",
    default=None,
    help=(
        "Start of time range. Treated as local time "
        "if a timezone is not supplied. "
        "Accepts an ISO 8601 datetime or relative time such as '2h' or '30m'."
    ),
)
@click.option(
    "--until",
    default=None,
    help=(
        "End of time range. Treated as local time if a timezone "
        "is not supplied. Accepts the same argument types as --since."
    ),
)
@click.option(
    "--container-id",
    type=str,
    default=None,
    metavar="CONTAINER_ID",
    help="Compute the stats only for this container.",
)
@click.option("--no-color", is_flag=True, default=False, help="Disable colors in the output.")
@click.option("--json", "json_output", is_flag=True, default=False, help="Output stats as JSON.")
@env_option
@synchronizer.create_blocking
async def stats(
    server_identifier: str,
    since: str | None = None,
    until: str | None = None,
    container_id: str | None = None,
    no_color: bool = False,
    json_output: bool = False,
    *,
    env: str | None = None,
):
    """Show aggregate statistics for a Modal Server.

    SERVER may be a Function ID or a deployed Server name in the form
    ``APP_NAME/SERVER_NAME``. The default time range is the most recent hour.

    The available metrics and their definitions are subject to change and are provided
    on a best-effort basis. They may be delayed or incomplete. Do not rely on this command
    for autoscaler management.

    Examples:

    Show stats for the last hour:

    ```
    modal server stats fu-abc123
    ```

    Show stats for relative window:

    ```
    modal server stats my-app/my-server --since 5h --until 2h
    ```

    Show stats for one hour window ending two hours ago:

    ```
    modal server stats my-app/my-server --until 2h
    ```

    Show stats for explicit window in json format:
    ```
    modal server stats my-app/my-server \
        --since 2026-08-28T14:00:00Z \
        --until 2026-08-28T16:00:00Z \
        --json
    ```

    Show stats for a specific container:

    ```
    modal server stats my-app/my-server --container-id ta-12345
    ```
    """
    since_dt, until_dt, history_duration = _server_stats_time_range(since, until)

    environment_name = _get_environment_name(ensure_env(env))
    client = await _Client.from_env()
    function_id, _, _ = await _resolve_function_id(
        client,
        server_identifier,
        environment_name,
        object_type="Server",
        command="stats",
    )
    await _run_server_stats(
        client,
        function_id,
        since_dt=since_dt,
        until_dt=until_dt,
        history_duration=history_duration,
        container_id=container_id,
        no_color=no_color,
        json_output=json_output,
    )


@server_cli.command("requests", no_args_is_help=True)
@click.argument("server_identifier", metavar="SERVER")
@click.option(
    "-n",
    "--tail",
    type=click.IntRange(min=1, max=_MAX_REQUEST_TAIL),
    default=None,
    help="Show up to the last N Server requests. Defaults to 10. Cannot be combined with --since or --until.",
)
@click.option("--since", help="Fetch requests at or after this time (e.g. '2h' or an ISO 8601 datetime).")
@click.option("--until", help="Fetch requests before this time (e.g. '1h' or an ISO 8601 datetime).")
@click.option("--json", "json_output", is_flag=True, default=False, help="Output requests as JSON.")
@click.option("--no-color", "no_color", is_flag=True, default=False, help="Disable colors in the output.")
@env_option
@synchronizer.create_blocking
async def requests(
    server_identifier: str,
    tail: int | None = None,
    json_output: bool = False,
    no_color: bool = False,
    *,
    env: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> None:
    """Show recent requests handled by a Modal Server.

    SERVER may be a Function ID or a deployed Server name in the form
    ``APP_NAME/SERVER_NAME``.

    Use ``--since`` and/or ``--until`` to fetch all inputs in the time range,
    oldest first. ``--until`` defaults to now. If ``--since`` is omitted, the preceeding hour
    is fetched. If timezones are omitted, the time is interpreted as local time.

    ``--tail`` cannot be combined with either date option.
    By default, ``--tail`` returns the latest 10 inputs, newest first.

    Examples:

    ```
    modal server requests my-app/my-server
    ```

    ```
    modal server requests my-app/my-server --tail 500
    ```

    ```
    modal server requests my-app/my-server --since 2h --until 1h
    ```

    Disable color in the output:

    ```
    modal server requests my-app/my-server --no-color
    ```
    """
    fetch_range = None
    if since is not None or until is not None:
        if tail is not None:
            raise UsageError("--tail cannot be combined with --since or --until.")
        fetch_range = _parse_fetch_range(since, until)
    environment_name = _get_environment_name(ensure_env(env))
    client = await _Client.from_env()
    function_id, _, _ = await _resolve_function_id(
        client,
        server_identifier,
        environment_name,
        object_type="Server",
        command="requests",
    )
    if fetch_range is None:
        response = await client._stub.ServerRequestFetch(
            api_pb2.ServerRequestFetchRequest(
                function_id=function_id,
                tail=api_pb2.ServerRequestFetchRequest.Tail(count=tail if tail is not None else _DEFAULT_REQUEST_TAIL),
            )
        )
        request_infos: Sequence[api_pb2.ServerRequestInfo] = response.requests
    else:
        since_dt, until_dt = fetch_range

        async def count_ranges(start: float, end: float, bucket_secs: int) -> list[tuple[float, float, int]]:
            counted = await client._stub.ServerCountRequests(
                api_pb2.ServerCountRequestsRequest(
                    function_id=function_id,
                    since=_seconds_to_timestamp(start),
                    until=_seconds_to_timestamp(end),
                    bucket_secs=bucket_secs,
                )
            )
            if sum(bucket.count for bucket in counted.buckets) > _MAX_REQUEST_FETCH:
                raise UsageError(
                    f"Cannot fetch more than {_MAX_REQUEST_FETCH:,} requests. Consider narrowing the range."
                )
            return [
                (
                    _timestamp_to_seconds(b.bucket_start_at),
                    _timestamp_to_seconds(b.bucket_start_at) + bucket_secs,
                    b.count,
                )
                for b in counted.buckets
            ]

        async def fetch_interval(start: float, end: float, limit: int) -> Sequence[api_pb2.ServerRequestInfo]:
            fetched = await client._stub.ServerRequestFetch(
                api_pb2.ServerRequestFetchRequest(
                    function_id=function_id,
                    range=api_pb2.ServerRequestFetchRequest.Range(
                        since=_seconds_to_timestamp(start), until=_seconds_to_timestamp(end), limit=limit
                    ),
                )
            )
            return fetched.requests

        try:
            request_infos = [
                item
                async for item in _fetch_time_range(
                    since_dt,
                    until_dt,
                    count_ranges,
                    fetch_interval,
                    max_concurrent_counts=10,
                )
            ]
        except LogsFetchError:
            raise UsageError("Too many requests to fetch in time range. Consider narrowing the range.")

    if json_output:
        OutputManager.get().print_json(
            json_lib.dumps(
                [
                    {
                        "timestamp": request.timestamp.ToJsonString(),
                        "route": request.route,
                        "container_id": request.container_id,
                        "duration_seconds": request.duration_seconds,
                        "status": request.status,
                    }
                    for request in request_infos
                ]
            )
        )
        return

    rows: list[list[Text | str]] = []
    previous_request_date = None
    for request in request_infos:
        timestamp, previous_request_date = grouped_utc_timestamp(request.timestamp, previous_request_date)
        rows.append(
            [
                Text(timestamp),
                Text(request.route),
                Text(request.container_id),
                Text(f"{request.duration_seconds:.2f}"),
                _server_request_status_cell(request.status, no_color=no_color),
            ]
        )
    output = OutputManager.get()
    output.print("")
    title = f"Server requests for {function_id}"
    if fetch_range is not None:
        since_dt, until_dt = fetch_range
        since_label = since_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        until_label = until_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        title += f" · fetch [{since_label}, {until_label}) UTC"
    output.print(Text(title))
    output.print(
        Text(
            f"Displaying {len(rows):,} {'row' if len(rows) == 1 else 'rows'}",
            style=STATS_METADATA_STYLE if not no_color else "",
        )
    )
    output.print("")
    display_table(
        [
            Column("Timestamp (UTC)", justify="right"),
            "Route",
            Column("Container", width=29),
            Column("Duration (s)", justify="right"),
            "Status",
        ],
        rows,
        table_box=HEADER_ONLY,
        border_style=STATS_METADATA_STYLE if not no_color else None,
    )


autoscaler_cli = ModalGroup(name="autoscaler", help="View and Configure Autoscaler settings for Modal Servers.")
server_cli.add_command(autoscaler_cli)


@autoscaler_cli.command("settings", no_args_is_help=True)
@click.argument("server_identifier", metavar="FUNCTION")
@click.option("--json", "json", is_flag=True, default=False, help="Output as JSON.")
@env_option
@synchronizer.create_blocking
async def autoscaler_info(server_identifier: str, json: bool = False, env: str | None = None):
    """Show the current Autsocaling settings for the provided Server.

    In addition to the current settings, this command also shows any active overrides to these
    settings, along with metadata about who overrode the setting, and when. SERVER may be a
    Server ID or a deployed Server name in the form ``APP_NAME/SERVER_NAME``.

    If a setting's value is the default and not overridden, its value is omitted in the output.

    Examples:

    Show autoscaling settings for a deployed Server:

    ```
    modal server autoscaler settings my-app/my-server
    ```

    Show autoscaling settings using an ID:

    ```
    modal server autoscaler settings fu-1234
    ```
    """

    environment_name = _get_environment_name(ensure_env(env))
    client = await _Client.from_env()
    function_id, _, _ = await _resolve_function_id(
        client,
        server_identifier,
        environment_name,
        object_type="Server",
        command="autoscaler settings",
    )

    autoscaler_response = await client._stub.FunctionGetSchedulingParams(
        api_pb2.FunctionGetSchedulingParamsRequest(function_id=function_id)
    )

    autoscaler_settings = ServerAutoscalerSettings._from_proto(autoscaler_response.autoscaler_configuration.settings)
    override_events = autoscaler_response.autoscaler_configuration.override_events

    output_manager = OutputManager.get()

    if json:
        current_settings = dataclasses.asdict(autoscaler_settings)
        if current_settings["target_concurrency"] == 0:
            current_settings["target_concurrency"] = None

        override_events_json: list[dict[str, str]] = []
        for k, v in override_events.items():
            if k == "target_concurrency_float":
                continue

            if v.user_id:
                obj = {
                    "setting": k,
                    "updated_at": str(v.timestamp),
                    "actor_name": v.requested_by,
                    "actor_id": v.user_id,
                    "actor_type": "user",
                }
            elif v.service_user_id:
                obj = {
                    "setting": k,
                    "updated_at": str(v.timestamp),
                    "actor_name": v.requested_by,
                    "actor_id": v.service_user_id,
                    "actor_type": "service-user",
                }
            else:
                obj = {
                    "setting": k,
                    "updated_at": str(v.timestamp),
                    "actor_id": "unknown",
                    "actor_name": "unknown",
                    "actor_type": "unknown",
                }

            override_events_json.append(obj)

        output_manager.print_json(
            json_lib.dumps(
                {
                    "current_settings": current_settings,
                    "override_events": override_events_json,
                }
            )
        )

        return

    human_readable = {
        "min_containers": "Min Containers",
        "max_containers": "Max Containers",
        "buffer_containers": "Buffer Containers",
        "scaledown_window": "Scaledown Window",
        "scaleup_window": "Scaleup Window",
        "target_concurrency": "Target Concurrency",
    }

    if override_events:
        t = Table("Setting", "Value", "Overridden By", box=SIMPLE)
    else:
        t = Table("Setting", "Value", box=SIMPLE)

    for key, pretty in human_readable.items():
        value = getattr(autoscaler_settings, key)
        value_str = str(value)

        override_meta = "-"
        if key in override_events:
            override = override_events[key]
            override_meta = f"{override.requested_by} at {timestamp_to_localized_str(override.timestamp, isotz=False)}"

        if key in {"scaledown_window", "scaleup_window"}:
            value_str = f"{value} seconds"

        if key == "target_concurrency":
            value_str = f"{value} requests/container"

        if not value and (key == "target_concurrency" or key not in override_events):
            value_str = "-"

        if override_events:
            t.add_row(pretty, value_str, override_meta)
        else:
            t.add_row(pretty, value_str)

    output_manager.print(t)


@autoscaler_cli.command("update", no_args_is_help=True)
@click.argument("server_identifier", metavar="SERVER")
@click.option("--max-containers", "max_containers", default=None, help="Maximum concurrent containers.")
@click.option("--min-containers", "min_containers", default=None, help="Minimum number of containers to keep running.")
@click.option(
    "--buffer-containers",
    "buffer_containers",
    default=None,
    help="Extra containers to keep warm beyond current demand.",
)
@click.option(
    "--scaledown-window",
    "scaledown_window",
    default=None,
    help="Maximum duration (in seconds) idle containers wait before scaling down.",
)
@click.option(
    "--scaleup-window",
    "scaleup_window",
    default=None,
    help="Seconds of sustained demand required before scaling up new containers. ",
)
@click.option(
    "--target-concurrency",
    "target_concurrency",
    default=None,
    help="Target number of concurrent requests per container. May be fractional.",
)
@env_option
@synchronizer.create_blocking
async def autoscaler_update(
    server_identifier: str,
    max_containers: str | None = None,
    min_containers: str | None = None,
    buffer_containers: str | None = None,
    scaledown_window: str | None = None,
    scaleup_window: str | None = None,
    target_concurrency: str | None = None,
    env: str | None = None,
):
    """Update the current Autsocaling settings for the provided Server.

    SERVER may be a Server ID or a deployed Server name in the form ``APP_NAME/SERVER_NAME``.

    Settings can be modified using the `--max-containers`, `--min-containers`, `--buffer-containers`,
    `--scaledown-window`, `--scaleup-window`, and `--target-concurrency` options.

    If provided, Max Containers should be a positive integer, Min and Buffer Containers should be
    nonnegative integers, Scaledown and Scaleup windows should be positive integers measured in seconds,
    and Target Concurrency should either be a positive floating point number, or the string "none".

    Examples:

    Update Max Containers for a deployed Server:

    ```
    modal server autoscaler update my-app/my-server --max-containers 10
    ```

    Update both Min Containers and Target Concurrency for a Server by ID:

    ```
    modal server autoscaler update fu-1234 --min-containers 3 --target-concurrency 2.5
    ```

    Set Target Concurrency to "none" (disabling autoscaling for the Server)

    ```
    modal server autoscaler update my-app/my-server --target-concurrency none
    ```
    """

    environment_name = _get_environment_name(ensure_env(env))
    client = await _Client.from_env()

    updates: dict[str, int | float] = {}
    for k, v in {
        "min_containers": min_containers,
        "max_containers": max_containers,
        "buffer_containers": buffer_containers,
        "scaledown_window": scaledown_window,
        "scaleup_window": scaleup_window,
        "target_concurrency": target_concurrency,
    }.items():
        if v is None:
            continue

        if k == "target_concurrency":
            error_message = (
                f"Invalid input for '--target-concurrency': {v!r} must be a positive float or the string \"none\""
            )

            if v == "none":
                updates[k] = None  # type: ignore
                continue

            try:
                updates[k] = float(v)
            except ValueError as e:
                raise click.UsageError(error_message) from e

            if updates[k] <= 0:
                raise click.UsageError(error_message)

            continue

        try:
            updates[k] = parse_int(
                v,
                allow_negative=False,
                allow_zero=k in {"min_containers", "buffer_containers"},
            )
        except ValueError as e:
            option_name = f"--{k.replace('_', '-')}"
            raise click.UsageError(f"Invalid input for {option_name!r}: {e}")

    output_manager = OutputManager.get()
    if not updates:
        output_manager.print("No autoscaler settings specified.")
        return

    function_id, handle_metadata, _ = await _resolve_function_id(
        client,
        server_identifier,
        environment_name,
        object_type="Server",
        command="autoscaler update",
    )

    s = _Server._new_from_function(function_id, client, handle_metadata)
    await s.autoscaler.update(**updates)  # type: ignore

    output_manager.print(f"[green]✓[/green] Updated autoscaler settings for Server {handle_metadata.function_name}")


@server_cli.command("metrics", no_args_is_help=True)
@click.argument("identifier", metavar="SERVER")
@click.option("--since", help="Start time: ISO 8601 or relative duration (e.g. 2h). Naive times are local.")
@click.option("--until", help="End time; same syntax as --since. Defaults to now.")
@click.option(
    "--group",
    "groups",
    multiple=True,
    type=str,
    help="Metric group; repeat to select multiple. Defaults to all. Valid values are "
    "'containers', 'cpu', 'memory', 'network', 'gpu', 'requests'.",
)
@click.option(
    "--bucket-size",
    type=str,
    metavar="DURATION",
    help=(
        "Bucket width: 1m, 2m, 3m, 4m, 5m, 6m, 10m, 12m, 15m, 20m, 30m, "
        "1h, 2h, 3h, 4h, 8h, 12h, 1d. By default, a value is chosen that splits the time range "
        "into approximately 100 buckets."
    ),
)
@click.option("--json", "json_output", is_flag=True, help="Fetch metric samples and output JSON.")
@click.option("--csv", "csv_output", is_flag=True, help="Fetch metric samples and output CSV.")
@env_option
@synchronizer.create_blocking
async def metrics(
    identifier: str,
    since: str | None,
    until: str | None,
    groups: tuple[str, ...],
    bucket_size: str | None,
    json_output: bool,
    csv_output: bool,
    env: str | None,
) -> None:
    """Preview and export metrics for a Modal Server.

    SERVER may be an Object ID or a deployed Server name in the form
    ``APP_NAME/SERVER_NAME``.

    Each metric is a time series over the window from ``--since`` to ``--until``
    (default: the last hour, maximum: 31 days). Samples are aggregated into
    fixed-width buckets.

    Set the width with ``--bucket-size``. Each series is limited to 500 buckets, so long windows
    need wider buckets. Time bounds are rounded down to bucket boundaries in UTC.
    Results may include data before --since and exclude data between the last boundary and --until.

    Series are grouped by name. Specify which metrics to return with ``--group``.

    By default, prints a schema summarizing the date range, bucket size, and time series that would
    be returned. Use ``--json`` or ``--csv`` to output the raw samples instead.

    Preview available Server metrics:

    ```shell
    modal server metrics my-app/my-server
    ```

    Export requests metrics for the last two hours as JSON:

    ```shell
    modal server metrics my-app/my-server --group requests --since 2h --json > metrics.json
    ```

    Export CPU, memory, network, and GPU metrics:

    ```shell
    modal server metrics my-app/my-server --group cpu --group memory --group network --group gpu --json
    ```

    Save container metrics with five-minute buckets as CSV:

    ```shell
    modal server metrics fu-abc123 --group containers --bucket-size 5m --csv > metrics.csv
    ```
    """
    await _export_metrics(
        identifier,
        since,
        until,
        groups,
        bucket_size,
        json_output,
        csv_output,
        env,
        is_server=True,
    )
