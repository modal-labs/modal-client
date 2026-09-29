# Copyright Modal Labs 2026
import re
from datetime import datetime
from json import dumps
from typing import Optional

import click
from rich.table import Column, Table
from rich.text import Text

from modal._environments import ensure_env
from modal._object import _get_environment_name
from modal._utils.async_utils import synchronizer
from modal._utils.time_utils import timestamp_to_localized_str
from modal.cli.utils import confirm_or_suggest_yes, display_table, env_option, yes_option
from modal.client import _Client
from modal.exception import NotFoundError
from modal.output import OutputManager
from modal.volume import _Volume
from modal_proto import api_pb2

from ._help import ModalGroup
from ._logs import _validate_logs_args
from .server import _run_server_logs, _run_server_stats, _server_stats_time_range

_ENDPOINT_HELP = """
Create and manage LLM inference endpoints.

Modal Endpoints deploy production-ready LLM inference servers with minimal coding or configuration.
Endpoints support pre-trained open models along with custom weights from a private Hugging Face repo
or Modal Volume.

See https://modal.com/docs/guide/endpoints for more information.
"""

endpoint_cli = ModalGroup(name="endpoint", help=_ENDPOINT_HELP)

_ENDPOINT_ID_RE = re.compile(r"^ep-[a-zA-Z0-9]{22}$")
_DEFAULT_ROUTING_REGION = "us-west"
_SERVING_MODE_LABELS = {
    api_pb2.ENDPOINT_SERVING_MODE_DEDICATED: "dedicated",
    api_pb2.ENDPOINT_SERVING_MODE_SHARED: "shared",
}
_ENDPOINT_STATUS_LABELS = {
    api_pb2.EndpointGetInfoResponse.ENDPOINT_STATUS_PROVISIONING: "provisioning",
    api_pb2.EndpointGetInfoResponse.ENDPOINT_STATUS_LIVE: "live",
    api_pb2.EndpointGetInfoResponse.ENDPOINT_STATUS_FAILED: "failed",
    api_pb2.EndpointGetInfoResponse.ENDPOINT_STATUS_CANCELLING: "cancelling",
    api_pb2.EndpointGetInfoResponse.ENDPOINT_STATUS_CANCELLED: "cancelled",
    api_pb2.EndpointGetInfoResponse.ENDPOINT_STATUS_STOPPED: "stopped",
}


def _single_routing_region_callback(ctx: click.Context, param: click.Parameter, value: tuple[str, ...]) -> str:
    if len(value) > 1:
        raise click.UsageError("--routing-region can only be specified once.")
    return value[0] if value else _DEFAULT_ROUTING_REGION


def _is_endpoint_id(value: str) -> bool:
    return bool(_ENDPOINT_ID_RE.match(value))


def _endpoint_created_at_table_value(item: api_pb2.EndpointListItem) -> str:
    return (timestamp_to_localized_str(item.metadata.creation_info.created_at, True) or "")[:16]


def _endpoint_list_item_is_stopped(item: api_pb2.EndpointListItem) -> bool:
    return item.app_state in (api_pb2.APP_STATE_STOPPING, api_pb2.APP_STATE_STOPPED)


async def _fetch_endpoint_lifecycle(client: _Client, endpoint_id: str) -> api_pb2.EndpointLifecycle:
    resp = await client._stub.EndpointGetLifecycle(api_pb2.EndpointGetLifecycleRequest(endpoint_id=endpoint_id))
    return resp.lifecycle


async def _get_endpoint_lifecycle(client: _Client, endpoint_id: str) -> api_pb2.EndpointLifecycle:
    try:
        return await _fetch_endpoint_lifecycle(client, endpoint_id)
    except NotFoundError as exc:
        raise click.ClickException(f"Endpoint '{endpoint_id}' not found.") from exc


async def _resolve_endpoint_name(
    client: _Client,
    endpoint_name: str,
    env_name: str | None,
    *,
    id_lookup_not_found: NotFoundError | None = None,
) -> tuple[str, str]:
    response = await client._stub.EndpointGetByName(
        api_pb2.EndpointGetByNameRequest(name=endpoint_name, environment_name=env_name or "")
    )
    environment_name = response.environment_name or env_name or ""
    if not response.endpoint_id:
        if id_lookup_not_found is not None:
            raise click.ClickException(f"Endpoint '{endpoint_name}' not found.") from id_lookup_not_found
        raise click.ClickException(f"Endpoint '{endpoint_name}' not found in environment '{environment_name}'.")
    return response.endpoint_id, environment_name


async def _resolve_endpoint_identifier(
    client: _Client,
    endpoint_identifier: str,
    env_name: str | None,
) -> tuple[str, str | None, str, api_pb2.EndpointLifecycle]:
    id_lookup_not_found: NotFoundError | None = None
    if _is_endpoint_id(endpoint_identifier):
        try:
            lifecycle = await _fetch_endpoint_lifecycle(client, endpoint_identifier)
        except NotFoundError as exc:
            id_lookup_not_found = exc
        else:
            return endpoint_identifier, None, lifecycle.environment_name, lifecycle

    endpoint_id, environment_name = await _resolve_endpoint_name(
        client,
        endpoint_identifier,
        env_name,
        id_lookup_not_found=id_lookup_not_found,
    )
    lifecycle = await _get_endpoint_lifecycle(client, endpoint_id)
    return endpoint_id, endpoint_identifier, environment_name, lifecycle


def _endpoint_already_stopped_message(
    endpoint_id: str,
    endpoint_name: str | None,
    environment_name: str,
) -> str:
    if endpoint_name:
        return f"Endpoint '{endpoint_name}' in environment '{environment_name}' is already stopped."
    return f"Endpoint {endpoint_id} is already stopped."


async def _get_endpoint_info(
    client: _Client,
    endpoint_identifier: str,
    env: str | None,
) -> tuple[str, api_pb2.EndpointGetInfoResponse]:
    id_lookup_not_found: NotFoundError | None = None
    if _is_endpoint_id(endpoint_identifier):
        try:
            response = await client._stub.EndpointGetInfo(
                api_pb2.EndpointGetInfoRequest(endpoint_id=endpoint_identifier)
            )
        except NotFoundError as exc:
            id_lookup_not_found = exc
        else:
            return endpoint_identifier, response

    endpoint_id, _ = await _resolve_endpoint_name(
        client,
        endpoint_identifier,
        env,
        id_lookup_not_found=id_lookup_not_found,
    )
    response = await client._stub.EndpointGetInfo(api_pb2.EndpointGetInfoRequest(endpoint_id=endpoint_id))
    return endpoint_id, response


def _dedicated_endpoint_metadata(
    endpoint_identifier: str,
    response: api_pb2.EndpointGetInfoResponse,
    command: str,
) -> api_pb2.EndpointGetInfoResponse.EndpointHandleMetadata:
    if response.info.serving_mode == api_pb2.ENDPOINT_SERVING_MODE_SHARED:
        raise click.ClickException(
            f"Endpoint '{endpoint_identifier}' is a shared Endpoint. "
            f"`modal endpoint {command}` is only available for dedicated Endpoints."
        )

    if response.info.serving_mode != api_pb2.ENDPOINT_SERVING_MODE_DEDICATED:
        raise click.ClickException(
            f"Endpoint '{endpoint_identifier}' uses an unsupported serving mode ({response.info.serving_mode})."
        )

    if not response.HasField("metadata") or not response.metadata.app_id or not response.metadata.server_id:
        if response.info.status == api_pb2.EndpointGetInfoResponse.ENDPOINT_STATUS_STOPPED:
            raise click.ClickException(
                f"{command.capitalize()} cannot be retrieved for Endpoint '{endpoint_identifier}' because information "
                "about its backing server has expired."
            )
        raise click.ClickException(
            f"{command.capitalize()} cannot be retrieved for Endpoint '{endpoint_identifier}' because it does not have "
            "a backing server."
        )
    return response.metadata


@endpoint_cli.command("list", panel="Management")
@click.option("--json", is_flag=True, default=False)
@env_option
@synchronizer.create_blocking
async def list_(*, json: bool = False, env: Optional[str] = None):
    """List Endpoints that are provisioning or running in an environment."""
    env_name = ensure_env(env)
    environment_name = _get_environment_name(env_name)
    client = await _Client.from_env()

    items: list[api_pb2.EndpointListItem] = []

    async def retrieve_page(created_before: float) -> bool:
        max_page_size = 100
        pagination = api_pb2.ListPagination(max_objects=max_page_size, created_before=created_before)
        req = api_pb2.EndpointListRequest(environment_name=environment_name, pagination=pagination)
        resp = await client._stub.EndpointList(req)
        items.extend(resp.items)
        return len(resp.items) < max_page_size

    finished = await retrieve_page(datetime.now().timestamp())
    while not finished:
        finished = await retrieve_page(items[-1].metadata.creation_info.created_at)

    active_items = [item for item in items if not _endpoint_list_item_is_stopped(item)]

    env_part = f" in environment '{env_name}'" if env_name else ""
    title = f"Endpoints{env_part}"
    if json:
        json_rows = [
            (
                item.name,
                item.endpoint_id,
                item.status,
                timestamp_to_localized_str(item.metadata.creation_info.created_at, json) or "",
                item.metadata.creation_info.created_by,
            )
            for item in active_items
        ]
        display_table(
            ["Name", "Endpoint ID", "Status", "Created at", "Created by"],
            json_rows,
            json=True,
            title=title,
        )
    else:
        table_rows = [
            (
                item.name,
                item.endpoint_id,
                item.status,
                _endpoint_created_at_table_value(item),
            )
            for item in active_items
        ]
        display_table(
            [
                Column("Name", width=14, overflow="ellipsis", no_wrap=True),
                Column("Endpoint ID", width=25, no_wrap=True),
                Column("Status", width=12, overflow="ellipsis", no_wrap=True),
                Column("Created at", width=16, no_wrap=True),
            ],
            table_rows,
            title=title,
        )


@endpoint_cli.command("create", panel="Management", no_args_is_help=True)
@env_option
@click.option(
    "--name",
    default=None,
    help="Endpoint name. If not provided, a default will be derived from the model name.",
)
@click.option(
    "--model",
    required=True,
    help="Hugging Face repo ID for the base model architecture (e.g., 'Qwen/Qwen3.6-27B-FP8').",
)
@click.option(
    "--routing-region",
    "routing_region",
    multiple=True,
    callback=_single_routing_region_callback,
    help=f"Region to route inference requests through. Defaults to {_DEFAULT_ROUTING_REGION}.",
)
@click.option(
    "--compute-region",
    "compute_regions",
    multiple=True,
    help=(
        "Region to run Endpoint containers in. May be specified multiple times. "
        "This incurs a region selection price multiplier."
    ),
)
@click.option(
    "--colocate-compute",
    is_flag=True,
    default=False,
    help="Run all containers within the routing region. This incurs a region selection price multiplier.",
)
@click.option(
    "--unauthenticated",
    is_flag=True,
    default=False,
    help="Allow unauthenticated HTTP requests to the endpoint.",
)
@click.option("--custom-hf-repo", default=None, help="Hugging Face repo ID for fine-tuned model weights.")
@click.option("--custom-hf-revision", default=None, help="Git revision for --custom-hf-repo.")
@click.option("--custom-hf-token", default=None, help="Hugging Face token for private --custom-hf-repo.")
@click.option("--custom-volume-name", default=None, help="Modal Volume name containing custom model weights.")
@click.option("--custom-volume-path", default=None, help="Path within Volume containing model weights.")
@synchronizer.create_blocking
async def create(
    name: Optional[str],
    model: Optional[str] = None,
    routing_region: str = _DEFAULT_ROUTING_REGION,
    compute_regions: tuple[str, ...] = (),
    colocate_compute: bool = False,
    unauthenticated: bool = False,
    custom_hf_repo: Optional[str] = None,
    custom_hf_revision: Optional[str] = None,
    custom_hf_token: Optional[str] = None,
    custom_volume_name: Optional[str] = None,
    custom_volume_path: Optional[str] = None,
    env: Optional[str] = None,
):
    """Deploy a new Endpoint.

    Examples:

    Create an Endpoint from a base model:
    ```bash
    modal endpoint create --model Qwen/Qwen3.6-27B-FP8
    ```

    Create an Endpoint with an explicit name:
    ```bash
    modal endpoint create --name qwen-chat --model Qwen/Qwen3.6-27B-FP8
    ```

    Create an Endpoint with explicit routing and compute regions:
    ```bash
    modal endpoint create --model Qwen/Qwen3.6-27B-FP8 \\
      --routing-region us-east --compute-region us-west
    ```

    Create an Endpoint from a private Hugging Face model:
    ```bash
    modal endpoint create --name my-ft --model Qwen/Qwen3.6-27B-FP8 \\
      --custom-hf-repo acme/qwen-ft --custom-hf-token $HF_TOKEN
    ```

    Create an Endpoint from custom weights in a Modal Volume:
    ```bash
    modal endpoint create --name my-ft --model Qwen/Qwen3.6-27B-FP8 \\
      --custom-volume-name qwen-ft --custom-volume-path /models/qwen
    ```

    """
    if compute_regions and colocate_compute:
        raise click.UsageError("--compute-region and --colocate-compute are mutually exclusive.")
    if custom_hf_repo and custom_volume_name:
        raise click.UsageError("--custom-hf-repo and --custom-volume-name are mutually exclusive.")
    if custom_volume_name and not custom_volume_path:
        raise click.UsageError("--custom-volume-path is required with --custom-volume-name.")
    if (custom_hf_revision or custom_hf_token) and not custom_hf_repo:
        raise click.UsageError("--custom-hf-revision and --custom-hf-token require --custom-hf-repo.")
    if custom_volume_path and not custom_volume_name:
        raise click.UsageError("--custom-volume-path requires --custom-volume-name.")

    env_name = ensure_env(env)
    environment_name = _get_environment_name(env_name)
    client = await _Client.from_env()

    compute_region_spec = api_pb2.EndpointComputeRegionSpec()
    if compute_regions:
        compute_region_spec.explicit.regions.extend(compute_regions)
    elif colocate_compute:
        compute_region_spec.colocated.SetInParent()
    else:
        compute_region_spec.auto.SetInParent()

    if custom_hf_repo:
        assert model is not None  # validated above
        hf_source = api_pb2.EndpointHuggingFaceModelSource(repo_id=custom_hf_repo)
        if custom_hf_revision:
            hf_source.revision = custom_hf_revision
        if custom_hf_token:
            hf_source.huggingface_token = custom_hf_token
        model_source = api_pb2.EndpointModelSource(
            custom=api_pb2.EndpointCustomModelSource(base_model_repo_id=model, huggingface=hf_source)
        )
    elif custom_volume_name:
        assert custom_volume_path is not None and model is not None  # validated above
        volume = await _Volume.from_name(custom_volume_name, environment_name=environment_name).hydrate(client)
        model_source = api_pb2.EndpointModelSource(
            custom=api_pb2.EndpointCustomModelSource(
                base_model_repo_id=model,
                modal_volume=api_pb2.EndpointModalVolumeModelSource(
                    volume_id=volume.object_id,
                    model_path=custom_volume_path,
                ),
            )
        )
    else:
        assert model is not None  # validated above
        model_source = api_pb2.EndpointModelSource(base_model_repo_id=model)

    req = api_pb2.EndpointCreateRequest(
        proxy_regions=[routing_region],
        compute_region=compute_region_spec,
        model=model_source,
        environment_name=environment_name,
        unauthenticated=unauthenticated,
    )
    if name:
        req.name = name
    resp = await client._stub.EndpointCreate(req)

    output = OutputManager.get()
    output.print(f"[green]✓[/green] Endpoint '{resp.name}' ({resp.endpoint_id}) was created and started provisioning.")
    if resp.endpoint_page_url:
        output.print(f"  → View progress at [magenta]{resp.endpoint_page_url}[/magenta].")
        output.print("  → The Endpoint will also appear in [cyan]modal endpoint list[/cyan].")
    else:
        output.print("  → The Endpoint will appear in [cyan]modal endpoint list[/cyan].")


@endpoint_cli.command("stop", panel="Management", no_args_is_help=True)
@click.argument("endpoint_identifier")
@yes_option
@env_option
@synchronizer.create_blocking
async def stop(
    endpoint_identifier: str,
    *,
    yes: bool = False,
    env: str | None = None,
):
    """Permanently stop an Endpoint and terminate any running containers."""
    env_name = ensure_env(env)
    client = await _Client.from_env()
    endpoint_id, endpoint_name, environment_name, lifecycle = await _resolve_endpoint_identifier(
        client, endpoint_identifier, env_name
    )
    if lifecycle.status == api_pb2.ENDPOINT_LIFECYCLE_STATUS_STOPPED:
        raise click.ClickException(_endpoint_already_stopped_message(endpoint_id, endpoint_name, environment_name))

    if not yes:
        if endpoint_name:
            msg = f"Are you sure you want to stop Endpoint '{endpoint_name}' in environment '{environment_name}'?"
        else:
            msg = f"Are you sure you want to stop Endpoint {endpoint_id}?"
        confirm_or_suggest_yes(msg)

    await client._stub.EndpointStop(
        api_pb2.EndpointStopRequest(endpoint_id=endpoint_id, source=api_pb2.ENDPOINT_STOP_SOURCE_CLI)
    )

    output = OutputManager.get()
    if endpoint_name:
        output.print(
            f"[green]✓[/green] Stopped Endpoint '{endpoint_name}' "
            f"in environment '{environment_name}' (ID: {endpoint_id})."
        )
    else:
        output.print(f"[green]✓[/green] Stopped Endpoint {endpoint_id}.")


@endpoint_cli.command("info", panel="Inspection", no_args_is_help=True)
@click.argument("endpoint_identifier")
@env_option
@click.option("--json", is_flag=True, default=False)
@click.option("--no-color", is_flag=True, default=False, help="Disable colors in the output.")
@synchronizer.create_blocking
async def info(
    endpoint_identifier: str,
    *,
    env: str | None = None,
    json: bool = False,
    no_color: bool = False,
) -> None:
    """Show details about an Endpoint, such as the model, URL, and status.

    Examples:

    Get information about an Endpoint by name:

    ```bash
    modal endpoint info qwen-chat
    ```

    Get information about an Endpoint by ID:

    ```bash
    modal endpoint info ep-123456
    ```

    Output the information as JSON:

    ```bash
    modal endpoint info qwen-chat --json
    ```

    Disable color output:

    ```bash
    modal endpoint info qwen-chat --no-color
    ```

    """
    env_name = ensure_env(env)
    client = await _Client.from_env()
    endpoint_id, response = await _get_endpoint_info(client, endpoint_identifier, env_name)
    endpoint_info = response.info
    endpoint_lifecycle = endpoint_info.lifecycle
    serving_mode = _SERVING_MODE_LABELS.get(endpoint_info.serving_mode, "unknown")
    status = _ENDPOINT_STATUS_LABELS.get(endpoint_info.status, "unknown")

    metadata = response.metadata if response.HasField("metadata") else None

    output = OutputManager.get()
    if json:
        output.print_json(
            dumps(
                {
                    "name": endpoint_info.name,
                    "endpoint_id": endpoint_id,
                    "repo_id": endpoint_info.repo_id,
                    "revision": endpoint_info.revision or None,
                    "volume_id": endpoint_info.volume_id or None,
                    "model_path": endpoint_info.model_path if endpoint_info.volume_id else None,
                    "service_url": endpoint_info.service_url or None,
                    "requires_proxy_auth": endpoint_info.requires_proxy_auth,
                    "serving_mode": serving_mode,
                    "status": status,
                    "lifecycle": {
                        "created_at": timestamp_to_localized_str(endpoint_lifecycle.created_at, json),
                        "created_by": endpoint_lifecycle.created_by or None,
                        "stopped_at": timestamp_to_localized_str(endpoint_lifecycle.stopped_at, json),
                        "stopped_by": endpoint_lifecycle.stopped_by or None,
                    },
                    "app_id": metadata.app_id if metadata is not None else None,
                    "server_id": metadata.server_id if metadata is not None else None,
                    "environment_name": (
                        metadata.environment_name if metadata is not None else endpoint_lifecycle.environment_name
                    ),
                }
            )
        )
        return

    header = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
    endpoint_text = Text(endpoint_info.name)
    endpoint_text.append(f" · {serving_mode.capitalize()}", style=None)
    header.add_row(Text("Endpoint:"), endpoint_text)
    header.add_row(Text("Endpoint ID:"), Text(endpoint_id))
    header.add_row(Text("State:"), Text(status))
    model_text = Text(endpoint_info.repo_id)
    if endpoint_info.revision:
        model_text.append(f"@{endpoint_info.revision[:7]}")
    header.add_row(Text("Model:"), model_text)
    if endpoint_info.volume_id:
        model_path = endpoint_info.model_path.strip("/")
        served_from = f"{endpoint_info.volume_id}:/{model_path}" if model_path else f"{endpoint_info.volume_id}:/"
        header.add_row(Text("Served From:"), Text(served_from))
    if endpoint_info.service_url:
        url_text = Text(endpoint_info.service_url)
        if not endpoint_info.requires_proxy_auth:
            url_text.append(" · ")
            url_text.append("Unauthenticated", style=None if no_color else "yellow")
        header.add_row(Text("URL:"), url_text)

    def event(label: str, timestamp: float, actor: str) -> None:
        parts = [timestamp_to_localized_str(timestamp, isotz=False), actor]
        header.add_row(Text(label + ":"), Text(" · ".join(part for part in parts if part)))

    if endpoint_lifecycle.stopped_at:
        event("Stopped", endpoint_lifecycle.stopped_at, endpoint_lifecycle.stopped_by)
    event("Created", endpoint_lifecycle.created_at, endpoint_lifecycle.created_by)
    output.print(header)


@endpoint_cli.command("logs", panel="Inspection", no_args_is_help=True)
@click.argument("endpoint_identifier")
@click.option("-f", "--follow", is_flag=True, default=False, help="Stream log output until interrupted")
@click.option(
    "--since",
    default=None,
    help="Start of time range. Accepts ISO 8601 datetime or relative time, e.g. '1d', '2h', or '30m'.",
)
@click.option("--until", default=None, help="End of time range; accepts the same argument types as --since")
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
    endpoint_identifier: str,
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
    """Fetch or stream logs for a dedicated Endpoint.

    By default, this command fetches the last 100 log entries and exits. Use ``-f`` to
    live-stream logs from a running Endpoint instead. Fetch and follow are mutually exclusive.

    Examples:

    Get recent logs by Endpoint name:

    ```bash
    modal endpoint logs qwen-chat
    ```

    Get recent logs by Endpoint ID:

    ```bash
    modal endpoint logs ep-123456
    ```

    Follow logs from a running Endpoint:

    ```bash
    modal endpoint logs qwen-chat -f
    ```

    Fetch the last 1000 entries:

    ```bash
    modal endpoint logs qwen-chat --tail 1000
    ```

    Fetch logs from the last two hours:

    ```bash
    modal endpoint logs qwen-chat --since 2h
    ```

    Fetch logs in a specific time range:

    ```bash
    modal endpoint logs qwen-chat --since 2026-09-01T05:00:00 --until 2026-09-01T08:00:00
    ```

    Filter logs and include timestamps:

    ```bash
    modal endpoint logs qwen-chat --source stderr --search timeout --timestamps
    ```

    """
    env_name = ensure_env(env)
    _validate_logs_args(follow=follow, since=since, until=until, tail=tail)
    client = await _Client.from_env()
    _, response = await _get_endpoint_info(client, endpoint_identifier, env_name)
    metadata = _dedicated_endpoint_metadata(endpoint_identifier, response, "logs")
    await _run_server_logs(
        metadata.app_id,
        metadata.server_id,
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


@endpoint_cli.command("stats", panel="Inspection", no_args_is_help=True)
@click.argument("endpoint_identifier")
@click.option(
    "--since",
    default=None,
    help=(
        "Start of time range. Treated as local time if a timezone is not supplied. "
        "Accepts an ISO 8601 datetime or relative time such as '2h' or '30m'."
    ),
)
@click.option(
    "--until",
    default=None,
    help="End of time range. Treated as local time if a timezone is not supplied.",
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
    endpoint_identifier: str,
    since: str | None = None,
    until: str | None = None,
    container_id: str | None = None,
    no_color: bool = False,
    json_output: bool = False,
    *,
    env: str | None = None,
) -> None:
    """Show aggregate Server statistics for a dedicated Endpoint.

    The default time range is the most recent hour. Metrics are provided on a
    best-effort basis and may be delayed or incomplete.

    Examples:

    Show stats for the last hour by Endpoint name:

    ```bash
    modal endpoint stats qwen-chat
    ```

    Show stats by Endpoint ID:

    ```bash
    modal endpoint stats ep-123456
    ```

    Show stats for a relative time range:

    ```bash
    modal endpoint stats qwen-chat --since 5h --until 2h
    ```

    Show stats for the hour ending two hours ago:

    ```bash
    modal endpoint stats qwen-chat --until 2h
    ```

    Show stats for an explicit time range as JSON:

    ```bash
    modal endpoint stats qwen-chat \\
        --since 2026-09-01T05:00:00Z \\
        --until 2026-09-01T08:00:00Z \\
        --json
    ```

    Show stats for a specific container:

    ```bash
    modal endpoint stats qwen-chat --container-id ta-12345
    ```

    """
    since_dt, until_dt, history_duration = _server_stats_time_range(since, until)

    env_name = ensure_env(env)
    client = await _Client.from_env()
    endpoint_id, response = await _get_endpoint_info(client, endpoint_identifier, env_name)
    metadata = _dedicated_endpoint_metadata(endpoint_identifier, response, "stats")
    await _run_server_stats(
        client,
        metadata.server_id,
        since_dt=since_dt,
        until_dt=until_dt,
        history_duration=history_duration,
        container_id=container_id,
        no_color=no_color,
        json_output=json_output,
        heading=f"Endpoint stats for {response.info.name} ({endpoint_id})",
        endpoint_id=endpoint_id,
    )
