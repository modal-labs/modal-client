# Copyright Modal Labs 2026
import re
from datetime import datetime
from typing import Optional

import click
from rich.table import Column

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
    resp = await client.stub.EndpointGetLifecycle(api_pb2.EndpointGetLifecycleRequest(endpoint_id=endpoint_id))
    return resp.lifecycle


async def _get_endpoint_lifecycle(client: _Client, endpoint_id: str) -> api_pb2.EndpointLifecycle:
    try:
        return await _fetch_endpoint_lifecycle(client, endpoint_id)
    except NotFoundError as exc:
        raise click.ClickException(f"Endpoint '{endpoint_id}' not found.") from exc


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

    resp = await client.stub.EndpointGetByName(
        api_pb2.EndpointGetByNameRequest(name=endpoint_identifier, environment_name=env_name or "")
    )
    environment_name = resp.environment_name or env_name or ""
    if not resp.endpoint_id:
        if id_lookup_not_found is not None:
            raise click.ClickException(f"Endpoint '{endpoint_identifier}' not found.") from id_lookup_not_found
        raise click.ClickException(f"Endpoint '{endpoint_identifier}' not found in environment '{environment_name}'.")

    lifecycle = await _get_endpoint_lifecycle(client, resp.endpoint_id)
    return resp.endpoint_id, endpoint_identifier, environment_name, lifecycle


def _endpoint_already_stopped_message(
    endpoint_id: str,
    endpoint_name: str | None,
    environment_name: str,
) -> str:
    if endpoint_name:
        return f"Endpoint '{endpoint_name}' in environment '{environment_name}' is already stopped."
    return f"Endpoint {endpoint_id} is already stopped."


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
    if custom_hf_repo and custom_volume_name:
        raise click.UsageError("--custom-hf-repo and --custom-volume-name are mutually exclusive.")
    if custom_volume_name and not custom_volume_path:
        raise click.UsageError("--custom-volume-path is required with --custom-volume-name.")
    if (custom_hf_revision or custom_hf_token) and not custom_hf_repo:
        raise click.UsageError("--custom-hf-revision and --custom-hf-token require --custom-hf-repo.")
    if custom_volume_path and not custom_volume_name:
        raise click.UsageError("--custom-volume-path requires --custom-volume-name.")

    env_name = ensure_env(env)
    environment_name = _get_environment_name(env_name) or ""
    client = await _Client.from_env()

    compute_region_spec = api_pb2.EndpointComputeRegionSpec()
    if colocate_compute:
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
    resp = await client.stub.EndpointCreate(req)

    output = OutputManager.get()
    output.print(f"[green]✓[/green] Endpoint '{resp.name}' ({resp.endpoint_id}) was created and started provisioning.")
    if resp.endpoint_page_url:
        output.print(f"  → View progress at [magenta]{resp.endpoint_page_url}[/magenta].")
        output.print("  → The Endpoint will also appear in [cyan]modal endpoint list[/cyan].")
    else:
        output.print("  → The Endpoint will appear in [cyan]modal endpoint list[/cyan].")


@endpoint_cli.command("list", panel="Management")
@click.option("--json", is_flag=True, default=False)
@env_option
@synchronizer.create_blocking
async def list_(*, json: bool = False, env: Optional[str] = None):
    """List Endpoints that are provisioning or running in an environment."""
    env_name = ensure_env(env)
    environment_name = _get_environment_name(env_name) or ""
    client = await _Client.from_env()

    items: list[api_pb2.EndpointListItem] = []

    async def retrieve_page(created_before: float) -> bool:
        max_page_size = 100
        pagination = api_pb2.ListPagination(max_objects=max_page_size, created_before=created_before)
        req = api_pb2.EndpointListRequest(environment_name=environment_name, pagination=pagination)
        resp = await client.stub.EndpointList(req)
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

    await client.stub.EndpointStop(
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
