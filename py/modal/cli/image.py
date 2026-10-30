# Copyright Modal Labs 2026

from __future__ import annotations

import asyncio
import json as json_module
import os
from collections.abc import AsyncIterator
from typing import Optional

import click
from rich.syntax import Syntax

from modal._environments import ensure_env
from modal._image import _Image, _infer_registry_auth_type, _upgrade_image_name
from modal._logs_manager import _resolve_source
from modal._utils.async_utils import synchronizer
from modal._utils.time_utils import timestamp_to_localized_str
from modal.app import _App
from modal.cli.utils import display_table, env_option
from modal.client import _Client
from modal.config import config
from modal.exception import InvalidError
from modal.image import Image
from modal.output import OutputManager
from modal.runner import _run_app
from modal.secret import _Secret
from modal_proto import api_pb2

from ._help import ModalGroup

IMAGE_NAMES_LIST_PAGE_SIZE = 100
IMAGE_NAMES_LIST_PAGE_DELAY_SECONDS = 1.0
IMAGE_NAMES_HISTORY_PAGE_SIZE = 100
IMAGE_NAMES_HISTORY_PAGE_DELAY_SECONDS = 1.0

image_cli = ModalGroup(name="image", help="Manage Images.")


def _print_result_summary(count: int) -> None:
    result_label = "result" if count == 1 else "results"
    OutputManager.get().print(f"Showing {count} {result_label}")


async def _registry_secret(
    secret_name: Optional[str], secret_vars: Optional[str], env: str
) -> tuple[Optional[_Secret], set[str]]:
    if secret_name and secret_vars:
        raise click.UsageError("--registry-secret and --registry-secret-from-env cannot be used together.")
    if secret_name:
        secret = _Secret.from_name(secret_name, environment_name=env)
        return secret, await secret._get_keys()
    keys = [key.strip() for key in (secret_vars or "").split(",") if key.strip()]
    if not keys:
        return None, set()
    return _Secret.from_local_environ(keys), set(keys)


@image_cli.command("build", no_args_is_help=True)
@click.argument("context_dir", metavar="CONTEXT_DIR")
@click.option(
    "-t",
    "--name",
    "name",
    metavar="[ENV/]NAME[:TAG]",
    required=True,
    help="Name to publish the Image as, e.g. `my-image`, `my-image:v2`, or `my-env/my-image`.",
)
@click.option(
    "-f",
    "--file",
    "dockerfile",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to the Dockerfile. Defaults to `CONTEXT_DIR/Dockerfile`.",
)
@click.option(
    "--registry-secret",
    "registry_secret_name",
    metavar="SECRET_NAME",
    default=None,
    help="Name of a Secret with credentials for pulling the base image from a private registry.",
)
@click.option(
    "--registry-secret-from-env",
    "registry_secret_vars",
    default=None,
    metavar="VAR[,VAR...]",
    help=(
        "Comma-separated local environment variable names holding registry credentials. "
        "Cannot be combined with --registry-secret."
    ),
)
@click.option("--force-build", is_flag=True, help="Rebuild every layer of the Image, ignoring cached builds.")
@env_option
@synchronizer.create_blocking
async def build(
    context_dir: str,
    name: str,
    dockerfile: Optional[str],
    registry_secret_name: Optional[str],
    registry_secret_vars: Optional[str],
    force_build: bool,
    env: Optional[str],
) -> None:
    """Build an Image from a Dockerfile and publish it as NAME.

    CONTEXT_DIR is the build context used to resolve relative COPY paths. Use `.` for the current directory.

    NAME defaults to the `latest` tag. `--env` sets the environment the build runs in and where
    `--registry-secret` is looked up. The Image is published into that environment unless NAME has an
    environment prefix, e.g. `my-env/my-image`.

    Examples:

    ```
    modal image build -t my-image .
    modal image build --name my-image:v2 -f docker/Dockerfile .
    modal image build -t my-env/my-image .
    modal image build -t my-image --registry-secret my-registry-creds .
    modal image build -t my-image --registry-secret-from-env REGISTRY_USERNAME,REGISTRY_PASSWORD .
    ```
    """
    if not os.path.isdir(context_dir):
        raise click.UsageError(f"CONTEXT_DIR {context_dir!r} is not a directory.")
    if dockerfile is None:
        dockerfile = os.path.join(context_dir, "Dockerfile")
        if not os.path.isfile(dockerfile):
            raise click.UsageError(f"No Dockerfile found at {dockerfile}. Use --file to specify one.")
    env = ensure_env(env)
    try:
        tag = _upgrade_image_name(name, fallback_environment_name=env)
    except InvalidError as exc:
        raise click.UsageError(str(exc))

    secret, keys = await _registry_secret(registry_secret_name, registry_secret_vars, env)
    if secret is not None:
        # Fail before creating the app; from_dockerfile runs the same check during the build
        try:
            _infer_registry_auth_type(keys)
        except InvalidError as exc:
            raise click.UsageError(str(exc)) from exc
    image = _Image.from_dockerfile(dockerfile, context_dir=context_dir, registry_secret=secret, force_build=force_build)
    app = _App("modal-image-build")
    # Only show the Image build output, which includes its logs
    async with _run_app(app, environment_name=env, stream_output=False):
        await image.build(app)
        await image.publish(tag)

    output_mgr = OutputManager.get()
    output_mgr.step_completed(f"Image built: [bold]{image.object_id}[/bold]")
    output_mgr.step_completed(f"Image published as [bold magenta]{tag}[/bold magenta]! 🎉")
    output_mgr.print("\nUse it with:\n")
    output_mgr.print(Syntax(f'modal.Image.from_name("{tag}")', "python"))


@image_cli.command("logs", no_args_is_help=True)
@click.argument("image_id")
@click.option(
    "--layers",
    type=int,
    default=None,
    help="Fetch logs from the last N build layers. Defaults to 1.",
)
@click.option("--all", "all_layers", is_flag=True, help="Fetch logs from all available build layers.")
@synchronizer.create_blocking
async def logs(image_id: str, layers: int | None, all_layers: bool) -> None:
    """Fetch build logs for an Image."""
    if layers is not None and layers < 1:
        raise click.UsageError("--layers must be at least 1.")
    if all_layers and layers is not None:
        raise click.UsageError("--all cannot be combined with --layers.")

    effective_layers = None if all_layers else layers or 1
    image = Image.from_id(image_id)
    output_mgr = OutputManager.get()
    last_message: str | None = None
    async for entry in image.logs.fetch.aio(layers=effective_layers):
        last_message = entry.message
        await output_mgr.put_fetched_log(
            api_pb2.TaskLogs(
                data=entry.message,
                timestamp=entry.timestamp.timestamp(),
                file_descriptor=_resolve_source(entry.source),  # type: ignore
            )
        )
    output_mgr.flush_lines()
    if last_message is None:
        output_mgr.print(f"No build logs found for Image {image_id}.")
    elif not last_message.endswith("\n"):
        output_mgr.print("")


def _tag_item_json(item: api_pb2.ImageListTagsItem) -> dict[str, str]:
    return {
        "tag": item.tag,
        "image_id": item.image_id,
        "created_at": timestamp_to_localized_str(item.created_at, True) or "-",
        "updated_at": timestamp_to_localized_str(item.updated_at, True) or "-",
    }


def _tag_item_row(item: api_pb2.ImageListTagsItem) -> tuple[str, str, str]:
    return (
        item.tag,
        item.image_id,
        timestamp_to_localized_str(item.updated_at, False) or "-",
    )


def _history_item_json(item: api_pb2.ImageTagRevisionsItem) -> dict[str, str]:
    return {
        "image_id": item.image_id,
        "published_at": timestamp_to_localized_str(item.created_at, True) or "-",
        "published_by": item.published_by or "-",
    }


def _history_item_row(item: api_pb2.ImageTagRevisionsItem) -> tuple[str, str, str]:
    return (item.image_id, timestamp_to_localized_str(item.created_at, False) or "-", item.published_by or "-")


async def _iter_tag_pages(
    client: _Client, env: str, prefix: str
) -> AsyncIterator[tuple[str, list[api_pb2.ImageListTagsItem]]]:
    page_token = ""
    environment_name = env

    while True:
        response = await client._stub.ImageListTags(
            api_pb2.ImageListTagsRequest(
                environment_name=env,
                tag_prefix=prefix,
                max_objects=IMAGE_NAMES_LIST_PAGE_SIZE,
                page_token=page_token,
            )
        )
        environment_name = response.environment_name
        yield environment_name, list(response.items)

        page_token = response.next_page_token
        if not page_token:
            return
        await asyncio.sleep(IMAGE_NAMES_LIST_PAGE_DELAY_SECONDS)


async def _fetch_history_page(
    client: _Client,
    name_tag: str,
    page_token: str,
) -> api_pb2.ImageTagRevisionsResponse:
    return await client._stub.ImageTagRevisions(
        api_pb2.ImageTagRevisionsRequest(
            tag=name_tag,
            max_objects=IMAGE_NAMES_HISTORY_PAGE_SIZE,
            page_token=page_token,
        )
    )


async def _iter_history_pages(
    client: _Client, name_tag: str, first_page: api_pb2.ImageTagRevisionsResponse
) -> AsyncIterator[api_pb2.ImageTagRevisionsResponse]:
    response = first_page

    while True:
        yield response

        if not response.next_page_token:
            return
        await asyncio.sleep(IMAGE_NAMES_HISTORY_PAGE_DELAY_SECONDS)
        response = await _fetch_history_page(client, name_tag, response.next_page_token)


image_names_cli = ModalGroup(name="names", help="Manage Modal Image names.")
image_cli.add_command(image_names_cli)


@image_names_cli.command("list", help="List named Images.")
@env_option
@click.option("--prefix", default="", help="Only include named image tags that start with this prefix.")
@click.option("--json", is_flag=True, default=False)
@synchronizer.create_blocking
async def list_(
    env: Optional[str] = None,
    prefix: str = "",
    json: bool = False,
):
    env = ensure_env(env)
    client = await _Client.from_env()

    if json:
        count = 0
        click.echo("[", nl=False)
        async for _, page_items in _iter_tag_pages(client, env, prefix):
            for item in page_items:
                if count:
                    click.echo(",", nl=False)
                click.echo(json_module.dumps(_tag_item_json(item)), nl=False)
                count += 1
        click.echo("]")
        return

    count = 0
    first_page = True
    async for environment_name, page_items in _iter_tag_pages(client, env, prefix):
        if first_page or page_items:
            title = f"Named Images in environment '{environment_name}'" if first_page else ""
            display_table(["Tag", "Image ID", "Updated at"], [_tag_item_row(item) for item in page_items], title=title)
            first_page = False
        count += len(page_items)

    _print_result_summary(count)


@image_names_cli.command(
    "history", help="Show publishing history for a named Image.", hidden=True, no_args_is_help=True
)
@click.argument("name")
@click.option("--json", is_flag=True, default=False)
@synchronizer.create_blocking
async def history(
    name: str,
    json: bool = False,
):
    upgraded_image_name = _upgrade_image_name(
        name,
        fallback_environment_name=config.get("environment"),
    )
    client = await _Client.from_env()

    first_page = await _fetch_history_page(client, upgraded_image_name, "")
    if not first_page.items:
        raise click.ClickException(f"No publishing history found for named image {first_page.tag!r}.")

    if json:
        click.echo("[", nl=False)
        count = 0
        async for response in _iter_history_pages(client, upgraded_image_name, first_page):
            for item in response.items:
                if count:
                    click.echo(",", nl=False)
                click.echo(json_module.dumps(_history_item_json(item)), nl=False)
                count += 1
        click.echo("]")
        return

    count = 0
    show_title = True
    async for response in _iter_history_pages(client, upgraded_image_name, first_page):
        if response.items:
            title = f"History for {response.tag}" if show_title else ""
            display_table(
                ["Image ID", "Published at", "Published by"],
                [_history_item_row(item) for item in response.items],
                title=title,
            )
            show_title = False
        count += len(response.items)

    _print_result_summary(count)
