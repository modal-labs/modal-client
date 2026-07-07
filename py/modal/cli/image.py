# Copyright Modal Labs 2026

from __future__ import annotations

import asyncio
import json as json_module
from collections.abc import AsyncIterator
from typing import Optional

import click

from modal._environments import ensure_env
from modal._image import _parse_named_image_ref
from modal._utils.async_utils import synchronizer
from modal._utils.time_utils import timestamp_to_localized_str
from modal.cli.utils import display_table, env_option
from modal.client import _Client
from modal.exception import InvalidError
from modal.output import OutputManager
from modal_proto import api_pb2

from ._help import ModalGroup

IMAGE_NAMES_LIST_PAGE_SIZE = 100
IMAGE_NAMES_LIST_PAGE_DELAY_SECONDS = 1.0
IMAGE_NAMES_HISTORY_PAGE_SIZE = 100
IMAGE_NAMES_HISTORY_PAGE_DELAY_SECONDS = 1.0

image_cli = ModalGroup(name="image", help="Manage Images.")
image_names_cli = ModalGroup(name="names", help="Manage Modal Image names.")
image_cli.add_command(image_names_cli)


def _print_result_summary(count: int) -> None:
    result_label = "result" if count == 1 else "results"
    OutputManager.get().print(f"Showing {count} {result_label}")


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
        response = await client.stub.ImageListTags(
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
    env: str,
    tag: str,
    page_token: str,
) -> api_pb2.ImageTagRevisionsResponse:
    return await client.stub.ImageTagRevisions(
        api_pb2.ImageTagRevisionsRequest(
            tag=tag,
            environment_name=env,
            max_objects=IMAGE_NAMES_HISTORY_PAGE_SIZE,
            page_token=page_token,
        )
    )


async def _iter_history_pages(
    client: _Client, env: str, tag: str, first_page: api_pb2.ImageTagRevisionsResponse
) -> AsyncIterator[api_pb2.ImageTagRevisionsResponse]:
    response = first_page

    while True:
        yield response

        if not response.next_page_token:
            return
        await asyncio.sleep(IMAGE_NAMES_HISTORY_PAGE_DELAY_SECONDS)
        response = await _fetch_history_page(client, env, tag, response.next_page_token)


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
@env_option
@click.option("--json", is_flag=True, default=False)
@synchronizer.create_blocking
async def history(
    name: str,
    env: Optional[str] = None,
    json: bool = False,
):
    explicit_env = env  # None when --env is not passed by the user
    env = ensure_env(env)
    try:
        namespace_prefix, name_tag = _parse_named_image_ref(name)
    except InvalidError as exc:
        raise click.UsageError(str(exc)) from exc

    if namespace_prefix:
        if explicit_env is not None:
            raise click.UsageError("Cannot specify '--env' when the image name contains a '/'.")
        env = ""

    client = await _Client.from_env()

    first_page = await _fetch_history_page(client, env, name_tag, "")
    if not first_page.items:
        raise click.ClickException(f"No publishing history found for named image {first_page.tag!r}.")

    if json:
        click.echo("[", nl=False)
        count = 0
        async for response in _iter_history_pages(client, env, name_tag, first_page):
            for item in response.items:
                if count:
                    click.echo(",", nl=False)
                click.echo(json_module.dumps(_history_item_json(item)), nl=False)
                count += 1
        click.echo("]")
        return

    count = 0
    show_title = True
    async for response in _iter_history_pages(client, env, name_tag, first_page):
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
