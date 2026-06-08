# Copyright Modal Labs 2026

from __future__ import annotations

import asyncio
import json as json_module
from collections.abc import AsyncIterator
from typing import Optional

import click

from modal._environments import ensure_env
from modal._utils.async_utils import synchronizer
from modal._utils.time_utils import timestamp_to_localized_str
from modal.cli.utils import display_table, env_option
from modal.client import _Client
from modal.output import OutputManager
from modal_proto import api_pb2

from ._help import ModalGroup

IMAGE_REGISTRY_LIST_PAGE_SIZE = 100
IMAGE_REGISTRY_LIST_PAGE_DELAY_SECONDS = 1.0

image_cli = ModalGroup(name="image", help="Manage Images.")
image_registry_cli = ModalGroup(name="registry", help="Manage the Modal Image name/tag registry.")
image_cli.add_command(image_registry_cli)


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
                max_objects=IMAGE_REGISTRY_LIST_PAGE_SIZE,
                page_token=page_token,
            )
        )
        environment_name = response.environment_name
        yield environment_name, list(response.items)

        page_token = response.next_page_token
        if not page_token:
            return
        await asyncio.sleep(IMAGE_REGISTRY_LIST_PAGE_DELAY_SECONDS)


@image_registry_cli.command("list", help="List named Images.")
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
