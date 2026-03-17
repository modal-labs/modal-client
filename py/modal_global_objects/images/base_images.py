# Copyright Modal Labs 2022
import asyncio
import os
import sys
from typing import Any, cast

import rich
from rich.table import Table

import modal
from modal.image import SUPPORTED_PYTHON_SERIES, ImageBuilderVersion


async def build_images(app: modal.App, constructor: Any, python_versions: list[str]) -> dict[str, modal.Image]:
    images = {v: constructor(python_version=v) for v in python_versions}
    tasks = [image.build.aio(app) for image in images.values()]
    await asyncio.gather(*tasks)
    return images


if __name__ == "__main__":
    _, name = sys.argv

    if name not in ("debian_slim", "micromamba"):
        raise ValueError(f"Unknown base image type: {name}")

    constructor = getattr(modal.Image, name)

    builder_version = os.environ.get("MODAL_IMAGE_BUILDER_VERSION")
    assert builder_version, "Script requires MODAL_IMAGE_BUILDER_VERSION environment variable"
    python_versions = SUPPORTED_PYTHON_SERIES[cast(ImageBuilderVersion, builder_version)]

    # Filter out unsupported free-threaded Python for base images.
    python_versions = [v for v in python_versions if not v.endswith("t")]

    app = modal.App(f"build-{name.replace('_', '-')}-image")
    with app.run(), modal.enable_output():  # Image.build needs these in reverse of normal order to avoid duplicate logs
        images = asyncio.run(build_images(app, constructor, python_versions))

    table = Table(title=f"Images for {name} ({builder_version})")
    table.add_column("Python version")
    table.add_column("Image ID")

    for v, image in images.items():
        table.add_row(v, image.object_id)

    rich.print()
    rich.print(table)
