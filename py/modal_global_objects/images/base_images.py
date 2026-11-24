# Copyright Modal Labs 2022
import asyncio
import os
import sys
from typing import Any, cast

import rich
from rich.table import Table

import modal
from modal._image import SUPPORTED_PYTHON_SERIES, ImageBuilderVersion

PLATFORMS = ("linux/amd64", "linux/arm64")


async def build_images(
    app: modal.App, constructor: Any, python_versions: list[str]
) -> dict[tuple[str, str], modal.Image]:
    images = {(p, v): constructor(python_version=v, platform=p) for p in PLATFORMS for v in python_versions}
    tasks = [image.build.aio(app) for image in images.values()]
    await asyncio.gather(*tasks)
    return images


if __name__ == "__main__":
    _, name = sys.argv

    constructor: Any  # todo: notebook_base_image has messed up type inference
    if name in ("debian_slim", "micromamba"):
        constructor = getattr(modal.Image, name)
    elif name == "from_scratch":
        constructor = None
    else:
        raise ValueError(f"Unknown base image type: {name}")

    builder_version = os.environ.get("MODAL_IMAGE_BUILDER_VERSION")
    assert builder_version, "Script requires MODAL_IMAGE_BUILDER_VERSION environment variable"

    app = modal.App(f"build-{name.replace('_', '-')}-image")

    if name == "from_scratch":
        scratch_images = {p: modal.Image.from_scratch(platform=p) for p in PLATFORMS}
        with app.run(), modal.enable_output():
            for image in scratch_images.values():
                image.build(app)

        table = Table(title=f"Images for {name} ({builder_version})")
        table.add_column("Platform")
        table.add_column("Image ID")
        for p, image in scratch_images.items():
            table.add_row(p, image.object_id)
    else:
        python_versions = SUPPORTED_PYTHON_SERIES[cast(ImageBuilderVersion, builder_version)]

        # Filter out unsupported free-threaded Python for base images.
        python_versions = [v for v in python_versions if not v.endswith("t")]

        with (
            app.run(),
            modal.enable_output(),
        ):  # Image.build needs these in reverse of normal order to avoid duplicate logs
            images = asyncio.run(build_images(app, constructor, python_versions))

        table = Table(title=f"Images for {name} ({builder_version})")
        table.add_column("Platform")
        table.add_column("Python version")
        table.add_column("Image ID")

        for (p, v), image in images.items():
            table.add_row(p, v, image.object_id)

    rich.print()
    rich.print(table)
