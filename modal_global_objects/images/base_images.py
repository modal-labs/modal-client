# Copyright Modal Labs 2022
import os
import sys
from typing import Any, cast

import rich
from rich.table import Table

import modal
from modal.experimental import notebook_base_image
from modal.image import SUPPORTED_PYTHON_SERIES, ImageBuilderVersion


def dummy():
    pass


if __name__ == "__main__":
    _, name = sys.argv

    constructor: Any  # todo: notebook_base_image has messed up type inference
    if name in ("debian_slim", "micromamba"):
        constructor = getattr(modal.Image, name)
    elif name == "notebook":
        constructor = notebook_base_image
    else:
        raise ValueError(f"Unknown base image type: {name}")

    builder_version = os.environ.get("MODAL_IMAGE_BUILDER_VERSION")
    assert builder_version, "Script requires MODAL_IMAGE_BUILDER_VERSION environment variable"
    python_versions = SUPPORTED_PYTHON_SERIES[cast(ImageBuilderVersion, builder_version)]

    app = modal.App(f"build-{name.replace('_', '-')}-image")
    images_map: dict[str, modal.Image] = {}
    for v in python_versions:
        image = constructor(python_version=v)
        images_map[v] = image
        app.function(image=image, name=f"{v}", serialized=True)(dummy)

    with modal.enable_output():
        with app.run():
            pass

    table = Table(title=f"Images for {name} ({builder_version})")
    table.add_column("Python version")
    table.add_column("Image ID")

    for v, image in images_map.items():
        table.add_row(v, image.object_id)

    rich.print()
    rich.print(table)
