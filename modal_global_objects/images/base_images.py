# Copyright Modal Labs 2022
import os
import sys
from typing import cast

import modal
from modal.image import SUPPORTED_PYTHON_SERIES, ImageBuilderVersion


def dummy():
    pass


if __name__ == "__main__":
    _, name = sys.argv
    constructor = getattr(modal.Image, name)

    builder_version = os.environ.get("MODAL_IMAGE_BUILDER_VERSION")
    assert builder_version, "Script requires MODAL_IMAGE_BUILDER_VERSION environment variable"
    python_versions = SUPPORTED_PYTHON_SERIES[cast(ImageBuilderVersion, builder_version)]

    app = modal.App(f"build-{name.replace('_', '-')}-image")
    for v in python_versions:
        app.function(image=constructor(python_version=v), name=f"{v}")(dummy)

    with modal.enable_output():
        with app.run():
            pass
