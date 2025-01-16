# Copyright Modal Labs 2023
import os

import modal

app = modal.App()
import pkg_a  # noqa


if int(os.environ["USE_EXPLICIT"]):
    image_1 = modal.Image.debian_slim().add_local_python_source("pkg_a")  # this should be reused
    # same as above, but different instance - should be app-deduplicated:
    image_2 = (
        modal.Image.debian_slim()
        .add_local_python_source("pkg_a")  # identical to first explicit mount and auto mounts
        .add_local_python_source(
            # custom ignore condition, include normally_not_included.pyc (but skip __pycache__)
            "pkg_a",
            ignore=["**/__pycache__"],
        )
    )
else:
    # only use automounting
    image_1 = modal.Image.debian_slim()
    image_2 = modal.Image.debian_slim()


@app.function(image=image_1)
def foo():
    pass


@app.function(image=image_2)
def bar():
    pass
