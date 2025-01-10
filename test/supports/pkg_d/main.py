import os

import modal

from . import sibling  # noqa  # warn if sibling source isn't attached

app = modal.App()

image = modal.Image.debian_slim()

if os.environ.get("ADD_SOURCE") == "add":
    image = image.add_local_python_source("pkg_d")

elif os.environ.get("ADD_SOURCE") == "copy":
    image = image.add_local_python_source("pkg_d", copy=True)


@app.function(image=image)
def f():
    pass
