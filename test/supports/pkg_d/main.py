import os

from pkg_a import a  # noqa  # this would cause an automount warning

import modal

app = modal.App()

image = modal.Image.debian_slim()

if os.environ.get("ADD_SOURCE") == "add":
    image = image.add_local_python_source("pkg_a")

elif os.environ.get("ADD_SOURCE") == "copy":
    image = image.add_local_python_source("pkg_a", copy=True)


@app.function(image=image)
def f():
    pass
