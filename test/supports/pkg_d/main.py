# Copyright Modal Labs 2025
import os

from pkg_a import a  # noqa  # this would cause an automount warning

import modal

app = modal.App()

image = modal.Image.debian_slim()

if os.environ.get("ADD_SOURCE") == "add":
    # intentionally makes add local not the last call, to make sure the added modules transfer to downstream layers
    image = image.add_local_python_source("pkg_a").add_local_file(__file__, "/tmp/blah")

elif os.environ.get("ADD_SOURCE") == "copy":
    # intentionally makes add local not the last call, to make sure the added modules transfer to downstream layers
    image = image.add_local_python_source("pkg_a", copy=True).run_commands("echo hello")


@app.function(image=image)
def f():
    pass
