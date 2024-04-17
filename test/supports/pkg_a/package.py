# Copyright Modal Labs 2022

import pkg_b.f  # noqa
import pkg_b.g.h  # noqa

import modal  # noqa

from .a import *  # noqa
from .b.c import *  # noqa


app = modal.App()


@app.function()
def f():
    pass
