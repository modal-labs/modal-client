# Copyright Modal Labs 2022
import a  # noqa
import b  # noqa
import b.c  # noqa
import pkg_b  # noqa
import six  # noqa

import modal  # noqa


app = modal.App()


@app.function(serialized=True)
def f():
    pass
