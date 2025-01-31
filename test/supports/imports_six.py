# Copyright Modal Labs 2025
import six  # noqa

import modal

app = modal.App("imports_six")


@app.function()
def some_func():
    pass
