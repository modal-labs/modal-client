# Copyright Modal Labs 2025
import ast  # noqa

import modal

app = modal.App("imports_ast")


@app.function()
def some_func():
    pass
