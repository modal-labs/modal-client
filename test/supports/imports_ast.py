# Copyright Modal Labs 2024
import ast  # noqa

import modal

app = modal.App("imports_ast", include_source=True)  # TODO: remove include_source=True)


@app.function()
def some_func():
    pass
