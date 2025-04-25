# Copyright Modal Labs 2024
import six  # noqa

import modal

app = modal.App("imports_six", include_source=True)  # TODO: remove include_source=True)


@app.function()
def some_func():
    pass
