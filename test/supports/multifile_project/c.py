# Copyright Modal Labs 2024
import modal

app = modal.App("c", include_source=True)  # TODO: remove include_source=True)


@app.function()
def c_func():
    pass
