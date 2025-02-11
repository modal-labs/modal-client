# Copyright Modal Labs 2022
import modal

app = modal.App("hello-world", include_source=True)  # TODO: remove include_source=True)

if not modal.is_local():
    import nonexistent_package  # noqa


@app.function()
def f(i):
    pass
