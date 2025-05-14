# Copyright Modal Labs 2022
import modal

app = modal.App("hello-world")

if not modal.is_local():
    import nonexistent_package  # noqa


@app.function()
def f(i):
    pass
