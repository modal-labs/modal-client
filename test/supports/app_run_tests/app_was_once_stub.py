# Copyright Modal Labs 2025
import modal

app = modal.Stub()


@app.function()
def foo():
    print("foo")
