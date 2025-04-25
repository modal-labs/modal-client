# Copyright Modal Labs 2024
import modal

app = modal.Stub()


@app.function()
def foo():
    print("foo")
