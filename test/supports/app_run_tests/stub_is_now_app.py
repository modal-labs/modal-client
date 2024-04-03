# Copyright Modal Labs 2024
import modal

app = modal.App()


@app.function()
def foo():
    print("foo")
