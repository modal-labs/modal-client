# Copyright Modal Labs 2023
import modal

app = modal.App()


@app.function()
def hello():
    print("hello")
    return "hello"


@app.function()
def other():
    return "other"
