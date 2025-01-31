# Copyright Modal Labs 2025
from modal import App

app = App()


@app.function()
def f(arg):
    return f"hello {arg}"


def undecorated_f(arg):
    return f"hello {arg}"
