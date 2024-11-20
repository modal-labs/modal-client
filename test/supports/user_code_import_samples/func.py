# Copyright Modal Labs 2024
from modal import App

app = App()


@app.function()
def f(arg):
    return f"hello {arg}"


def undecorated_f(arg):
    return f"hello {arg}"
