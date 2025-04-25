# Copyright Modal Labs 2024
from modal import App

app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


@app.function()
def f(arg):
    return f"hello {arg}"


def undecorated_f(arg):
    return f"hello {arg}"
