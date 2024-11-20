# Copyright Modal Labs 2024
from modal import App

app = App()


@app.function()
def f():
    pass


def undecorated_f():
    pass
