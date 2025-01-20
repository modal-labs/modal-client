# Copyright Modal Labs 2024
import a
import b

import modal

app = modal.App()
app.include(a.app)
app.include(b.app)


@app.function()
def main_function():
    pass


other_app = modal.App()


@other_app.function()
def other_function():
    pass
