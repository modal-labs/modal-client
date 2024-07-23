# Copyright Modal Labs 2024
import c

import modal

app = modal.App()

d = modal.Dict.from_name("my-queue", create_if_missing=True)


@app.function()
def a_func():
    d["foo"] = "bar"


app.include(c.app)
