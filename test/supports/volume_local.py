# Copyright Modal Labs 2022
from __future__ import annotations

from modal import App, Volume

app2 = App()


@app2.function(volumes={"/foo": Volume.from_name("my-vol")})
def volume_func():
    pass


@app2.function()
def volume_func_outer():
    volume_func.local()
