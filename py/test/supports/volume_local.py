# Copyright Modal Labs 2024
from modal import App, Volume

app2 = App(include_source=False)


@app2.function(volumes={"/foo": Volume.from_name("my-vol")})
def volume_func():
    pass


@app2.function()
def volume_func_outer():
    volume_func.local()
