# Copyright Modal Labs 2022
from modal import App, web_endpoint

app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


@app.function()
@web_endpoint()
def foo():
    return {"bar": "baz"}
