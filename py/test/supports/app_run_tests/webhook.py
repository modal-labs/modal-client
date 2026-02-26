# Copyright Modal Labs 2022
from modal import App, fastapi_endpoint

app = App(include_source=False)


@app.function()
@fastapi_endpoint()
def foo():
    return {"bar": "baz"}
