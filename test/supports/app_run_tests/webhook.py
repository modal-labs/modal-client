# Copyright Modal Labs 2022
from modal import App, web_endpoint

app = App()


@app.function()
@web_endpoint()
def foo():
    return {"bar": "baz"}
