# Copyright Modal Labs 2024
import modal
from modal import enter, fastapi_endpoint, method

from . import a, b

app = modal.App()
app.include(a.app)
app.include(b.app)


@app.function()
def main_function():
    pass


@app.function()
@fastapi_endpoint()
def web():
    pass


other_app = modal.App()


@other_app.cls()
class Cls:
    @enter()
    def startup(self):
        pass

    @method()
    def method_on_other_app_class(self):
        pass

    @fastapi_endpoint()
    def web_endpoint_on_other_app(self):
        pass
