# Copyright Modal Labs 2024
import modal
from modal import method, web_endpoint

from . import a, b

app = modal.App()
app.include(a.app)
app.include(b.app)


@app.function()
def main_function():
    pass


@app.function()
@web_endpoint()
def web():
    pass


other_app = modal.App()


@other_app.cls()
class Cls:
    @method()
    def method_on_other_app_class(self):
        pass

    @web_endpoint()
    def web_endpoint_on_other_app(self):
        pass
