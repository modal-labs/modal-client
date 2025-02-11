# Copyright Modal Labs 2025
from modal import App, asgi_app, method, web_endpoint

app_with_one_web_function = App(
    include_source=True
)  # TODO: remove include_source=True when automount is disabled by default


@app_with_one_web_function.function()
@web_endpoint()
def web1():
    pass


app_with_one_function_one_web_endpoint = App(
    include_source=True
)  # TODO: remove include_source=True when automount is disabled by default


@app_with_one_function_one_web_endpoint.function()
def f1():
    pass


@app_with_one_function_one_web_endpoint.function()
@web_endpoint()
def web2():
    pass


app_with_one_web_method = App(
    include_source=True
)  # TODO: remove include_source=True when automount is disabled by default


@app_with_one_web_method.cls()
class C1:
    @asgi_app()
    def web_3(self):
        pass


app_with_one_web_method_one_method = App(
    include_source=True
)  # TODO: remove include_source=True when automount is disabled by default


@app_with_one_web_method_one_method.cls()
class C2:
    @asgi_app()
    def web_4(self):
        pass

    @method()
    def f2(self):
        pass


app_with_local_entrypoint_and_function = App(
    include_source=True
)  # TODO: remove include_source=True when automount is disabled by default


@app_with_local_entrypoint_and_function.local_entrypoint()
def le_1():
    pass


@app_with_local_entrypoint_and_function.function()
def f3():
    pass
