# Copyright Modal Labs 2022
from modal import App, method

app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


@app.cls()
class VaClass:
    @method()
    def va_method(self, *args):
        pass  # Set via @servicer.function_body

    @method()
    def va_method_invalid(self, x: int, *args):
        pass  # Set via @servicer.function_body


@app.function()
def va_function(*args):
    pass  # Set via @servicer.function_body


@app.function()
def va_function_invalid(x: int, *args):
    pass  # Set via @servicer.function_body


@app.local_entrypoint()
def va_entrypoint(*args):
    print(f"args: {args}")


@app.local_entrypoint()
def va_entrypoint_invalid(x: int, *args):
    print(f"args: {args}")
