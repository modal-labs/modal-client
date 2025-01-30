# Copyright Modal Labs 2023
import modal

app = modal.App()


@app.cls()
class AParametrized:
    def __init__(self, x: int):
        self._x = x

    @modal.method()
    def some_method(self, y: int):
        ...

    @modal.asgi_app()
    def other_method(self):
        ...
