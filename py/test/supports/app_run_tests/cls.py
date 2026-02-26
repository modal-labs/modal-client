# Copyright Modal Labs 2023
import modal

app = modal.App()


@app.cls()
class AParametrized:
    x: int = modal.parameter()

    @modal.method()
    def some_method(self, y: int): ...

    @modal.asgi_app()
    def other_method(self): ...
