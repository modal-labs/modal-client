# Copyright Modal Labs 2023
import modal

stub = modal.Stub()


@stub.cls()
class AParametrized:
    def __init__(self, x: int):
        self._x = x

    @modal.method()
    def some_method(self, y: int):
        ...
