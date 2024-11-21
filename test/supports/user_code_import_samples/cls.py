# Copyright Modal Labs 2024
import modal
from modal import App

app = App()


class UndecoratedC:
    @modal.method()
    def f(self, arg):
        return f"hello {arg}"

    @modal.method()
    def f2(self, arg):
        return f"other {arg}"


C = app.cls()(UndecoratedC)
