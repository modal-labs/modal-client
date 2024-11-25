# Copyright Modal Labs 2024
import modal
from modal import App

app = App()


class C:
    @modal.method()
    def f(self, arg):
        return f"hello {arg}"

    @modal.method()
    def f2(self, arg):
        return f"other {arg}"


UndecoratedC = C  # keep a reference to original class before overwriting

C = app.cls()(C)  # "decorator" of C
