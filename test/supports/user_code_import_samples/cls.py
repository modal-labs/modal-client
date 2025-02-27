# Copyright Modal Labs 2024
import modal
from modal import App

app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


class C:
    @modal.method()
    def f(self, arg):
        return f"hello {arg}"

    @modal.method()
    def f2(self, arg):
        return f"other {arg}"

    @modal.method()
    def calls_f_remote(self, arg):
        return self.f.remote(arg)


UndecoratedC = C  # keep a reference to original class before overwriting

C = app.cls()(C)  # type: ignore[misc,assignment]   # Cls[C] gets replaced by user type type[C]
