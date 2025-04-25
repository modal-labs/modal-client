# Copyright Modal Labs 2025
import modal

remote_func = modal.Function.from_name("app", "some_func")
remote_cls = modal.Cls.from_name("app", "some_class")

app = modal.App()


@app.function()
def local_f():
    pass
