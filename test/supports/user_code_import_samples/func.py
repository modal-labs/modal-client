# Copyright Modal Labs 2024

from modal import App

# TODO: remove include_source=True when automount is disabled by default

print("IMPORT", __name__)
app = App(name="user_code_import_samples_func_app", include_source=True)


@app.function()
def f(arg):
    return f"hello {arg}"


def undecorated_f(arg):
    return f"hello {arg}"
