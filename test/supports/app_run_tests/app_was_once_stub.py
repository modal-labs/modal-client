# Copyright Modal Labs 2024
import modal

# Trigger the deprecation warning for the class itself,
# but not the symbol name (see app_was_once_stub_2 for the opposite)
app = modal.Stub()


@app.function()
def foo():
    print("foo")
