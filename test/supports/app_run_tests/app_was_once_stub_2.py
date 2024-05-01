# Copyright Modal Labs 2024
import modal

# Don't trigger the deprecation warning for the class itself,
# but there should be a separate deprecation warning because of the symbol name
stub = modal.App()


@stub.function()
def foo():
    print("foo")
