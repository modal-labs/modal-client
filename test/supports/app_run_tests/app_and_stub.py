# Copyright Modal Labs 2024
import modal

# If both `app` and `stub` is present, we want Modal to fall back to `stub`,
# rather than complaining about the type of `app`
app = 123

stub = modal.App()


@stub.function()
def foo():
    print("foo")
