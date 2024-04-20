# Copyright Modal Labs 2024
import modal

stub = modal.Stub()


@stub.function()
def foo():
    print("foo")
