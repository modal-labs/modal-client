# Copyright Modal Labs 2024
import modal

stub = modal.Stub("c")


@stub.function()
def c_func():
    pass
