# Copyright Modal Labs 2024
import modal

stub = modal.Stub()


@stub.function(gpu="NOT_A_GPU")
def f():
    pass
