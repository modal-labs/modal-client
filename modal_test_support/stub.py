# Copyright Modal Labs 2022
import modal

stub = modal.Stub()


@stub.function
def f(x):
    # not actually used in test (servicer returns sum of square of all args)
    pass
