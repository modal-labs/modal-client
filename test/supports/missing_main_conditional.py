# Copyright Modal Labs 2022
import modal

stub = modal.Stub()


@stub.function()
def square(x):
    return x**2


# This should fail in a container
with stub.run():
    print(square(42))
