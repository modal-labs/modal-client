# Copyright Modal Labs 2022
import modal
import modal.functions

stub = modal.Stub()


@stub.function
def square(x):
    return x**2


assert isinstance(square, modal.functions.FunctionHandle)

# This should fail in a container
with stub.run():
    print(square(42))
