# Copyright Modal Labs 2022
import modal

stub = modal.Stub()

print(f"stub.description: {stub.description}")


@stub.function
def foo():
    pass
