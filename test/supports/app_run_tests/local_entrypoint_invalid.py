# Copyright Modal Labs 2023
import modal

stub = modal.Stub()


@stub.function()
def foo():
    pass


@stub.local_entrypoint()
def main():
    with stub.run():  # should error here
        print("unreachable")
        foo.call()  # should not get here
