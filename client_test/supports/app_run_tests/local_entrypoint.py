# Copyright Modal Labs 2022

import modal

stub = modal.Stub()


@stub.function
def foo():
    pass


@stub.local_entrypoint
def main():
    print("called locally")
    foo.call()
    foo.call()
