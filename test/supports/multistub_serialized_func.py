# Copyright Modal Labs 2023
import modal

stub = modal.Stub()


def foo(i):
    return 1


foo_handle = stub.function(serialized=True)(foo)


other_stub = modal.Stub()


@other_stub.function()
def bar(i):
    return 2
