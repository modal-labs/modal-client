# Copyright Modal Labs 2023
import modal


stub = modal.Stub("dummy")


def foo(i):
    return 1


foo_handle = stub.function()(foo)


other_stub = modal.Stub("dummy")


@other_stub.function()
def bar(i):
    return 2
