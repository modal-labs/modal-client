# Copyright Modal Labs 2023
import modal

stub = modal.Stub()


def foo(i):
    return 1


foo_handle = stub.function()(foo)  #  "privately" decorated, by not override the original function


other_stub = modal.Stub()


@other_stub.function()
def bar(i):
    return 2
