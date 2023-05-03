# Copyright Modal Labs 2023
import modal

a_stub = modal.Stub()
b_stub = modal.Stub()


if a_stub.is_inside():
    print("inside a")


if b_stub.is_inside():
    print("inside b")


@a_stub.function()
def foo(i):
    pass
