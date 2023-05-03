# Copyright Modal Labs 2023
import modal

stub = modal.Stub()

if stub.is_inside():
    print("in container!")
else:
    print("in local!")


@stub.function()
def foo(i):
    pass
