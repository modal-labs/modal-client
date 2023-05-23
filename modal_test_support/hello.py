# Copyright Modal Labs 2023
import modal

stub = modal.Stub()


@stub.function()
def hello():
    print("hello")
    return "hello"


@stub.function()
def other():
    return "other"
