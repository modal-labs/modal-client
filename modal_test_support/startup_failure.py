# Copyright Modal Labs 2022
import modal

stub = modal.Stub("hello-world")

if not modal.is_local():
    import nonexistent_package  # noqa


@stub.function
def f(i):
    pass
