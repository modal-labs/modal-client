# Copyright Modal Labs 2022
import modal

stub = modal.Stub("a")
other = modal.Stub("b")


def builder_function():
    print("ran builder function")


image = modal.Image.debian_slim().run_function(builder_function)


@stub.function(image=image)
def foo():
    pass
