import modal

stub = modal.Stub()


@stub.function
def foo():
    pass


@stub.function
def bar():
    pass
