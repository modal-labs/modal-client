import modal

stub = modal.Stub()


@stub.function()
def hello():
    print("hello")
    return "hello"
