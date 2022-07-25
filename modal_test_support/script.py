from modal import Stub

stub = Stub()


@stub.function()
def foo():
    pass  # not actually used in test (servicer returns sum of square of all args)


if __name__ == "__main__":
    with stub.run():
        assert foo(2, 4) == 20
