# Copyright Modal Labs 2022
import modal

stub = modal.Stub()


@stub.function
@stub.web_endpoint()
def foo():
    return {"bar": "baz"}
