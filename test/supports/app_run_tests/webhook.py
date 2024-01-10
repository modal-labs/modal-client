# Copyright Modal Labs 2022
from modal import Stub, web_endpoint

stub = Stub()


@stub.function()
@web_endpoint()
def foo():
    return {"bar": "baz"}
