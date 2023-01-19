# Copyright Modal Labs 2022
import modal

stub = modal.Stub()


@stub.webhook
def foo():
    return {"bar": "baz"}
