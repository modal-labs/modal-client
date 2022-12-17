# Copyright Modal Labs 2023
import modal

stub = modal.Stub()


@stub.webhook(method="GET")
def dummy():
    pass
