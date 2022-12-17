# Copyright Modal Labs 2023
import modal

stub = modal.Stub()


@stub.webhook(method="DELETE")
def dummy():
    pass
