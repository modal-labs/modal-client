# Copyright Modal Labs 2024
import c

import modal

stub = modal.Stub()


@stub.function(secrets=[modal.Secret.from_dict({"foo": "bar"})])
def b_func():
    pass


stub.include(c.stub)
