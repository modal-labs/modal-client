import c

import modal

stub = modal.Stub()

d = modal.Dict.from_name("my-queue", create_if_missing=True)


@stub.function()
def a_func():
    d["foo"] = "bar"


stub.include(c.stub)
