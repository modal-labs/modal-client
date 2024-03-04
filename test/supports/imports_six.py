import six  # noqa

import modal

stub = modal.Stub("imports_six")


@stub.function()
def some_func():
    pass
