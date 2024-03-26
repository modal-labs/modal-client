# Copyright Modal Labs 2024
import ast  # noqa

import modal

stub = modal.Stub("imports_ast")


@stub.function()
def some_func():
    pass
