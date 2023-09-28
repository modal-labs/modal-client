# Copyright Modal Labs 2022
import modal

stub = modal.Stub()

# This is in module scope, so will show what the `description`
# value is at import time, which may be different if some code
# changes the `description` post-import.
print(f"stub.description: {stub.description}")


@stub.function()
def foo():
    pass
