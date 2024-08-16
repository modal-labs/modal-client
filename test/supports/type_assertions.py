# Copyright Modal Labs 2024
from typing_extensions import assert_type

import modal

app = modal.App()


@app.function()
def typed_func(a: str) -> float:
    return 0.0


ret = typed_func.remote(a="hello")
assert_type(ret, float)
