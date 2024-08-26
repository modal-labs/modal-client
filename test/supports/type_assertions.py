# Copyright Modal Labs 2024
from typing_extensions import assert_type

import modal

app = modal.App()


@app.function()
def typed_func(a: str) -> float:
    return 0.0


should_be_float = typed_func.remote(a="hello")
assert_type(should_be_float, float)


@app.function()
async def async_typed_func(b: bool) -> str:
    return ""


async def async_block() -> None:
    should_be_str = async_typed_func.remote(False)
    assert_type(should_be_str, str)

    should_be_str_2 = async_typed_func.remote.aio(True)
    assert_type(should_be_str_2, str)
