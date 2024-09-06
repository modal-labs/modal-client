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


async_typed_func

should_be_str = async_typed_func.remote(False)  # should be blocking without aio
assert_type(should_be_str, str)


async def async_block() -> None:
    should_be_str_2 = await async_typed_func.remote.aio(True)
    assert_type(should_be_str_2, str)
    should_also_be_str = await async_typed_func.local(False)  # local should be the original return type (!)
    assert_type(should_also_be_str, str)
