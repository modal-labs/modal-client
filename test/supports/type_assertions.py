# Copyright Modal Labs 2024
from typing_extensions import assert_type

import modal
from modal.partial_function import method

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


@app.cls()
class Cls:
    @method()
    def foo(self, a: str) -> int:
        return 1

    @method()
    async def bar(self, a: str) -> int:
        return 1


instance = Cls()
should_be_int = instance.foo.remote("foo")
assert_type(should_be_int, int)

should_be_int = instance.bar.remote("bar")
assert_type(should_be_int, int)


async def async_block() -> None:
    should_be_str_2 = await async_typed_func.remote.aio(True)
    assert_type(should_be_str_2, str)
    should_also_be_str = await async_typed_func.local(False)  # local should be the original return type (!)
    assert_type(should_also_be_str, str)
    should_be_int = await instance.bar.local("bar")
    assert_type(should_be_int, int)
