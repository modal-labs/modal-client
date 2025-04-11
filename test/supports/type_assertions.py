# Copyright Modal Labs 2024
import typing

from typing_extensions import assert_type

import modal
from modal.partial_function import method

app = modal.App()


@app.function()
def typed_func(a: str) -> float:
    return 0.0


@app.function()
def other_func() -> str:
    return "foo"


ret = typed_func.remote(a="hello")
assert_type(ret, float)

ret2 = modal.FunctionCall.gather(typed_func.spawn("bar"), other_func.spawn())
# This assertion doesn't work in mypy (it infers the more generic list[object]), but does work in pyright/vscode:
# assert_type(ret2, typing.List[typing.Union[float, str]])
mypy_compatible_ret: typing.Sequence[object] = ret2  # mypy infers to the broader "object" type instead


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


# check sandboxes
sandbox = modal.Sandbox.create("dummy")
assert_type(sandbox.stdout.read(), str)

for line_str in sandbox.stdout:
    assert_type(line_str, str)

cmd = sandbox.exec("other")
assert_type(cmd.stdout.read(), str)

for line_str in cmd.stdout:
    assert_type(line_str, str)

cmd2 = sandbox.exec("other_bin", text=False)
assert_type(cmd2.stdout.read(), bytes)

for line_bytes in cmd2.stdout:
    assert_type(line_bytes, bytes)

# check file_io
file_io = sandbox.open("foo", "w")
assert_type(file_io.read(), str)
assert_type(file_io.readline(), str)
assert_type(file_io.readlines(), typing.Sequence[str])

file_io2 = sandbox.open("foo", "rb")
assert_type(file_io2.read(), bytes)
assert_type(file_io2.readline(), bytes)
assert_type(file_io2.readlines(), typing.Sequence[bytes])
