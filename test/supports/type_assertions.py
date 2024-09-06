# Copyright Modal Labs 2024
import typing

from typing_extensions import assert_type

import modal

app = modal.App()


@app.function()
def typed_func(a: str) -> float:
    return 0.0


@app.function()
def other_func() -> str:
    return "foo"


ret = typed_func.remote(a="hello")
assert_type(ret, float)

ret2 = modal.functions.gather(typed_func.spawn("bar"), other_func.spawn())
# This assertion doesn't work in mypy (it infers the more generic list[object]), but does work in pyright/vscode:
# assert_type(ret2, typing.List[typing.Union[float, str]])
mypy_compatible_ret: typing.Sequence[object] = ret2  # mypy infers to the broader "object" type instead
