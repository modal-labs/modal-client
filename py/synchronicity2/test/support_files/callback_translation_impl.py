# Copyright Modal Labs 2026
import typing

import synchronicity2

P = typing.ParamSpec("P")

mod = synchronicity2.Module("callback_translation")


@mod.wrap_class()
class Node:
    value: int

    def __init__(self, value: int):
        self.value = value


@mod.wrap_function()
async def apply_to_node(node: Node, callback: typing.Callable[[Node], Node]) -> Node:
    return callback(node)


@mod.wrap_function()
async def map_node_to_int(node: Node, callback: typing.Callable[[Node], int]) -> int:
    return callback(node)


@mod.wrap_function()
def listify(c: typing.Callable[P, Node]) -> typing.Callable[P, list[Node]]:
    def c2(*args: P.args, **kwargs: P.kwargs):
        return [c(*args, **kwargs)]

    return c2
