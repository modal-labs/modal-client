# Copyright Modal Labs 2026
"""Integration support for Sequence translation and Callable ellipsis emission."""

import typing

from synchronicity2 import Module

mod = Module("sequence_callable_annotations")


@mod.wrap_class()
class Node:
    value: int

    def __init__(self, value: int):
        self.value = value


@mod.wrap_function()
async def clone_all(nodes: typing.Sequence[Node]) -> typing.Sequence[Node]:
    return [Node(node.value) for node in nodes]


@mod.wrap_function()
def make_callback(node: Node) -> typing.Callable[..., typing.Sequence[Node]]:
    return lambda *args, **kwargs: [node]


@mod.wrap_function()
def make_callback2() -> typing.Callable[[Node], typing.Sequence[Node]]:
    return lambda n: [n]
