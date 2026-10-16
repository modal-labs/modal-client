# Copyright Modal Labs 2026
"""Classes and functions that require type translation."""

import typing

from synchronicity2 import Module

wrapper_module = Module("class_with_translation")


@wrapper_module.wrap_class()
class Node:
    """A simple node class for testing translation."""

    value: int

    def __init__(self, value: int):
        self.value = value

    async def create_child(self, child_value: int) -> "Node":
        """Create a child node."""
        return Node(child_value)

    async def get_children(self, count: int) -> "typing.AsyncGenerator[Node, None]":
        """Generate child nodes."""
        for i in range(count):
            yield Node(self.value + i)


@wrapper_module.wrap_function()
async def create_node(value: int) -> Node:
    """Create a new node."""
    return Node(value)


@wrapper_module.wrap_function()
async def connect_nodes(parent: Node, child: Node) -> typing.Tuple[Node, Node]:
    """Connect two nodes (just returns them as a tuple)."""
    return (parent, child)


@wrapper_module.wrap_function()
async def get_node_list(nodes: typing.List[Node]) -> typing.List[Node]:
    """Process a list of nodes."""
    return nodes


@wrapper_module.wrap_function()
async def get_optional_node(node: typing.Optional[Node]) -> typing.Optional[Node]:
    """Process an optional node."""
    return node


@wrapper_module.wrap_class()
class WrappedTypeInConstructor:
    def __init__(self, node: Node):
        assert isinstance(node, Node), "node should be unwrapped"
