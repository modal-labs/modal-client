# Copyright Modal Labs 2026
"""Consumer typing checks for class_with_translation (sync, async, keyword usage)."""

from class_with_translation import Node, WrappedTypeInConstructor, connect_nodes, create_node
from typing_extensions import assert_type


def _sync_usage() -> None:
    node = create_node(42)
    assert_type(node, Node)
    node_kw = create_node(value=100)
    assert_type(node_kw, Node)

    child = node.create_child(100)
    assert_type(child, Node)

    parent = Node(1)
    child_node = Node(2)
    result = connect_nodes(parent, child_node)
    assert_type(result, tuple[Node, Node])

    result_kw = connect_nodes(parent=parent, child=child_node)
    assert_type(result_kw, tuple[Node, Node])

    _ = WrappedTypeInConstructor(create_node(10))


async def _async_usage() -> None:
    node2 = await create_node.aio(42)
    assert_type(node2, Node)

    child2 = await node2.create_child.aio(100)
    assert_type(child2, Node)

    parent2 = await create_node.aio(1)
    child_node2 = await create_node.aio(2)
    result2 = await connect_nodes.aio(parent2, child_node2)
    assert_type(result2, tuple[Node, Node])
