# Copyright Modal Labs 2026
import callback_translation_impl
from callback_translation import Node, apply_to_node, listify, map_node_to_int
from typing_extensions import assert_type


def clone_node(node: callback_translation_impl.Node) -> callback_translation_impl.Node:
    return callback_translation_impl.Node(node.value)


def read_node(node: callback_translation_impl.Node) -> int:
    return node.value


def make_node(x: int) -> callback_translation_impl.Node:
    return callback_translation_impl.Node(x)


node = Node(1)

assert_type(apply_to_node(node, clone_node), Node)
assert_type(map_node_to_int(node, read_node), int)

listified = listify(make_node)
assert_type(listified(3), list[Node])
