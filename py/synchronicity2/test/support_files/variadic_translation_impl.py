# Copyright Modal Labs 2026
"""Integration support for translated *args/**kwargs on class/static methods."""

import asyncio

from synchronicity2 import Module

wrapper_module = Module("variadic_translation")


@wrapper_module.wrap_class()
class Node:
    def __init__(self, name: str):
        self.name = name


@wrapper_module.wrap_class()
class Collector:
    @staticmethod
    async def static_collect(*nodes: Node, **named_nodes: Node) -> tuple[list[str], dict[str, str]]:
        await asyncio.sleep(0.01)
        return [node.name for node in nodes], {key: value.name for key, value in named_nodes.items()}

    @classmethod
    async def class_collect(cls, *nodes: Node, **named_nodes: Node) -> tuple[list[str], dict[str, str]]:
        await asyncio.sleep(0.01)
        return [node.name for node in nodes], {key: value.name for key, value in named_nodes.items()}
