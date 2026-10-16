# Copyright Modal Labs 2026
"""Consumer typing checks for Sequence annotations."""

import typing

import sequence_callable_annotations
from typing_extensions import assert_type

nodes = [sequence_callable_annotations.Node(1), sequence_callable_annotations.Node(2)]
cloned = sequence_callable_annotations.clone_all(nodes)
assert_type(cloned, typing.Sequence[sequence_callable_annotations.Node])

callback = sequence_callable_annotations.make_callback(nodes[0])
assert_type(callback, typing.Callable[..., typing.Sequence[sequence_callable_annotations.Node]])
