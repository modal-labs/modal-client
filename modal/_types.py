# Copyright Modal Labs 2023
import typing
from typing import TypeVar

from modal.config import config

F = TypeVar("F", bound=typing.Callable[..., typing.Any])


def typechecked(f: F) -> F:
    is_local = not bool(config.get("image_id"))
    if is_local:
        # TODO: add back run-time type checking based on signature
        # The `typeguard` package was too slow at import time, adding roughly 1s latency due to function recompilation etc.
        return f
    else:
        return f
