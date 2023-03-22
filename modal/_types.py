# Copyright Modal Labs 2023
from typing import TypeVar

from modal.config import config

F = TypeVar("F")


def typechecked(f: F) -> F:
    # type checking function adds significant overhead when
    # the typeguard.typechecked is applied, so only do it locally
    # where it matters the most
    is_local = not bool(config.get("image_id"))
    if is_local:
        from typeguard import typechecked

        return typechecked(f)
    else:
        return f
