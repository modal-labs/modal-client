# Copyright Modal Labs 2023
import typing
from typing import TypeVar

from modal.config import config

F = TypeVar("F", bound=typing.Callable[..., typing.Any])


def typechecked(f: F) -> F:
    # type checking function adds significant overhead when
    # the typeguard.typechecked is applied, so only do it locally
    # where it matters the most
    is_local = not bool(config.get("image_id"))
    if is_local:
        import typeguard

        typeguard.config.collection_check_strategy = typeguard.CollectionCheckStrategy.ALL_ITEMS
        return typeguard.typechecked(f)  # noqa
    else:
        return f
