from typing import TypeVar

F = TypeVar("F")


def typechecked(f: F) -> F:
    # type checking function adds significant overhead when
    # the typeguard.typechecked is applied, so only do it locally
    # where it matters the most
    from modal.app import is_local

    if is_local():
        from typeguard import typechecked

        return typechecked(f)
    else:
        return f
