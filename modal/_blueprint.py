from typing import Iterator, List, Tuple

from .object import Object


class Blueprint:
    """Contains a tag -> object mapping of objects.

    These objects are not created yet but may be created later."""

    _objects: List[Tuple[str, Object]]

    def __init__(self):
        self._objects = []

    def register(self, tag: str, obj: Object):
        """Registers an object to be created by the app so that it's available in modal.

        This is only used by factories and functions.
        """
        self._objects.append((tag, obj))

    def get_objects(self) -> Iterator[Tuple[str, Object]]:
        for obj in self._objects:
            yield obj
