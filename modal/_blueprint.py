from typing import Dict, Iterator, Tuple

from .object import Object


class Blueprint:
    """Contains a tag -> object mapping of objects.

    These objects are not created yet but may be created later."""

    _objects: Dict[str, Object]

    def __init__(self):
        self._objects = {}

    def register(self, tag: str, obj: Object):
        """Registers an object to be created by the app so that it's available in modal.

        This is only used by factories and functions.
        """
        self._objects[tag] = obj

    def get_object(self, tag: str) -> Object:
        return self._objects[tag]

    def get_objects(self) -> Iterator[Tuple[str, Object]]:
        for tag, obj in self._objects.items():
            yield tag, obj
