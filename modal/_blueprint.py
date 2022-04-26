from typing import List

from .object import Object


class Blueprint:
    """Contains a tag -> object mapping of objects.

    These objects are not created yet but may be created later."""

    _objects: List[Object]

    def __init__(self):
        self._objects = []

    def register(self, obj):
        """Registers an object to be created by the app so that it's available in modal.

        This is only used by factories and functions.
        """
        if obj.tag is None:
            raise Exception("Can only register named objects")

        self._objects.append(obj)

    def get_objects(self):
        for obj in self._objects:
            yield obj
