from typing import List

from .object import Object


class Blueprint:
    """Contains a tag -> object mapping of objects.

    These objects are not created yet but may be created later."""

    _pending_tagged_objects: List[Object]

    def __init__(self):
        self._pending_tagged_objects = []

    def register(self, obj):
        """Registers an object to be created by the app so that it's available in modal.

        This is only used by factories and functions.
        """
        if obj.tag is None:
            raise Exception("Can only register named objects")

        self._pending_tagged_objects.append(obj)

    def get_objects(self):
        for obj in self._pending_tagged_objects:
            yield obj
