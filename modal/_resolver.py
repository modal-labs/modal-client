# Copyright Modal Labs 2023
from typing import TYPE_CHECKING, Optional, TypeVar

if TYPE_CHECKING:
    from rich.spinner import Spinner
    from rich.tree import Tree
else:
    Spinner = TypeVar("Spinner")
    Tree = TypeVar("Tree")


class StatusRow:
    def __init__(self, progress):
        from ._output import step_progress  # Lazy import to only import `rich` when necessary.

        self._spinner = None
        self._step_node = None
        if progress is not None:
            self._spinner = step_progress()
            self._step_node = progress.add(self._spinner)

    def message(self, message):
        from ._output import step_progress_update

        if self._spinner is not None:
            step_progress_update(self._spinner, message)

    def finish(self, message):
        from ._output import step_progress_update, step_completed

        if self._step_node is not None:
            step_progress_update(self._spinner, message)
            self._step_node.label = step_completed(message, is_substep=True)


class Resolver:
    # Unfortunately we can't use type annotations much in this file,
    # since that leads to circular dependencies
    _progress: Optional[Tree]

    def __init__(self, progress: Optional[Tree], client, app_id: Optional[str] = None):
        self._progress = progress
        self._local_uuid_to_object = {}

        # Accessible by objects
        self.client = client
        self.app_id = app_id

    async def load(self, obj, existing_object_id: Optional[str] = None):
        cached_obj = self._local_uuid_to_object.get(obj.local_uuid)
        if cached_obj is not None:
            # We already created this object before, shortcut this method
            return cached_obj
        created_obj = await obj._load(self, existing_object_id)
        self._local_uuid_to_object[obj.local_uuid] = created_obj
        return created_obj

    def add_status_row(self) -> StatusRow:
        return StatusRow(self._progress)
