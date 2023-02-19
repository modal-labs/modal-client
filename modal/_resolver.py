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

    def __init__(self, app, progress: Optional[Tree], client, app_id: str):
        self._app = app
        self._progress = progress

        # Accessible by objects
        self.client = client
        self.app_id = app_id

    async def load(self, obj) -> str:
        # assert isinstance(obj, Provider)
        created_obj = await self._app._load(obj, progress=self._progress)
        # assert isinstance(created_obj, Handle)
        return created_obj.object_id

    def add_status_row(self) -> StatusRow:
        return StatusRow(self._progress)
