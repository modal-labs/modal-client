# Copyright Modal Labs 2023
from typing import Awaitable, Callable, Optional


class Resolver:
    _app: "_App"

    client: "_Client"
    app_id: str
    existing_object_id: Optional[str]
    stub: "_Stub"

    def __init__(
        self,
        stub: "_Stub",
        app: "_App",
        progress: Optional["Tree"],
        client: "_Client",
        app_id: str,
        existing_object_id: Optional[str],
    ):
        self._app = app
        self._progress = progress
        self._last_message = None
        self._spinner = None
        self._step_node = None

        # Accessible by objects
        self.stub = stub
        self.client = client
        self.app_id = app_id
        self.existing_object_id = existing_object_id

    async def load(self, obj: "Provider") -> str:
        # assert isinstance(obj, Provider)
        created_obj = await self._app._load(obj, progress=self._progress)
        # assert isinstance(created_obj, Handle)
        return created_obj.object_id

    def set_message(self, message):
        from ._output import (  # Lazy import to only import `rich` when necessary.
            step_progress,
            step_progress_update,
        )

        self._last_message = message
        if self._progress:
            if self._step_node is None:
                self._spinner = step_progress()
                self._step_node = self._progress.add(self._spinner)
            step_progress_update(self._spinner, message)

    def set_finish(self):
        # Change message to a completed step
        # TODO: make this a context mgr __exit__ ?
        from ._output import step_completed

        if self._progress and self._last_message:
            self._step_node.label = step_completed(self._last_message, is_substep=True)
