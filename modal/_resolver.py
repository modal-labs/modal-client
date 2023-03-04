# Copyright Modal Labs 2023
import contextlib
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar

if TYPE_CHECKING:
    from rich.console import Console
    from rich.progress import Progress
    from rich.spinner import Spinner
    from rich.tree import Tree
else:
    Console = TypeVar("Console")
    Progress = TypeVar("Progress")
    Spinner = TypeVar("Spinner")
    Tree = TypeVar("Tree")

from modal.exception import ExecutionError


class StatusRow:
    def __init__(self, progress: Tree):
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
    _progress: Tree
    _local_uuid_to_object: Dict[str, Any]

    def __init__(self, output_mgr, client, app_id: Optional[str] = None):
        from ._output import step_progress
        from rich.tree import Tree

        self._output_mgr = output_mgr
        self._local_uuid_to_object = {}
        self._progress = Tree(step_progress("Creating objects..."), guide_style="gray50")

        # Accessible by objects
        self._client = client
        self._app_id = app_id

    @property
    def app_id(self) -> str:
        if self._app_id is None:
            raise ExecutionError("Resolver has no app")
        return self._app_id

    @property
    def client(self):
        return self._client

    async def load(self, obj, existing_object_id: Optional[str] = None):
        cached_obj = self._local_uuid_to_object.get(obj.local_uuid)
        if cached_obj is not None:
            # We already created this object before, shortcut this method
            return cached_obj

        created_obj = await obj._load(self, existing_object_id)

        if existing_object_id is not None and created_obj.object_id != existing_object_id:
            # TODO(erikbern): this is a very ugly fix to a problem that's on the server side.
            # Unlike every other object, images are not assigned random ids, but rather an
            # id given by the hash of its contents. This means we can't _force_ an image to
            # have a particular id. The better solution is probably to separate "images"
            # from "image definitions" or something like that, but that's a big project.
            if not existing_object_id.startswith("im-"):
                raise Exception(
                    f"Tried creating an object using existing id {existing_object_id}"
                    f" but it has id {created_obj.object_id}"
                )

        self._local_uuid_to_object[obj.local_uuid] = created_obj
        return created_obj

    @contextlib.contextmanager
    def display_progress(self):
        from ._output import step_completed

        with self._output_mgr.ctx_if_visible(self._output_mgr.make_live(self._progress)):
            yield
        self._progress.label = step_completed("Created objects.")
        self._output_mgr.print_if_visible(self._progress)

    def add_status_row(self) -> StatusRow:
        return StatusRow(self._progress)

    def get_console(self) -> Console:
        return self._output_mgr._console

    def objects(self) -> List:
        return list(self._local_uuid_to_object.values())

    def add_progress(self, description: str, total: float) -> Callable[[float], None]:
        """Creates a `rich.Progress` instance with custom columns for image snapshot progress."""
        from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TimeElapsedColumn

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeElapsedColumn(),
            console=self._output_mgr._console,
            transient=True,
        )
        self._progress.add(progress)
        task_id = progress.add_task(f"[yellow]{description}", total=total)

        def update(completed: float):
            if completed < total:
                progress.update(task_id, completed=completed)
            else:
                progress.remove_task(task_id)

        return update
