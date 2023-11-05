# Copyright Modal Labs 2023
import asyncio
import contextlib
from asyncio import Future
from typing import TYPE_CHECKING, Dict, List, Optional, TypeVar

from modal_proto import api_pb2

if TYPE_CHECKING:
    from rich.spinner import Spinner
    from rich.tree import Tree
else:
    Spinner = TypeVar("Spinner")
    Tree = TypeVar("Tree")

from modal.exception import ExecutionError


class StatusRow:
    def __init__(self, progress: Optional[Tree]):
        from ._output import (
            step_progress,
        )

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
        from ._output import step_completed, step_progress_update

        if self._step_node is not None:
            step_progress_update(self._spinner, message)
            self._step_node.label = step_completed(message, is_substep=True)


class Resolver:
    # Unfortunately we can't use type annotations much in this file,
    # since that leads to circular dependencies
    _tree: Tree
    _local_uuid_to_future: Dict[str, Future]
    _environment_name: Optional[str]
    _app_id: Optional[str]

    def __init__(
        self,
        client=None,
        *,
        output_mgr=None,
        environment_name: Optional[str] = None,
        app_id: Optional[str] = None,
        shell: Optional[bool] = False,
    ):
        from rich.tree import Tree

        from ._output import step_progress

        self._output_mgr = output_mgr
        self._local_uuid_to_future = {}
        self._tree = Tree(step_progress("Creating objects..."), guide_style="gray50")
        self._client = client
        self._app_id = app_id
        self._environment_name = environment_name
        self._shell = shell

    @property
    def app_id(self) -> str:
        if self._app_id is None:
            raise ExecutionError("Resolver has no app")
        return self._app_id

    @property
    def client(self):
        return self._client

    @property
    def environment_name(self):
        return self._environment_name

    @property
    def shell(self):
        return self._shell

    async def preload(self, obj, existing_object_id: Optional[str]):
        if obj._preload is not None:
            await obj._preload(obj, self, existing_object_id)

    async def load(self, obj, existing_object_id: Optional[str] = None):
        cached_future = self._local_uuid_to_future.get(obj.local_uuid)

        if not cached_future:
            # don't run any awaits within this if-block to prevent race conditions
            async def loader():
                # Wait for all its dependencies
                # TODO(erikbern): do we need existing_object_id for those?
                await asyncio.gather(*[self.load(dep) for dep in obj.deps()])

                # Load the object itself
                await obj._load(obj, self, existing_object_id)
                if existing_object_id is not None and obj.object_id != existing_object_id:
                    # TODO(erikbern): ignoring images is an ugly fix to a problem that's on the server.
                    # Unlike every other object, images are not assigned random ids, but rather an
                    # id given by the hash of its contents. This means we can't _force_ an image to
                    # have a particular id. The better solution is probably to separate "images"
                    # from "image definitions" or something like that, but that's a big project.
                    #
                    # Persisted refs are ignored because their life cycle is managed independently.
                    # The same tag on an app can be pointed at different objects.
                    if not obj._is_another_app and not existing_object_id.startswith("im-"):
                        raise Exception(
                            f"Tried creating an object using existing id {existing_object_id}"
                            f" but it has id {obj.object_id}"
                        )

                return obj

            cached_future = asyncio.create_task(loader())
            self._local_uuid_to_future[obj.local_uuid] = cached_future

        if cached_future.done():
            return cached_future.result()

        return await cached_future

    def objects(self) -> List:
        for fut in self._local_uuid_to_future.values():
            if not fut.done():
                # this will raise an exception if not all loads have been awaited, but that *should* never happen
                raise RuntimeError(
                    "All loaded objects have not been resolved yet, can't get all objects for the resolver!"
                )
        return [fut.result() for fut in self._local_uuid_to_future.values()]

    @contextlib.contextmanager
    def display(self):
        from ._output import step_completed

        if self._output_mgr is None:
            yield
        else:
            with self._output_mgr.ctx_if_visible(self._output_mgr.make_live(self._tree)):
                yield
            self._tree.label = step_completed("Created objects.")
            self._output_mgr.print_if_visible(self._tree)

    def add_status_row(self) -> StatusRow:
        return StatusRow(self._tree)

    async def console_write(self, log: api_pb2.TaskLogs):
        if self._output_mgr is not None:
            await self._output_mgr.put_log_content(log)

    def console_flush(self):
        if self._output_mgr is not None:
            self._output_mgr.flush_lines()

    def image_snapshot_update(self, image_id: str, task_progress: api_pb2.TaskProgress):
        if self._output_mgr is not None:
            self._output_mgr.update_snapshot_progress(image_id, task_progress)
