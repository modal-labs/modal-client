# Copyright Modal Labs 2023
import asyncio
import contextlib
from asyncio import Future
from typing import TYPE_CHECKING, Dict, Hashable, List, Optional

from grpclib import GRPCError, Status

from modal_proto import api_pb2

from ._utils.async_utils import TaskContext
from .client import _Client
from .exception import NotFoundError

if TYPE_CHECKING:
    from rich.tree import Tree

    from modal.object import _Object


class StatusRow:
    def __init__(self, progress: "Optional[Tree]"):
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
    _local_uuid_to_future: Dict[str, Future]
    _environment_name: Optional[str]
    _app_id: Optional[str]
    _deduplication_cache: Dict[Hashable, Future]
    _client: _Client

    def __init__(
        self,
        client=None,
        *,
        output_mgr=None,
        environment_name: Optional[str] = None,
        app_id: Optional[str] = None,
    ):
        from rich.tree import Tree

        from ._output import step_progress

        self._output_mgr = output_mgr
        self._local_uuid_to_future = {}
        self._tree = Tree(step_progress("Creating objects..."), guide_style="gray50")
        self._client = client
        self._app_id = app_id
        self._environment_name = environment_name
        self._deduplication_cache = {}

    @property
    def app_id(self) -> Optional[str]:
        return self._app_id

    @property
    def client(self):
        return self._client

    @property
    def environment_name(self):
        return self._environment_name

    async def preload(self, obj, existing_object_id: Optional[str]):
        if obj._preload is not None:
            await obj._preload(obj, self, existing_object_id)

    async def load(self, obj: "_Object", existing_object_id: Optional[str] = None):
        if obj._is_hydrated and obj._is_another_app:
            # No need to reload this, it won't typically change
            if obj.local_uuid not in self._local_uuid_to_future:
                # a bit dumb - but we still need to store a reference to the object here
                # to be able to include all referenced objects when setting up the app
                fut: Future = Future()
                fut.set_result(obj)
                self._local_uuid_to_future[obj.local_uuid] = fut
            return obj

        deduplication_key: Optional[Hashable] = None
        if obj._deduplication_key:
            deduplication_key = await obj._deduplication_key()

        cached_future = self._local_uuid_to_future.get(obj.local_uuid)

        if not cached_future and deduplication_key is not None:
            # deduplication cache makes sure duplicate mounts are resolved only
            # once, even if they are different instances - as long as they have
            # the same content
            cached_future = self._deduplication_cache.get(deduplication_key)
            if cached_future:
                hydrated_object = await cached_future
                obj._hydrate(hydrated_object.object_id, self._client, hydrated_object._get_metadata())
                return obj

        if not cached_future:
            # don't run any awaits within this if-block to prevent race conditions
            async def loader():
                # Wait for all its dependencies
                # TODO(erikbern): do we need existing_object_id for those?
                await TaskContext.gather(*[self.load(dep) for dep in obj.deps()])

                # Load the object itself
                try:
                    await obj._load(obj, self, existing_object_id)
                except GRPCError as exc:
                    if exc.status == Status.NOT_FOUND:
                        raise NotFoundError(exc.message)
                    raise

                # Check that the id of functions and classes didn't change
                # TODO(erikbern): revisit this once stub assignments have been disallowed
                if not obj._is_another_app and (obj.object_id.startswith("fu-") or obj.object_id.startswith("cs-")):
                    # Persisted refs are ignored because their life cycle is managed independently.
                    # The same tag on an app can be pointed at different objects.
                    if existing_object_id is not None and obj.object_id != existing_object_id:
                        raise Exception(
                            f"Tried creating an object using existing id {existing_object_id}"
                            f" but it has id {obj.object_id}"
                        )

                return obj

            cached_future = asyncio.create_task(loader())
            self._local_uuid_to_future[obj.local_uuid] = cached_future
            if deduplication_key is not None:
                self._deduplication_cache[deduplication_key] = cached_future

        return await cached_future

    def objects(self) -> List["_Object"]:
        unique_objects: Dict[str, "_Object"] = {}
        for fut in self._local_uuid_to_future.values():
            if not fut.done():
                # this will raise an exception if not all loads have been awaited, but that *should* never happen
                raise RuntimeError(
                    "All loaded objects have not been resolved yet, can't get all objects for the resolver!"
                )
            obj = fut.result()
            unique_objects.setdefault(obj.object_id, obj)
        return list(unique_objects.values())

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
