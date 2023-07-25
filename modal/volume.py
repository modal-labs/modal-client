# Copyright Modal Labs 2023
import asyncio
from typing import AsyncIterator, List, Optional

from modal_proto import api_pb2
from modal_utils.async_utils import asyncnullcontext, synchronize_api
from modal_utils.grpc_utils import retry_transient_errors, unary_stream

from ._resolver import Resolver
from .object import _Handle, _Provider


class _VolumeHandle(_Handle, type_prefix="vo"):
    """mdmd:hidden Handle to a `Volume` object."""

    _lock: asyncio.Lock

    def _initialize_from_empty(self):
        # To (mostly*) prevent multiple concurrent operations on the same volume, which can cause problems under
        # some unlikely circumstances.
        # *: You can bypass this by creating multiple handles to the same volume, e.g. via lookup. But this
        # covers the typical case = good enough.
        self._lock = asyncio.Lock()

    async def commit(self):
        """Commit changes to the volume and fetch any other changes made to the volume by other tasks.

        Committing always triggers a reload after saving changes.

        If successful, the changes made are now persisted in durable storage and available for other functions/tasks.

        Committing will fail if there are open files for the volume.
        """
        async with self._lock:
            req = api_pb2.VolumeCommitRequest(volume_id=self.object_id)
            _ = await retry_transient_errors(self._client.stub.VolumeCommit, req)
            # Reload changes on successful commit.
            await self._do_reload(lock=False)

    async def reload(self):
        """Make changes made by other tasks/functions visible in the volume.

        Uncommitted changes to the volume, such as new or modified files, will be preserved during reload. Uncommitted
        changes will shadow any changes made by other tasks - e.g. if you have an uncommitted modified a file that was
        also updated by another task/function you will not see the changes made by the other function/task.

        Reloading will fail if there are open files for the volume.
        """
        await self._do_reload()

    async def iterdir(self, path: str) -> AsyncIterator[api_pb2.VolumeListFilesEntry]:
        """Iterate over all files in a directory in the volume.

        * Passing a directory path lists all files in the directory (names are relative to the directory)
        * Passing a file path returns a list containing only that file's listing description
        * Passing a glob path (including at least one * or ** sequence) returns all files matching that glob path (using absolute paths)
        """
        req = api_pb2.VolumeListFilesRequest(volume_id=self._object_id, path=path)
        async for batch in unary_stream(self._client.stub.VolumeListFiles, req):
            for entry in batch.entries:
                yield entry

    async def listdir(self, path: str) -> List[api_pb2.VolumeListFilesEntry]:
        """List all files under a path prefix in the modal.Volume.

        * Passing a directory path lists all files in the directory
        * Passing a file path returns a list containing only that file's listing description
        * Passing a glob path (including at least one * or ** sequence) returns all files matching that glob path (using absolute paths)
        """
        return [entry async for entry in self.iterdir(path)]

    async def _do_reload(self, lock=True):
        async with self._lock if lock else asyncnullcontext():
            req = api_pb2.VolumeReloadRequest(volume_id=self.object_id)
            _ = await retry_transient_errors(self._client.stub.VolumeReload, req)


VolumeHandle = synchronize_api(_VolumeHandle)


class _Volume(_Provider[_VolumeHandle]):
    """mdmd:hidden A writeable volume that can be used to share files between one or more Modal functions.

    The contents of a volume is exposed as a filesystem. You can use it to share data between different functions, or
    to persist durable state across several instances of the same function.

    Unlike a networked filesystem, you need to explicitly reload the volume to see changes made since it was mounted.
    Similarly, you need to explicitly commit any changes you make to the volume for the changes to become visible
    outside the current task.

    Concurrent modification is supported, but concurrent modifications of the same files should be avoided! Last write
    wins in case of concurrent modification of the same file - any data the last writer didn't have when committing
    changes will be lost!

    As a result, volumes are typically not a good fit for use cases where you need to make concurrent modifications to
    the same file (nor is distributed file locking supported).

    Volumes can only be committed and reloaded if there are no open files for the volume - attempting to reload or
    commit with open files will result in an error.

    **Usage**

    ```python
    import modal

    stub = modal.Stub()
    stub.volume = modal.Volume.new()

    @stub.function(volumes={"/root/foo": stub.volume})
    def f():
        with open("/root/foo/bar.txt", "w") as f:
            f.write("hello")
        stub.app.volume.commit()  # Persist changes

    @stub.function(volumes={"/root/foo": stub.volume})
    def g():
        stub.app.volume.reload()  # Fetch latest changes
        with open("/root/foo/bar.txt", "r") as f:
            print(f.read())
    ```
    """

    @staticmethod
    def new() -> "_Volume":
        """Construct a new volume, which is empty by default."""

        async def _load(resolver: Resolver, existing_object_id: Optional[str], handle: _VolumeHandle):
            status_row = resolver.add_status_row()
            if existing_object_id:
                # Volume already exists; do nothing.
                handle._hydrate(existing_object_id, resolver.client, None)
                return

            status_row.message("Creating volume...")
            req = api_pb2.VolumeCreateRequest(app_id=resolver.app_id)
            resp = await retry_transient_errors(resolver.client.stub.VolumeCreate, req)
            status_row.finish("Created volume.")
            handle._hydrate(resp.volume_id, resolver.client, None)

        return _Volume._from_loader(_load, "Volume()")

    @staticmethod
    def persisted(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
    ) -> "_Volume":
        """Deploy a Modal app containing this object. This object can then be imported from other apps using
        the returned reference, or by calling `modal.Volume.from_name(label)` (or the equivalent method
        on respective class).

        **Example Usage**

        ```python
        import modal

        volume = modal.Volume.persisted("my-volume")

        stub = modal.Stub()

        # Volume refers to the same object, even across instances of `stub`.
        @stub.function(volumes={"/vol": volume})
        def f():
            pass
        ```

        """
        return _Volume.new()._persist(label, namespace, environment_name)

    # Methods on live handles

    async def commit(self):
        return await self._handle.commit()

    async def reload(self):
        return await self._handle.reload()

    async def iterdir(self, path: str) -> AsyncIterator[api_pb2.VolumeListFilesEntry]:
        async for entry in self._handle.iterdir(path):
            yield entry

    async def listdir(self, path: str) -> List[api_pb2.VolumeListFilesEntry]:
        return self._handle.listdir(path)


Volume = synchronize_api(_Volume)
