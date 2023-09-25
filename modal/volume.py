# Copyright Modal Labs 2023
import asyncio
import time
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, List, Optional, Union

from modal_proto import api_pb2
from modal_utils.async_utils import ConcurrencyPool, asyncnullcontext, synchronize_api
from modal_utils.grpc_utils import retry_transient_errors, unary_stream

from ._blob_utils import blob_iter, blob_upload_file, get_file_upload_spec
from ._resolver import Resolver
from .config import logger
from .mount import MOUNT_PUT_FILE_CLIENT_TIMEOUT
from .object import _Object, live_method, live_method_gen


class _Volume(_Object, type_prefix="vo"):
    """A writeable volume that can be used to share files between one or more Modal functions.

    The contents of a volume is exposed as a filesystem. You can use it to share data between different functions, or
    to persist durable state across several instances of the same function.

    Unlike a networked filesystem, you need to explicitly reload the volume to see changes made since it was mounted.
    Similarly, you need to explicitly commit any changes you make to the volume for the changes to become visible
    outside the current container.

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

    _lock: asyncio.Lock

    def _initialize_from_empty(self):
        # To (mostly*) prevent multiple concurrent operations on the same volume, which can cause problems under
        # some unlikely circumstances.
        # *: You can bypass this by creating multiple handles to the same volume, e.g. via lookup. But this
        # covers the typical case = good enough.
        self._lock = asyncio.Lock()

    @staticmethod
    def new() -> "_Volume":
        """Construct a new volume, which is empty by default."""

        async def _load(provider: _Volume, resolver: Resolver, existing_object_id: Optional[str]):
            status_row = resolver.add_status_row()
            if existing_object_id:
                # Volume already exists; do nothing.
                provider._hydrate(existing_object_id, resolver.client, None)
                return

            status_row.message("Creating volume...")
            req = api_pb2.VolumeCreateRequest(app_id=resolver.app_id)
            resp = await retry_transient_errors(resolver.client.stub.VolumeCreate, req)
            status_row.finish("Created volume.")
            provider._hydrate(resp.volume_id, resolver.client, None)

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

    @live_method
    async def _do_reload(self, lock=True):
        async with self._lock if lock else asyncnullcontext():
            req = api_pb2.VolumeReloadRequest(volume_id=self.object_id)
            _ = await retry_transient_errors(self._client.stub.VolumeReload, req)

    @live_method
    async def commit(self):
        """Commit changes to the volume and fetch any other changes made to the volume by other containers.

        Committing always triggers a reload after saving changes.

        If successful, the changes made are now persisted in durable storage and available to other containers accessing the volume.

        Committing will fail if there are open files for the volume.
        """
        async with self._lock:
            req = api_pb2.VolumeCommitRequest(volume_id=self.object_id)
            # TODO(gongy): only apply indefinite retries on 504 status.
            _ = await retry_transient_errors(self._client.stub.VolumeCommit, req, max_retries=90)
            # Reload changes on successful commit.
            await self._do_reload(lock=False)

    @live_method
    async def reload(self):
        """Make latest committed state of volume available in the running container.

        Uncommitted changes to the volume, such as new or modified files, will be preserved during reload. Uncommitted
        changes will shadow any changes made by other writers - e.g. if you have an uncommitted modified a file that was
        also updated by another writer you will not see the other change.

        Reloading will fail if there are open files for the volume.
        """
        await self._do_reload()

    @live_method_gen
    async def iterdir(self, path: str) -> AsyncIterator[api_pb2.VolumeListFilesEntry]:
        """Iterate over all files in a directory in the volume.

        * Passing a directory path lists all files in the directory (names are relative to the directory)
        * Passing a file path returns a list containing only that file's listing description
        * Passing a glob path (including at least one * or ** sequence) returns all files matching that glob path (using absolute paths)
        """
        req = api_pb2.VolumeListFilesRequest(volume_id=self.object_id, path=path)
        async for batch in unary_stream(self._client.stub.VolumeListFiles, req):
            for entry in batch.entries:
                yield entry

    @live_method
    async def listdir(self, path: str) -> List[api_pb2.VolumeListFilesEntry]:
        """List all files under a path prefix in the modal.Volume.

        * Passing a directory path lists all files in the directory
        * Passing a file path returns a list containing only that file's listing description
        * Passing a glob path (including at least one * or ** sequence) returns all files matching that glob path (using absolute paths)
        """
        return [entry async for entry in self.iterdir(path)]

    @live_method_gen
    async def read_file(self, path: Union[str, bytes]) -> AsyncIterator[bytes]:
        """
        Read a file from the modal.Volume.

        **Example:**

        ```python notest
        vol = modal.Volume.lookup("my-modal-volume")
        data = b""
        for chunk in vol.read_file("1mb.csv"):
            data += chunk
        print(len(data))  # == 1024 * 1024
        ```
        """
        if isinstance(path, str):
            path = path.encode("utf-8")
        req = api_pb2.VolumeGetFileRequest(volume_id=self.object_id, path=path)
        response = await retry_transient_errors(self._client.stub.VolumeGetFile, req)
        if response.WhichOneof("data_oneof") == "data":
            yield response.data
        else:
            async for data in blob_iter(response.data_blob_id, self._client.stub):
                yield data

    @live_method
    async def _add_local_file(
        self, local_path: Union[Path, str], remote_path: Optional[Union[str, PurePosixPath, None]] = None
    ):
        mount_file = await self._upload_file(local_path, remote_path)
        request = api_pb2.VolumePutFilesRequest(volume_id=self.object_id, files=[mount_file])
        await retry_transient_errors(self._client.stub.VolumePutFiles, request, base_delay=1)

    @live_method
    async def _add_local_dir(
        self, local_path: Union[Path, str], remote_path: Optional[Union[str, PurePosixPath, None]] = None
    ):
        _local_path = Path(local_path)
        if remote_path is None:
            remote_path = PurePosixPath("/", _local_path.name).as_posix()
        else:
            remote_path = PurePosixPath(remote_path).as_posix()

        assert _local_path.is_dir()

        def gen_transfers():
            for subpath in _local_path.rglob("*"):
                if subpath.is_dir():
                    continue
                relpath_str = subpath.relative_to(_local_path).as_posix()
                yield self._upload_file(subpath, PurePosixPath(remote_path, relpath_str))

        files = await ConcurrencyPool(20).run_coros(gen_transfers(), return_exceptions=False)
        request = api_pb2.VolumePutFilesRequest(volume_id=self.object_id, files=files)
        await retry_transient_errors(self._client.stub.VolumePutFiles, request, base_delay=1)

    @live_method
    async def _upload_file(
        self, local_path: Union[Path, str], remote_path: Optional[Union[str, PurePosixPath, None]] = None
    ) -> api_pb2.MountFile:
        local_path = Path(local_path)
        if remote_path is None:
            remote_path = PurePosixPath("/", local_path.name).as_posix()
        else:
            remote_path = PurePosixPath(remote_path).as_posix()

        file_spec = get_file_upload_spec(local_path, str(remote_path))
        remote_filename = file_spec.mount_filename

        request = api_pb2.MountPutFileRequest(sha256_hex=file_spec.sha256_hex)
        response = await retry_transient_errors(self._client.stub.MountPutFile, request, base_delay=1)

        if not response.exists:
            if file_spec.use_blob:
                logger.debug(f"Creating blob file for {file_spec.filename} ({file_spec.size} bytes)")
                with open(file_spec.filename, "rb") as fp:
                    blob_id = await blob_upload_file(fp, self._client.stub)
                logger.debug(f"Uploading blob file {file_spec.filename} as {remote_filename}")
                request2 = api_pb2.MountPutFileRequest(data_blob_id=blob_id, sha256_hex=file_spec.sha256_hex)
            else:
                logger.debug(f"Uploading file {file_spec.filename} to {remote_filename} ({file_spec.size} bytes)")
                request2 = api_pb2.MountPutFileRequest(data=file_spec.content, sha256_hex=file_spec.sha256_hex)

            start_time = time.monotonic()
            while time.monotonic() - start_time < MOUNT_PUT_FILE_CLIENT_TIMEOUT:
                response = await retry_transient_errors(self._client.stub.MountPutFile, request2, base_delay=1)
                if response.exists:
                    break

        return api_pb2.MountFile(
            filename=remote_filename,
            sha256_hex=file_spec.sha256_hex,
            mode=file_spec.mode,
        )


Volume = synchronize_api(_Volume)
