# Copyright Modal Labs 2022
import os
from typing import AsyncIterator, BinaryIO, List, Optional

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.grpc_utils import retry_transient_errors
from modal_utils.hash_utils import get_sha256_hex

from ._blob_utils import LARGE_FILE_LIMIT, blob_iter, blob_upload_file
from ._resolver import Resolver
from .object import Handle, Provider


class _SharedVolumeHandle(Handle, type_prefix="sv"):
    """A handle to a Modal SharedVolume

    Should typically not be used directly in a Modal function,
    and instead referenced through the file system, see `modal.SharedVolume`.

    Also see the CLI methods for accessing shared volumes:

    ```modal volume --help```

    A SharedVolumeHandle *can* however be useful for some local scripting scenarios, e.g.:

    ```python notest
    vol = modal.lookup("my-shared-volume")
    for chunk in vol.read_file("my_db_dump.csv"):
        ...
    ```
    """

    async def write_file(self, remote_path: str, fp: BinaryIO):
        """Write from a file object to a path on the shared volume, atomically.

        Will create any needed parent directories automatically

        If remote_path ends with `/` it's assumed to be a directory and the
        file will be uploaded with its current name to that directory.
        """
        sha_hash = get_sha256_hex(fp)
        fp.seek(0, os.SEEK_END)
        data_size = fp.tell()
        fp.seek(0)
        if data_size > LARGE_FILE_LIMIT:
            blob_id = await blob_upload_file(fp, self._client.stub)
            req = api_pb2.SharedVolumePutFileRequest(
                shared_volume_id=self._object_id, path=remote_path, data_blob_id=blob_id, sha256_hex=sha_hash
            )
        else:
            data = fp.read()
            req = api_pb2.SharedVolumePutFileRequest(shared_volume_id=self._object_id, path=remote_path, data=data)
        await retry_transient_errors(self._client.stub.SharedVolumePutFile, req)
        return data_size  # might be better if this is returned from the server

    async def read_file(self, path: str) -> AsyncIterator[bytes]:
        """Read a file from the shared volume"""
        req = api_pb2.SharedVolumeGetFileRequest(shared_volume_id=self._object_id, path=path)
        response = await retry_transient_errors(self._client.stub.SharedVolumeGetFile, req)
        if response.WhichOneof("data_oneof") == "data":
            yield response.data
        else:
            async for data in blob_iter(response.data_blob_id, self._client.stub):
                yield data

    async def listdir(self, path: str) -> List[api_pb2.SharedVolumeListFilesEntry]:
        """List all files in a directory in the shared volume.

        * Passing a directory path lists all files in the directory (names are relative to the directory)
        * Passing a file path returns a list containing only that file's listing description.
        * Passing a glob path (including at least one * or ** sequence) returns all files matching that glob path (using absolute paths)
        """
        req = api_pb2.SharedVolumeListFilesRequest(shared_volume_id=self._object_id, path=path)
        response = await retry_transient_errors(self._client.stub.SharedVolumeListFiles, req)
        return list(response.entries)

    async def remove_file(self, path: str, recursive=False):
        """Remove a file in a shared volume"""
        req = api_pb2.SharedVolumeRemoveFileRequest(shared_volume_id=self._object_id, path=path, recursive=recursive)
        await retry_transient_errors(self._client.stub.SharedVolumeRemoveFile, req)


SharedVolumeHandle, AioSharedVolumeHandle = synchronize_apis(_SharedVolumeHandle)


class _SharedVolume(Provider[_SharedVolumeHandle]):
    """A shared, writable file system accessible by one or more Modal functions.

    By attaching this file system as a mount to one or more functions, they can
    share and persist data with each other.

    **Usage**

    ```python
    import modal

    stub = modal.Stub()

    @stub.function(shared_volumes={"/root/foo": modal.SharedVolume()})
    def f():
        pass
    ```

    It is often the case that you would want to persist a shared volume object
    separately from the currently attached app. Refer to the persistence
    [guide section](/docs/guide/shared-volumes#persisting-volumes) to see how to
    persist this object across app runs.
    """

    def __init__(self, cloud_provider: "Optional[api_pb2.CloudProvider.V]" = None) -> None:
        """Construct a new shared volume, which is empty by default."""

        async def _load(resolver: Resolver) -> _SharedVolumeHandle:
            if resolver.existing_object_id:
                # Volume already exists; do nothing.
                return _SharedVolumeHandle(resolver.client, resolver.existing_object_id)

            resolver.set_message("Creating shared volume...")
            req = api_pb2.SharedVolumeCreateRequest(app_id=resolver.app_id, cloud_provider=cloud_provider)
            resp = await retry_transient_errors(resolver.client.stub.SharedVolumeCreate, req)
            resolver.set_message("Created shared volume.")
            return _SharedVolumeHandle(resolver.client, resp.shared_volume_id)

        rep = "SharedVolume()"
        super().__init__(_load, rep)


SharedVolume, AioSharedVolume = synchronize_apis(_SharedVolume)
