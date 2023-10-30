# Copyright Modal Labs 2023
import os
import time
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, BinaryIO, List, Optional, Tuple, Union

import modal
from modal._location import parse_cloud_provider
from modal_proto import api_pb2
from modal_utils.async_utils import ConcurrencyPool, synchronize_api
from modal_utils.grpc_utils import retry_transient_errors, unary_stream
from modal_utils.hash_utils import get_sha256_hex

from ._blob_utils import LARGE_FILE_LIMIT, blob_iter, blob_upload_file
from ._resolver import Resolver
from ._types import typechecked
from .object import _Object, live_method, live_method_gen

NETWORK_FILE_SYSTEM_PUT_FILE_CLIENT_TIMEOUT = (
    10 * 60
)  # 10 min max for transferring files (does not include upload time to s3)


def network_file_system_mount_protos(
    validated_network_file_systems: List[Tuple[str, "_NetworkFileSystem"]],
    allow_cross_region_volumes: bool,
) -> List[api_pb2.SharedVolumeMount]:
    network_file_system_mounts = []
    # Relies on dicts being ordered (true as of Python 3.6).
    for path, volume in validated_network_file_systems:
        network_file_system_mounts.append(
            api_pb2.SharedVolumeMount(
                mount_path=path,
                shared_volume_id=volume.object_id,
                allow_cross_region=allow_cross_region_volumes,
            )
        )
    return network_file_system_mounts


class _NetworkFileSystem(_Object, type_prefix="sv"):
    """A shared, writable file system accessible by one or more Modal functions.

    By attaching this file system as a mount to one or more functions, they can
    share and persist data with each other.

    **Usage**

    ```python
    import modal

    volume = modal.NetworkFileSystem.new()
    stub = modal.Stub()

    @stub.function(network_file_systems={"/root/foo": volume})
    def f():
        pass

    @stub.function(network_file_systems={"/root/goo": volume})
    def g():
        pass
    ```

    It is often the case that you would want to persist a network file system object
    separately from the currently attached app. Refer to the persistence
    [guide section](/docs/guide/network-file-systems#persisting-volumes) to see how to
    persist this object across app runs.

    Also see the CLI methods for accessing network file systems:

    ```bash
    modal nfs --help
    ```

    A `NetworkFileSystem` can also be useful for some local scripting scenarios, e.g.:

    ```python notest
    vol = modal.NetworkFileSystem.lookup("my-network-file-system")
    for chunk in vol.read_file("my_db_dump.csv"):
        ...
    ```
    """

    @typechecked
    @staticmethod
    def new(cloud: Optional[str] = None) -> "_NetworkFileSystem":
        """Construct a new network file system, which is empty by default."""

        async def _load(provider: _NetworkFileSystem, resolver: Resolver, existing_object_id: Optional[str]):
            status_row = resolver.add_status_row()
            if existing_object_id:
                # Volume already exists; do nothing.
                provider._hydrate(existing_object_id, resolver.client, None)
                return

            cloud_provider = parse_cloud_provider(cloud) if cloud else None

            status_row.message("Creating network file system...")
            req = api_pb2.SharedVolumeCreateRequest(app_id=resolver.app_id, cloud_provider=cloud_provider)
            resp = await retry_transient_errors(resolver.client.stub.SharedVolumeCreate, req)
            status_row.finish("Created network file system.")
            provider._hydrate(resp.shared_volume_id, resolver.client, None)

        return _NetworkFileSystem._from_loader(_load, "NetworkFileSystem()")

    @staticmethod
    def persisted(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        cloud: Optional[str] = None,
    ) -> "_NetworkFileSystem":
        """Deploy a Modal app containing this object.

        The deployed object can then be imported from other apps, or by calling
        `NetworkFileSystem.from_name(label)` from that same app.

        **Examples**

        ```python notest
        # In one app:
        volume = NetworkFileSystem.persisted("my-volume")

        # Later, in another app or Python file:
        volume = NetworkFileSystem.from_name("my-volume")

        @stub.function(network_file_systems={"/vol": volume})
        def f():
            pass
        ```
        """
        return _NetworkFileSystem.new(cloud)._persist(label, namespace, environment_name)

    def persist(
        self,
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        cloud: Optional[str] = None,
    ):
        """`NetworkFileSystem().persist("my-volume")` is deprecated. Use `NetworkFileSystem.persisted("my-volume")` instead."""
        return self.persisted(label, namespace, environment_name, cloud)

    @live_method
    async def write_file(self, remote_path: str, fp: BinaryIO) -> int:
        """Write from a file object to a path on the network file system, atomically.

        Will create any needed parent directories automatically.

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
                shared_volume_id=self.object_id,
                path=remote_path,
                data_blob_id=blob_id,
                sha256_hex=sha_hash,
                resumable=True,
            )
        else:
            data = fp.read()
            req = api_pb2.SharedVolumePutFileRequest(
                shared_volume_id=self.object_id, path=remote_path, data=data, resumable=True
            )

        t0 = time.monotonic()
        while time.monotonic() - t0 < NETWORK_FILE_SYSTEM_PUT_FILE_CLIENT_TIMEOUT:
            response = await retry_transient_errors(self._client.stub.SharedVolumePutFile, req)
            if response.exists:
                break
        else:
            raise modal.exception.TimeoutError(f"Uploading of {remote_path} timed out")

        return data_size  # might be better if this is returned from the server

    @live_method_gen
    async def read_file(self, path: str) -> AsyncIterator[bytes]:
        """Read a file from the network file system"""
        req = api_pb2.SharedVolumeGetFileRequest(shared_volume_id=self.object_id, path=path)
        response = await retry_transient_errors(self._client.stub.SharedVolumeGetFile, req)
        if response.WhichOneof("data_oneof") == "data":
            yield response.data
        else:
            async for data in blob_iter(response.data_blob_id, self._client.stub):
                yield data

    @live_method_gen
    async def iterdir(self, path: str) -> AsyncIterator[api_pb2.SharedVolumeListFilesEntry]:
        """Iterate over all files in a directory in the network file system.

        * Passing a directory path lists all files in the directory (names are relative to the directory)
        * Passing a file path returns a list containing only that file's listing description
        * Passing a glob path (including at least one * or ** sequence) returns all files matching that glob path (using absolute paths)
        """
        req = api_pb2.SharedVolumeListFilesRequest(shared_volume_id=self.object_id, path=path)
        async for batch in unary_stream(self._client.stub.SharedVolumeListFilesStream, req):
            for entry in batch.entries:
                yield entry

    @live_method
    async def add_local_file(
        self, local_path: Union[Path, str], remote_path: Optional[Union[str, PurePosixPath, None]] = None
    ):
        local_path = Path(local_path)
        if remote_path is None:
            remote_path = PurePosixPath("/", local_path.name).as_posix()
        else:
            remote_path = PurePosixPath(remote_path).as_posix()

        with local_path.open("rb") as local_file:
            return await self.write_file(remote_path, local_file)

    @live_method
    async def add_local_dir(
        self,
        local_path: Union[Path, str],
        remote_path: Optional[Union[str, PurePosixPath, None]] = None,
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
                yield self.add_local_file(subpath, PurePosixPath(remote_path, relpath_str))

        await ConcurrencyPool(20).run_coros(gen_transfers(), return_exceptions=True)

    @live_method
    async def listdir(self, path: str) -> List[api_pb2.SharedVolumeListFilesEntry]:
        """List all files in a directory in the network file system.

        * Passing a directory path lists all files in the directory (names are relative to the directory)
        * Passing a file path returns a list containing only that file's listing description
        * Passing a glob path (including at least one * or ** sequence) returns all files matching that glob path (using absolute paths)
        """
        return [entry async for entry in self.iterdir(path)]

    @live_method
    async def remove_file(self, path: str, recursive=False):
        """Remove a file in a network file system."""
        req = api_pb2.SharedVolumeRemoveFileRequest(shared_volume_id=self.object_id, path=path, recursive=recursive)
        await retry_transient_errors(self._client.stub.SharedVolumeRemoveFile, req)


NetworkFileSystem = synchronize_api(_NetworkFileSystem)
