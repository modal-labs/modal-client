# Copyright Modal Labs 2023
import asyncio
import concurrent.futures
import enum
import functools
import multiprocessing
import os
import platform
import re
import time
import typing
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Awaitable,
    BinaryIO,
    Callable,
    Optional,
    Union,
)

from google.protobuf.message import Message
from grpclib import GRPCError, Status
from synchronicity.async_wrap import asynccontextmanager

import modal_proto.api_pb2
from modal.exception import VolumeUploadTimeoutError
from modal_proto import api_pb2

from ._object import EPHEMERAL_OBJECT_HEARTBEAT_SLEEP, _get_environment_name, _Object, live_method, live_method_gen
from ._resolver import Resolver
from ._utils.async_utils import (
    TaskContext,
    aclosing,
    async_map,
    async_map_ordered,
    asyncnullcontext,
    synchronize_api,
)
from ._utils.blob_utils import (
    BLOCK_SIZE,
    FileUploadSpec,
    FileUploadSpec2,
    blob_iter,
    blob_upload_file,
    get_file_upload_spec_from_fileobj,
    get_file_upload_spec_from_path,
)
from ._utils.deprecation import deprecation_error, deprecation_warning, renamed_parameter
from ._utils.grpc_utils import retry_transient_errors
from ._utils.http_utils import ClientSessionRegistry
from ._utils.name_utils import check_object_name
from .client import _Client
from .config import logger

# Max duration for uploading to volumes files
# As a guide, files >40GiB will take >10 minutes to upload.
VOLUME_PUT_FILE_CLIENT_TIMEOUT = 60 * 60


class FileEntryType(enum.IntEnum):
    """Type of a file entry listed from a Modal volume."""

    UNSPECIFIED = 0
    FILE = 1
    DIRECTORY = 2
    SYMLINK = 3


@dataclass(frozen=True)
class FileEntry:
    """A file or directory entry listed from a Modal volume."""

    path: str
    type: FileEntryType
    mtime: int
    size: int

    @classmethod
    def _from_proto(cls, proto: api_pb2.FileEntry) -> "FileEntry":
        return cls(
            path=proto.path,
            type=FileEntryType(proto.type),
            mtime=proto.mtime,
            size=proto.size,
        )


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

    Volumes can only be reloaded if there are no open files for the volume - attempting to reload with open files
    will result in an error.

    **Usage**

    ```python
    import modal

    app = modal.App()
    volume = modal.Volume.from_name("my-persisted-volume", create_if_missing=True)

    @app.function(volumes={"/root/foo": volume})
    def f():
        with open("/root/foo/bar.txt", "w") as f:
            f.write("hello")
        volume.commit()  # Persist changes

    @app.function(volumes={"/root/foo": volume})
    def g():
        volume.reload()  # Fetch latest changes
        with open("/root/foo/bar.txt", "r") as f:
            print(f.read())
    ```
    """

    _lock: Optional[asyncio.Lock] = None
    _metadata: "typing.Optional[api_pb2.VolumeMetadata]"

    async def _get_lock(self):
        # To (mostly*) prevent multiple concurrent operations on the same volume, which can cause problems under
        # some unlikely circumstances.
        # *: You can bypass this by creating multiple handles to the same volume, e.g. via lookup. But this
        # covers the typical case = good enough.

        # Note: this function runs no async code but is marked as async to ensure it's
        # being run inside the synchronicity event loop and binds the lock to the
        # correct event loop on Python 3.9 which eagerly assigns event loops on
        # constructions of locks
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @staticmethod
    @renamed_parameter((2024, 12, 18), "label", "name")
    def from_name(
        name: str,
        *,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
        version: "typing.Optional[modal_proto.api_pb2.VolumeFsVersion.ValueType]" = None,
    ) -> "_Volume":
        """Reference a Volume by name, creating if necessary.

        In contrast to `modal.Volume.lookup`, this is a lazy method
        that defers hydrating the local object with metadata from
        Modal servers until the first time is is actually used.

        ```python
        vol = modal.Volume.from_name("my-volume", create_if_missing=True)

        app = modal.App()

        # Volume refers to the same object, even across instances of `app`.
        @app.function(volumes={"/data": vol})
        def f():
            pass
        ```
        """
        check_object_name(name, "Volume")

        async def _load(self: _Volume, resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.VolumeGetOrCreateRequest(
                deployment_name=name,
                namespace=namespace,
                environment_name=_get_environment_name(environment_name, resolver),
                object_creation_type=(api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING if create_if_missing else None),
                version=version,
            )
            response = await resolver.client.stub.VolumeGetOrCreate(req)
            self._hydrate(response.volume_id, resolver.client, response.metadata)

        return _Volume._from_loader(_load, "Volume()", hydrate_lazily=True)

    def _hydrate_metadata(self, metadata: Optional[Message]):
        if metadata and isinstance(metadata, api_pb2.VolumeMetadata):
            self._metadata = metadata
        else:
            raise TypeError(
                "_hydrate_metadata() requires an `api_pb2.VolumeMetadata` to determine volume version"
            )

    def _get_metadata(self) -> Optional[Message]:
        return self._metadata


    @property
    def _is_v1(self) -> bool:
        return self._metadata.version in [
            None,
            api_pb2.VolumeFsVersion.VOLUME_FS_VERSION_UNSPECIFIED,
            api_pb2.VolumeFsVersion.VOLUME_FS_VERSION_V1
        ]


    @classmethod
    @asynccontextmanager
    async def ephemeral(
        cls: type["_Volume"],
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        version: "typing.Optional[modal_proto.api_pb2.VolumeFsVersion.ValueType]" = None,
        _heartbeat_sleep: float = EPHEMERAL_OBJECT_HEARTBEAT_SLEEP,
    ) -> AsyncGenerator["_Volume", None]:
        """Creates a new ephemeral volume within a context manager:

        Usage:
        ```python
        import modal
        with modal.Volume.ephemeral() as vol:
            assert vol.listdir("/") == []
        ```

        ```python notest
        async with modal.Volume.ephemeral() as vol:
            assert await vol.listdir("/") == []
        ```
        """
        if client is None:
            client = await _Client.from_env()
        request = api_pb2.VolumeGetOrCreateRequest(
            object_creation_type=api_pb2.OBJECT_CREATION_TYPE_EPHEMERAL,
            environment_name=_get_environment_name(environment_name),
            version=version,
        )
        response = await client.stub.VolumeGetOrCreate(request)
        async with TaskContext() as tc:
            request = api_pb2.VolumeHeartbeatRequest(volume_id=response.volume_id)
            tc.infinite_loop(lambda: client.stub.VolumeHeartbeat(request), sleep=_heartbeat_sleep)
            yield cls._new_hydrated(response.volume_id, client, response.metadata, is_another_app=True)

    @staticmethod
    @renamed_parameter((2024, 12, 18), "label", "name")
    async def lookup(
        name: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
        version: "typing.Optional[modal_proto.api_pb2.VolumeFsVersion.ValueType]" = None,
    ) -> "_Volume":
        """Lookup a named Volume.

        DEPRECATED: This method is deprecated in favor of `modal.Volume.from_name`.

        In contrast to `modal.Volume.from_name`, this is an eager method
        that will hydrate the local object with metadata from Modal servers.

        ```python notest
        vol = modal.Volume.from_name("my-volume")
        print(vol.listdir("/"))
        ```
        """
        deprecation_warning(
            (2025, 1, 27),
            "`modal.Volume.lookup` is deprecated and will be removed in a future release."
            " It can be replaced with `modal.Volume.from_name`."
            "\n\nSee https://modal.com/docs/guide/modal-1-0-migration for more information.",
        )
        obj = _Volume.from_name(
            name,
            namespace=namespace,
            environment_name=environment_name,
            create_if_missing=create_if_missing,
            version=version,
        )
        if client is None:
            client = await _Client.from_env()
        resolver = Resolver(client=client)
        await resolver.load(obj)
        return obj

    @staticmethod
    async def create_deployed(
        deployment_name: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        version: "typing.Optional[modal_proto.api_pb2.VolumeFsVersion.ValueType]" = None,
    ) -> str:
        """mdmd:hidden"""
        check_object_name(deployment_name, "Volume")
        if client is None:
            client = await _Client.from_env()
        request = api_pb2.VolumeGetOrCreateRequest(
            deployment_name=deployment_name,
            namespace=namespace,
            environment_name=_get_environment_name(environment_name),
            object_creation_type=api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS,
            version=version,
        )
        resp = await retry_transient_errors(client.stub.VolumeGetOrCreate, request)
        return resp.volume_id

    @live_method
    async def _do_reload(self, lock=True):
        async with (await self._get_lock()) if lock else asyncnullcontext():
            req = api_pb2.VolumeReloadRequest(volume_id=self.object_id)
            _ = await retry_transient_errors(self._client.stub.VolumeReload, req)

    @live_method
    async def commit(self):
        """Commit changes to the volume.

        If successful, the changes made are now persisted in durable storage and available to other containers accessing
        the volume.
        """
        async with await self._get_lock():
            req = api_pb2.VolumeCommitRequest(volume_id=self.object_id)
            try:
                # TODO(gongy): only apply indefinite retries on 504 status.
                resp = await retry_transient_errors(self._client.stub.VolumeCommit, req, max_retries=90)
                if not resp.skip_reload:
                    # Reload changes on successful commit.
                    await self._do_reload(lock=False)
            except GRPCError as exc:
                raise RuntimeError(exc.message) if exc.status in (Status.FAILED_PRECONDITION, Status.NOT_FOUND) else exc

    @live_method
    async def reload(self):
        """Make latest committed state of volume available in the running container.

        Any uncommitted changes to the volume, such as new or modified files, may implicitly be committed when
        reloading.

        Reloading will fail if there are open files for the volume.
        """
        try:
            await self._do_reload()
        except GRPCError as exc:
            # TODO(staffan): This is brittle and janky, as it relies on specific paths and error messages which can
            #  change server-side at any time. Consider returning the open files directly in the error emitted from the
            #  server.
            if exc.message == "there are open files preventing the operation":
                # Attempt to identify what open files are problematic and include information about the first (to avoid
                # really verbose errors) open file in the error message to help troubleshooting.
                # This is best-effort and not necessarily bulletproof, as the view of open files inside the container
                # might differ from that outside - but it will at least catch common errors.
                vol_path = f"/__modal/volumes/{self.object_id}"
                annotation = _open_files_error_annotation(vol_path)
                if annotation:
                    raise RuntimeError(f"{exc.message}: {annotation}")

            raise RuntimeError(exc.message) if exc.status in (Status.FAILED_PRECONDITION, Status.NOT_FOUND) else exc

    @live_method_gen
    async def iterdir(self, path: str, *, recursive: bool = True) -> AsyncIterator[FileEntry]:
        """Iterate over all files in a directory in the volume.

        Passing a directory path lists all files in the directory. For a file path, return only that
        file's description. If `recursive` is set to True, list all files and folders under the path
        recursively.
        """
        from modal_version import major_number, minor_number

        # This allows us to remove the server shim after 0.62 is no longer supported.
        deprecation = deprecation_warning if (major_number, minor_number) <= (0, 62) else deprecation_error
        if path.endswith("**"):
            msg = (
                "Glob patterns in `volume get` and `Volume.listdir()` are deprecated. "
                "Please pass recursive=True instead. For the CLI, just remove the glob suffix."
            )
            deprecation(
                (2024, 4, 23),
                msg,
            )
        elif path.endswith("*"):
            deprecation(
                (2024, 4, 23),
                (
                    "Glob patterns in `volume get` and `Volume.listdir()` are deprecated. "
                    "Please remove the glob `*` suffix."
                ),
            )

        req = api_pb2.VolumeListFilesRequest(volume_id=self.object_id, path=path, recursive=recursive)
        async for batch in self._client.stub.VolumeListFiles.unary_stream(req):
            for entry in batch.entries:
                yield FileEntry._from_proto(entry)

    @live_method
    async def listdir(self, path: str, *, recursive: bool = False) -> list[FileEntry]:
        """List all files under a path prefix in the modal.Volume.

        Passing a directory path lists all files in the directory. For a file path, return only that
        file's description. If `recursive` is set to True, list all files and folders under the path
        recursively.
        """
        return [entry async for entry in self.iterdir(path, recursive=recursive)]

    @live_method_gen
    def read_file(self, path: str) -> AsyncIterator[bytes]:
        """
        Read a file from the modal.Volume.

        **Example:**

        ```python notest
        vol = modal.Volume.from_name("my-modal-volume")
        data = b""
        for chunk in vol.read_file("1mb.csv"):
            data += chunk
        print(len(data))  # == 1024 * 1024
        ```
        """
        return self._read_file1(path) if self._is_v1 else self._read_file2(path)


    async def _read_file1(self, path: str) -> AsyncIterator[bytes]:
        req = api_pb2.VolumeGetFileRequest(volume_id=self.object_id, path=path)
        try:
            response = await retry_transient_errors(self._client.stub.VolumeGetFile, req)
        except GRPCError as exc:
            raise FileNotFoundError(exc.message) if exc.status == Status.NOT_FOUND else exc
        # TODO(Jonathon): use ranged requests.
        if response.WhichOneof("data_oneof") == "data":
            yield response.data
            return
        else:
            async for data in blob_iter(response.data_blob_id, self._client.stub):
                yield data


    async def _read_file2(self, path: str) -> AsyncIterator[bytes]:
        req = api_pb2.VolumeGetFile2Request(volume_id=self.object_id, path=path)

        try:
            response = await retry_transient_errors(self._client.stub.VolumeGetFile2, req)
        except GRPCError as exc:
            raise FileNotFoundError(exc.message) if exc.status == Status.NOT_FOUND else exc

        async def read_block(block_url: str) -> bytes:
            async with ClientSessionRegistry.get_session().get(block_url) as get_response:
                return await get_response.content.read()

        async def iter_urls() -> AsyncGenerator[str]:
            for url in response.get_urls:
                yield url

        # TODO(dflemstr): Reasonable default? Make configurable?
        prefetch_num_blocks = multiprocessing.cpu_count()

        async with aclosing(async_map_ordered(iter_urls(), read_block, concurrency=prefetch_num_blocks)) as stream:
            async for value in stream:
                yield value


    @live_method
    async def remove_file(self, path: str, recursive: bool = False) -> None:
        """Remove a file or directory from a volume."""
        req = api_pb2.VolumeRemoveFileRequest(volume_id=self.object_id, path=path, recursive=recursive)
        await retry_transient_errors(self._client.stub.VolumeRemoveFile, req)

    @live_method
    async def copy_files(self, src_paths: Sequence[str], dst_path: str) -> None:
        """
        Copy files within the volume from src_paths to dst_path.
        The semantics of the copy operation follow those of the UNIX cp command.

        The `src_paths` parameter is a list. If you want to copy a single file, you should pass a list with a
        single element.

        `src_paths` and `dst_path` should refer to the desired location *inside* the volume. You do not need to prepend
        the volume mount path.

        **Usage**

        ```python notest
        vol = modal.Volume.from_name("my-modal-volume")

        vol.copy_files(["bar/example.txt"], "bar2")  # Copy files to another directory
        vol.copy_files(["bar/example.txt"], "bar/example2.txt")  # Rename a file by copying
        ```

        Note that if the volume is already mounted on the Modal function, you should use normal filesystem operations
        like `os.rename()` and then `commit()` the volume. The `copy_files()` method is useful when you don't have
        the volume mounted as a filesystem, e.g. when running a script on your local computer.
        """
        request = api_pb2.VolumeCopyFilesRequest(volume_id=self.object_id, src_paths=src_paths, dst_path=dst_path)
        await retry_transient_errors(self._client.stub.VolumeCopyFiles, request, base_delay=1)

    @live_method
    async def batch_upload(self, force: bool = False) -> "_AbstractVolumeUploadContextManager":
        """
        Initiate a batched upload to a volume.

        To allow overwriting existing files, set `force` to `True` (you cannot overwrite existing directories with
        uploaded files regardless).

        **Example:**

        ```python notest
        vol = modal.Volume.from_name("my-modal-volume")

        with vol.batch_upload() as batch:
            batch.put_file("local-path.txt", "/remote-path.txt")
            batch.put_directory("/local/directory/", "/remote/directory")
            batch.put_file(io.BytesIO(b"some data"), "/foobar")
        ```
        """
        return _AbstractVolumeUploadContextManager.resolve(
            self._metadata.version,
            self.object_id,
            self._client,
            force=force
        )


    @live_method
    async def _instance_delete(self):
        await retry_transient_errors(
            self._client.stub.VolumeDelete, api_pb2.VolumeDeleteRequest(volume_id=self.object_id)
        )

    @staticmethod
    @renamed_parameter((2024, 12, 18), "label", "name")
    async def delete(name: str, client: Optional[_Client] = None, environment_name: Optional[str] = None):
        obj = await _Volume.from_name(name, environment_name=environment_name).hydrate(client)
        req = api_pb2.VolumeDeleteRequest(volume_id=obj.object_id)
        await retry_transient_errors(obj._client.stub.VolumeDelete, req)

    @staticmethod
    async def rename(
        old_name: str,
        new_name: str,
        *,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
    ):
        obj = await _Volume.from_name(old_name, environment_name=environment_name).hydrate(client)
        req = api_pb2.VolumeRenameRequest(volume_id=obj.object_id, name=new_name)
        await retry_transient_errors(obj._client.stub.VolumeRename, req)


Volume = synchronize_api(_Volume)

# TODO(dflemstr): Find a way to add ABC or AbstractAsyncContextManager superclasses while keeping synchronicity happy.
class _AbstractVolumeUploadContextManager:
    async def __aenter__(self):
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...


    def put_file(
        self,
        local_file: Union[Path, str, BinaryIO, BytesIO],
        remote_path: Union[PurePosixPath, str],
        mode: Optional[int] = None,
    ):
        ...

    def put_directory(
        self,
        local_path: Union[Path, str],
        remote_path: Union[PurePosixPath, str],
        recursive: bool = True,
    ):
        ...

    @staticmethod
    def resolve(
        version: "modal_proto.api_pb2.VolumeFsVersion.ValueType",
        object_id: str,
        client,
        progress_cb: Optional[Callable[..., Any]] = None,
        force: bool = False
    ) -> "_AbstractVolumeUploadContextManager":

        if version in [
            None,
            api_pb2.VolumeFsVersion.VOLUME_FS_VERSION_UNSPECIFIED,
            api_pb2.VolumeFsVersion.VOLUME_FS_VERSION_V1
        ]:
            return _VolumeUploadContextManager(object_id, client, progress_cb=progress_cb, force=force)
        elif version == api_pb2.VolumeFsVersion.VOLUME_FS_VERSION_V2:
            return _VolumeUploadContextManager2(object_id, client, progress_cb=progress_cb, force=force)
        else:
            raise RuntimeError(f"unsupported volume version: {version}")


AbstractVolumeUploadContextManager = synchronize_api(_AbstractVolumeUploadContextManager)

class _VolumeUploadContextManager(_AbstractVolumeUploadContextManager):
    """Context manager for batch-uploading files to a Volume."""

    _volume_id: str
    _client: _Client
    _force: bool
    progress_cb: Callable[..., Any]
    _upload_generators: list[Generator[Callable[[], FileUploadSpec], None, None]]

    def __init__(
        self, volume_id: str, client: _Client, progress_cb: Optional[Callable[..., Any]] = None, force: bool = False
    ):
        """mdmd:hidden"""
        self._volume_id = volume_id
        self._client = client
        self._upload_generators = []
        self._progress_cb = progress_cb or (lambda *_, **__: None)
        self._force = force

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not exc_val:
            # Flatten all the uploads yielded by the upload generators in the batch
            def gen_upload_providers():
                for gen in self._upload_generators:
                    yield from gen

            async def gen_file_upload_specs() -> AsyncGenerator[FileUploadSpec, None]:
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as exe:
                    # TODO: avoid eagerly expanding
                    futs = [loop.run_in_executor(exe, f) for f in gen_upload_providers()]
                    logger.debug(f"Computing checksums for {len(futs)} files using {exe._max_workers} workers")
                    for fut in asyncio.as_completed(futs):
                        yield await fut

            # Compute checksums & Upload files
            files: list[api_pb2.MountFile] = []
            async with aclosing(async_map(gen_file_upload_specs(), self._upload_file, concurrency=20)) as stream:
                async for item in stream:
                    files.append(item)

            self._progress_cb(complete=True)

            request = api_pb2.VolumePutFilesRequest(
                volume_id=self._volume_id,
                files=files,
                disallow_overwrite_existing_files=not self._force,
            )
            try:
                await retry_transient_errors(self._client.stub.VolumePutFiles, request, base_delay=1)
            except GRPCError as exc:
                raise FileExistsError(exc.message) if exc.status == Status.ALREADY_EXISTS else exc

    def put_file(
        self,
        local_file: Union[Path, str, BinaryIO, BytesIO],
        remote_path: Union[PurePosixPath, str],
        mode: Optional[int] = None,
    ):
        """Upload a file from a local file or file-like object.

        Will create any needed parent directories automatically.

        If `local_file` is a file-like object it must remain readable for the lifetime of the batch.
        """
        remote_path = PurePosixPath(remote_path).as_posix()
        if remote_path.endswith("/"):
            raise ValueError(f"remote_path ({remote_path}) must refer to a file - cannot end with /")

        def gen():
            if isinstance(local_file, str) or isinstance(local_file, Path):
                yield lambda: get_file_upload_spec_from_path(local_file, PurePosixPath(remote_path), mode)
            else:
                yield lambda: get_file_upload_spec_from_fileobj(local_file, PurePosixPath(remote_path), mode or 0o644)

        self._upload_generators.append(gen())

    def put_directory(
        self,
        local_path: Union[Path, str],
        remote_path: Union[PurePosixPath, str],
        recursive: bool = True,
    ):
        """
        Upload all files in a local directory.

        Will create any needed parent directories automatically.
        """
        local_path = Path(local_path)
        assert local_path.is_dir()
        remote_path = PurePosixPath(remote_path)

        def create_file_spec_provider(subpath):
            relpath_str = subpath.relative_to(local_path)
            return lambda: get_file_upload_spec_from_path(subpath, remote_path / relpath_str)

        def gen():
            glob = local_path.rglob("*") if recursive else local_path.glob("*")
            for subpath in glob:
                # Skip directories and unsupported file types (e.g. block devices)
                if subpath.is_file():
                    yield create_file_spec_provider(subpath)

        self._upload_generators.append(gen())

    async def _upload_file(self, file_spec: FileUploadSpec) -> api_pb2.MountFile:
        remote_filename = file_spec.mount_filename
        progress_task_id = self._progress_cb(name=remote_filename, size=file_spec.size)
        request = api_pb2.MountPutFileRequest(sha256_hex=file_spec.sha256_hex)
        response = await retry_transient_errors(self._client.stub.MountPutFile, request, base_delay=1)

        start_time = time.monotonic()
        if not response.exists:
            if file_spec.use_blob:
                logger.debug(f"Creating blob file for {file_spec.source_description} ({file_spec.size} bytes)")
                with file_spec.source() as fp:
                    blob_id = await blob_upload_file(
                        fp,
                        self._client.stub,
                        functools.partial(self._progress_cb, progress_task_id),
                        sha256_hex=file_spec.sha256_hex,
                        md5_hex=file_spec.md5_hex,
                    )
                logger.debug(f"Uploading blob file {file_spec.source_description} as {remote_filename}")
                request2 = api_pb2.MountPutFileRequest(data_blob_id=blob_id, sha256_hex=file_spec.sha256_hex)
            else:
                logger.debug(
                    f"Uploading file {file_spec.source_description} to {remote_filename} ({file_spec.size} bytes)"
                )
                request2 = api_pb2.MountPutFileRequest(data=file_spec.content, sha256_hex=file_spec.sha256_hex)
                self._progress_cb(task_id=progress_task_id, complete=True)

            while (time.monotonic() - start_time) < VOLUME_PUT_FILE_CLIENT_TIMEOUT:
                response = await retry_transient_errors(self._client.stub.MountPutFile, request2, base_delay=1)
                if response.exists:
                    break

            if not response.exists:
                raise VolumeUploadTimeoutError(f"Uploading of {file_spec.source_description} timed out")
        else:
            self._progress_cb(task_id=progress_task_id, complete=True)
        return api_pb2.MountFile(
            filename=remote_filename,
            sha256_hex=file_spec.sha256_hex,
            mode=file_spec.mode,
        )


VolumeUploadContextManager = synchronize_api(_VolumeUploadContextManager)

_FileUploader2 = Callable[[asyncio.Semaphore], Awaitable[FileUploadSpec2]]

class _VolumeUploadContextManager2(_AbstractVolumeUploadContextManager):
    """Context manager for batch-uploading files to a Volume version 2."""

    _volume_id: str
    _client: _Client
    _progress_cb: Callable[..., Any]
    _force: bool
    _hash_concurrency: int
    _put_concurrency: int
    _uploader_generators: list[Generator[_FileUploader2]]

    def __init__(
        self,
        volume_id: str,
        client: _Client,
        progress_cb: Optional[Callable[..., Any]] = None,
        force: bool = False,
        hash_concurrency: int = multiprocessing.cpu_count(),
        put_concurrency: int = multiprocessing.cpu_count(),
    ):
        """mdmd:hidden"""
        self._volume_id = volume_id
        self._client = client
        self._uploader_generators = []
        self._progress_cb = progress_cb or (lambda *_, **__: None)
        self._force = force
        self._hash_concurrency = hash_concurrency
        self._put_concurrency = put_concurrency

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not exc_val:
            # Flatten all the uploads yielded by the upload generators in the batch
            def gen_upload_providers():
                for gen in self._uploader_generators:
                    yield from gen

            async def gen_file_upload_specs() -> list[FileUploadSpec2]:
                hash_semaphore = asyncio.Semaphore(self._hash_concurrency)

                uploads = [asyncio.create_task(fut(hash_semaphore)) for fut in gen_upload_providers()]
                logger.debug(f"Computing checksums for {len(uploads)} files")

                file_specs = []
                for file_spec in asyncio.as_completed(uploads):
                    file_specs.append(await file_spec)
                return file_specs

            upload_specs = await gen_file_upload_specs()
            await self._put_file_specs(upload_specs)


    def put_file(
        self,
        local_file: Union[Path, str, BinaryIO, BytesIO],
        remote_path: Union[PurePosixPath, str],
        mode: Optional[int] = None,
    ):
        """Upload a file from a local file or file-like object.

        Will create any needed parent directories automatically.

        If `local_file` is a file-like object it must remain readable for the lifetime of the batch.
        """
        remote_path = PurePosixPath(remote_path).as_posix()
        if remote_path.endswith("/"):
            raise ValueError(f"remote_path ({remote_path}) must refer to a file - cannot end with /")

        def gen():
            if isinstance(local_file, str) or isinstance(local_file, Path):
                yield lambda hash_semaphore: FileUploadSpec2.from_path(
                    local_file,
                    PurePosixPath(remote_path),
                    hash_semaphore,
                    mode
                )
            else:
                yield lambda hash_semaphore: FileUploadSpec2.from_fileobj(
                    local_file,
                    PurePosixPath(remote_path),
                    hash_semaphore,
                    mode or 0o644
                )

        self._uploader_generators.append(gen())

    def put_directory(
        self,
        local_path: Union[Path, str],
        remote_path: Union[PurePosixPath, str],
        recursive: bool = True,
    ):
        """
        Upload all files in a local directory.

        Will create any needed parent directories automatically.
        """
        local_path = Path(local_path)
        assert local_path.is_dir()
        remote_path = PurePosixPath(remote_path)

        def create_spec(subpath):
            relpath_str = subpath.relative_to(local_path)
            return lambda hash_semaphore: FileUploadSpec2.from_path(subpath, remote_path / relpath_str, hash_semaphore)

        def gen():
            glob = local_path.rglob("*") if recursive else local_path.glob("*")
            for subpath in glob:
                # Skip directories and unsupported file types (e.g. block devices)
                if subpath.is_file():
                    yield create_spec(subpath)

        self._uploader_generators.append(gen())

    async def _put_file_specs(self, file_specs: list[FileUploadSpec2]):
        put_responses = {}
        # num_blocks_total = sum(len(file_spec.blocks_sha256) for file_spec in file_specs)

        logger.debug(f"Ensuring {len(file_specs)} files are uploaded...")

        # We should only need two iterations: Once to possibly get some missing_blocks; the second time we should have
        # all blocks uploaded
        for _ in range(2):
            files = []

            for file_spec in file_specs:
                blocks = [
                    api_pb2.VolumePutFiles2Request.Block(
                        contents_sha256=block_sha256,
                        put_response=put_responses.get(block_sha256)
                    ) for block_sha256 in file_spec.blocks_sha256
                ]
                files.append(api_pb2.VolumePutFiles2Request.File(
                    path=file_spec.path,
                    mode=file_spec.mode,
                    size=file_spec.size,
                    blocks=blocks
                ))

            request = api_pb2.VolumePutFiles2Request(
                volume_id=self._volume_id,
                files=files,
                disallow_overwrite_existing_files=not self._force,
            )

            try:
                response = await retry_transient_errors(self._client.stub.VolumePutFiles2, request, base_delay=1)
            except GRPCError as exc:
                raise FileExistsError(exc.message) if exc.status == Status.ALREADY_EXISTS else exc

            if not response.missing_blocks:
                break

            await _put_missing_blocks(
                file_specs,
                response.missing_blocks,
                put_responses,
                self._put_concurrency,
                self._progress_cb
            )
        else:
            raise RuntimeError("Did not succeed at uploading all files despite supplying all missing blocks")

        self._progress_cb(complete=True)


VolumeUploadContextManager2 = synchronize_api(_VolumeUploadContextManager2)


async def _put_missing_blocks(
    file_specs: list[FileUploadSpec2],
    # TODO(dflemstr): Element type is `api_pb2.VolumePutFiles2Response.MissingBlock` but synchronicity gets confused
    # by the nested class (?)
    missing_blocks: list,
    put_responses: dict[bytes, bytes],
    put_concurrency: int,
    progress_cb: Callable[..., Any]
):
    @dataclass
    class FileProgress:
        task_id: str
        pending_blocks: set[int]

    put_semaphore = asyncio.Semaphore(put_concurrency)
    file_progresses: dict[str, FileProgress] = dict()

    logger.debug(f"Uploading {len(missing_blocks)} missing blocks...")

    async def put_missing_block(
        # TODO(dflemstr): Type is `api_pb2.VolumePutFiles2Response.MissingBlock` but synchronicity gets confused
        # by the nested class (?)
        missing_block
    ) -> (bytes, bytes):
        # Lazily import to keep the eager loading time of this module down
        from ._utils.bytes_io_segment_payload import BytesIOSegmentPayload

        assert isinstance(missing_block, api_pb2.VolumePutFiles2Response.MissingBlock)

        file_spec = file_specs[missing_block.file_index]
        # TODO(dflemstr): What if the underlying file has changed here in the meantime; should we check the
        #  hash again just to be sure?
        block_sha256 = file_spec.blocks_sha256[missing_block.block_index]
        block_start = missing_block.block_index * BLOCK_SIZE
        block_length = min(BLOCK_SIZE, file_spec.size - block_start)

        if file_spec.path not in file_progresses:
            file_task_id = progress_cb(name=file_spec.path, size=file_spec.size)
            file_progresses[file_spec.path] = FileProgress(task_id=file_task_id, pending_blocks=set())

        file_progress = file_progresses[file_spec.path]
        file_progress.pending_blocks.add(missing_block.block_index)
        task_progress_cb = functools.partial(progress_cb, task_id=file_progress.task_id)

        async with put_semaphore:
            with file_spec.source() as source_fp:
                payload = BytesIOSegmentPayload(
                    source_fp,
                    block_start,
                    block_length,
                    # limit chunk size somewhat to not keep event loop busy for too long
                    chunk_size=256*1024,
                    progress_report_cb=task_progress_cb
                )

                async with ClientSessionRegistry.get_session().put(
                    missing_block.put_url,
                    data=payload,
                ) as response:
                    response.raise_for_status()
                    resp_data = await response.content.read()

        file_progress.pending_blocks.remove(missing_block.block_index)

        if len(file_progress.pending_blocks) == 0:
            task_progress_cb(complete=True)

        return block_sha256, resp_data

    tasks = [
        asyncio.create_task(put_missing_block(missing_block))
        for missing_block in missing_blocks
    ]
    for task_result in asyncio.as_completed(tasks):
        digest, resp = await task_result
        put_responses[digest] = resp


def _open_files_error_annotation(mount_path: str) -> Optional[str]:
    if platform.system() != "Linux":
        return None

    self_pid = os.readlink("/proc/self")

    def find_open_file_for_pid(pid: str) -> Optional[str]:
        # /proc/{pid}/cmdline is null separated
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
            parts = raw.split(b"\0")
            cmdline = " ".join([part.decode() for part in parts]).rstrip(" ")

        cwd = PurePosixPath(os.readlink(f"/proc/{pid}/cwd"))
        if cwd.is_relative_to(mount_path):
            if pid == self_pid:
                return "cwd is inside volume"
            else:
                return f"cwd of '{cmdline}' is inside volume"

        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                path = PurePosixPath(os.readlink(f"/proc/{pid}/fd/{fd}"))
                try:
                    rel_path = path.relative_to(mount_path)
                    if pid == self_pid:
                        return f"path {rel_path} is open"
                    else:
                        return f"path {rel_path} is open from '{cmdline}'"
                except ValueError:
                    pass

            except FileNotFoundError:
                # File was closed
                pass
        return None

    pid_re = re.compile("^[1-9][0-9]*$")
    for dirent in os.listdir("/proc/"):
        if pid_re.match(dirent):
            try:
                annotation = find_open_file_for_pid(dirent)
                if annotation:
                    return annotation
            except (FileNotFoundError, PermissionError):
                pass

    return None
