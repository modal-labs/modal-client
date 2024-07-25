# Copyright Modal Labs 2023
import asyncio
import concurrent.futures
import enum
import functools
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import (
    IO,
    AsyncGenerator,
    AsyncIterator,
    BinaryIO,
    Callable,
    Generator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

import aiostream
from grpclib import GRPCError, Status
from synchronicity.async_wrap import asynccontextmanager

from modal.exception import InvalidError, VolumeUploadTimeoutError, deprecation_error, deprecation_warning
from modal_proto import api_pb2

from ._resolver import Resolver
from ._utils.async_utils import TaskContext, asyncnullcontext, synchronize_api
from ._utils.blob_utils import (
    FileUploadSpec,
    blob_iter,
    blob_upload_file,
    get_file_upload_spec_from_fileobj,
    get_file_upload_spec_from_path,
)
from ._utils.grpc_utils import retry_transient_errors, unary_stream
from ._utils.name_utils import check_object_name
from .client import _Client
from .config import logger
from .object import EPHEMERAL_OBJECT_HEARTBEAT_SLEEP, _get_environment_name, _Object, live_method, live_method_gen

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

    def __getattr__(self, name: str):
        deprecation_error(
            (2024, 4, 15),
            (
                f"The FileEntry dataclass was introduced to replace a private Protobuf message. "
                f"This dataclass does not have the {name} attribute."
            ),
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

    app = modal.App()  # Note: "app" was called "stub" up until April 2024
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

    _lock: asyncio.Lock

    def _initialize_from_empty(self):
        # To (mostly*) prevent multiple concurrent operations on the same volume, which can cause problems under
        # some unlikely circumstances.
        # *: You can bypass this by creating multiple handles to the same volume, e.g. via lookup. But this
        # covers the typical case = good enough.
        self._lock = asyncio.Lock()

    @staticmethod
    def new():
        """`Volume.new` is deprecated.

        Please use `Volume.from_name` (for persisted) or `Volume.ephemeral` (for ephemeral) volumes.
        """
        deprecation_error((2024, 3, 20), Volume.new.__doc__)

    @staticmethod
    def from_name(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
        version: "Optional[api_pb2.VolumeFsVersion.ValueType]" = None,
    ) -> "_Volume":
        """Create a reference to a persisted volume. Optionally create it lazily.

        **Example Usage**

        ```python
        import modal

        volume = modal.Volume.from_name("my-volume", create_if_missing=True)

        app = modal.App()  # Note: "app" was called "stub" up until April 2024

        # Volume refers to the same object, even across instances of `app`.
        @app.function(volumes={"/vol": volume})
        def f():
            pass
        ```
        """
        check_object_name(label, "Volume")

        async def _load(self: _Volume, resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.VolumeGetOrCreateRequest(
                deployment_name=label,
                namespace=namespace,
                environment_name=_get_environment_name(environment_name, resolver),
                object_creation_type=(api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING if create_if_missing else None),
                version=version,
            )
            response = await resolver.client.stub.VolumeGetOrCreate(req)
            self._hydrate(response.volume_id, resolver.client, None)

        return _Volume._from_loader(_load, "Volume()", hydrate_lazily=True)

    @classmethod
    @asynccontextmanager
    async def ephemeral(
        cls: Type["_Volume"],
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        version: "Optional[api_pb2.VolumeFsVersion.ValueType]" = None,
        _heartbeat_sleep: float = EPHEMERAL_OBJECT_HEARTBEAT_SLEEP,
    ) -> AsyncIterator["_Volume"]:
        """Creates a new ephemeral volume within a context manager:

        Usage:
        ```python
        with Volume.ephemeral() as vol:
            assert vol.listdir() == []

        async with Volume.ephemeral() as vol:
            assert await vol.listdir() == []
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
            yield cls._new_hydrated(response.volume_id, client, None, is_another_app=True)

    @staticmethod
    def persisted(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        cloud: Optional[str] = None,
    ):
        """Deprecated! Use `Volume.from_name(name, create_if_missing=True)`."""
        deprecation_error((2024, 3, 1), _Volume.persisted.__doc__)

    @staticmethod
    async def lookup(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
        version: "Optional[api_pb2.VolumeFsVersion.ValueType]" = None,
    ) -> "_Volume":
        """Lookup a volume with a given name

        ```python
        n = modal.Volume.lookup("my-volume")
        print(n.listdir("/"))
        ```
        """
        obj = _Volume.from_name(
            label,
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
        version: "Optional[api_pb2.VolumeFsVersion.ValueType]" = None,
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
        async with self._lock if lock else asyncnullcontext():
            req = api_pb2.VolumeReloadRequest(volume_id=self.object_id)
            _ = await retry_transient_errors(self._client.stub.VolumeReload, req)

    @live_method
    async def commit(self):
        """Commit changes to the volume.

        If successful, the changes made are now persisted in durable storage and available to other containers accessing
        the volume.
        """
        async with self._lock:
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
        async for batch in unary_stream(self._client.stub.VolumeListFiles, req):
            for entry in batch.entries:
                yield FileEntry._from_proto(entry)

    @live_method
    async def listdir(self, path: str, *, recursive: bool = False) -> List[FileEntry]:
        """List all files under a path prefix in the modal.Volume.

        Passing a directory path lists all files in the directory. For a file path, return only that
        file's description. If `recursive` is set to True, list all files and folders under the path
        recursively.
        """
        return [entry async for entry in self.iterdir(path, recursive=recursive)]

    @live_method_gen
    async def read_file(self, path: str) -> AsyncIterator[bytes]:
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

    @live_method
    async def read_file_into_fileobj(self, path: str, fileobj: IO[bytes]) -> int:
        """mdmd:hidden

        Read volume file into file-like IO object.
        In the future, this will replace the current generator implementation of the `read_file` method.
        """

        chunk_size_bytes = 8 * 1024 * 1024
        start = 0
        req = api_pb2.VolumeGetFileRequest(volume_id=self.object_id, path=path, start=start, len=chunk_size_bytes)
        try:
            response = await retry_transient_errors(self._client.stub.VolumeGetFile, req)
        except GRPCError as exc:
            raise FileNotFoundError(exc.message) if exc.status == Status.NOT_FOUND else exc
        if response.WhichOneof("data_oneof") != "data":
            raise RuntimeError("expected to receive 'data' in response")

        n = fileobj.write(response.data)
        if n != len(response.data):
            raise IOError(f"failed to write {len(response.data)} bytes to output. Wrote {n}.")
        elif n == response.size:
            return response.size
        elif n > response.size:
            raise RuntimeError(f"length of returned data exceeds reported filesize: {n} > {response.size}")
        # else: there's more data to read. continue reading with further ranged GET requests.
        file_size = response.size
        written = n

        while True:
            req = api_pb2.VolumeGetFileRequest(volume_id=self.object_id, path=path, start=written, len=chunk_size_bytes)
            response = await retry_transient_errors(self._client.stub.VolumeGetFile, req)
            if response.WhichOneof("data_oneof") != "data":
                raise RuntimeError("expected to receive 'data' in response")
            if len(response.data) > chunk_size_bytes:
                raise RuntimeError(f"received more data than requested: {len(response.data)} > {chunk_size_bytes}")
            elif (written + len(response.data)) > file_size:
                raise RuntimeError(f"received data exceeds filesize of {chunk_size_bytes}")

            n = fileobj.write(response.data)
            if n != len(response.data):
                raise IOError(f"failed to write {len(response.data)} bytes to output. Wrote {n}.")
            written += n
            if written == file_size:
                break

        return written

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
        vol = modal.Volume.lookup("my-modal-volume")

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
    async def batch_upload(self, force: bool = False) -> "_VolumeUploadContextManager":
        """
        Initiate a batched upload to a volume.

        To allow overwriting existing files, set `force` to `True` (you cannot overwrite existing directories with
        uploaded files regardless).

        **Example:**

        ```python notest
        vol = modal.Volume.lookup("my-modal-volume")

        with vol.batch_upload() as batch:
            batch.put_file("local-path.txt", "/remote-path.txt")
            batch.put_directory("/local/directory/", "/remote/directory")
            batch.put_file(io.BytesIO(b"some data"), "/foobar")
        ```
        """
        return _VolumeUploadContextManager(self.object_id, self._client, force=force)

    @live_method
    async def _instance_delete(self):
        await retry_transient_errors(
            self._client.stub.VolumeDelete, api_pb2.VolumeDeleteRequest(volume_id=self.object_id)
        )

    # @staticmethod  # TODO uncomment when enforcing deprecation of instance method invocation
    async def delete(*args, label: str = "", client: Optional[_Client] = None, environment_name: Optional[str] = None):
        # -- Backwards-compatibility section
        # TODO(michael) Upon enforcement of this deprecation, remove *args and the default argument for label=.
        if args:
            if isinstance(self := args[0], _Volume):
                msg = (
                    "Calling Volume.delete as an instance method is deprecated."
                    " Please update your code to call it as a static method, passing"
                    " the name of the volume to delete, e.g. `modal.Volume.delete('my-volume')`."
                )
                deprecation_warning((2024, 4, 23), msg)
                await self._instance_delete()
                return
            elif isinstance(args[0], type):
                args = args[1:]

            if isinstance(args[0], str):
                if label:
                    raise InvalidError("`label` specified as both positional and keyword argument")
                label = args[0]
        # -- Backwards-compatibility code ends here

        obj = await _Volume.lookup(label, client=client, environment_name=environment_name)
        req = api_pb2.VolumeDeleteRequest(volume_id=obj.object_id)
        await retry_transient_errors(obj._client.stub.VolumeDelete, req)


class _VolumeUploadContextManager:
    """Context manager for batch-uploading files to a Volume."""

    _volume_id: str
    _client: _Client
    _force: bool
    progress_cb: Callable
    _upload_generators: List[Generator[Callable[[], FileUploadSpec], None, None]]

    def __init__(self, volume_id: str, client: _Client, progress_cb: Optional[Callable] = None, force: bool = False):
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

            # Compute checksums
            files_stream = aiostream.stream.iterate(gen_file_upload_specs())
            # Upload files
            uploads_stream = aiostream.stream.map(files_stream, self._upload_file, task_limit=20)
            files: List[api_pb2.MountFile] = await aiostream.stream.list(uploads_stream)
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
        local_file: Union[Path, str, BinaryIO],
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
                        fp, self._client.stub, functools.partial(self._progress_cb, progress_task_id)
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


Volume = synchronize_api(_Volume)
VolumeUploadContextManager = synchronize_api(_VolumeUploadContextManager)


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
        # NOTE(staffan): Python 3.8 doesn't have is_relative_to(), so we're stuck with catching ValueError until
        # we drop Python 3.8 support.
        try:
            _rel_cwd = cwd.relative_to(mount_path)
            if pid == self_pid:
                return "cwd is inside volume"
            else:
                return f"cwd of '{cmdline}' is inside volume"
        except ValueError:
            pass

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
