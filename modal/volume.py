# Copyright Modal Labs 2023
import asyncio
import concurrent.futures
import time
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from typing import IO, AsyncGenerator, AsyncIterator, BinaryIO, Callable, Generator, List, Optional, Sequence, Union

import aiostream
from grpclib import GRPCError, Status

from modal.exception import VolumeUploadTimeoutError, deprecation_warning
from modal_proto import api_pb2

from ._resolver import Resolver
from ._utils.async_utils import asyncnullcontext, synchronize_api
from ._utils.blob_utils import (
    FileUploadSpec,
    blob_iter,
    blob_upload_file,
    get_file_upload_spec_from_fileobj,
    get_file_upload_spec_from_path,
)
from ._utils.grpc_utils import retry_transient_errors, unary_stream
from .client import _Client
from .config import logger
from .object import _get_environment_name, _Object, live_method, live_method_gen

# 15 min max for uploading to volumes files
# As a guide, files >40GiB will take >10 minutes to upload.
VOLUME_PUT_FILE_CLIENT_TIMEOUT = 15 * 60


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

        async def _load(self: _Volume, resolver: Resolver, existing_object_id: Optional[str]):
            status_row = resolver.add_status_row()
            if existing_object_id:
                # Volume already exists; do nothing.
                self._hydrate(existing_object_id, resolver.client, None)
                return

            status_row.message("Creating volume...")
            req = api_pb2.VolumeCreateRequest(app_id=resolver.app_id)
            resp = await retry_transient_errors(resolver.client.stub.VolumeCreate, req)
            status_row.finish("Created volume.")
            self._hydrate(resp.volume_id, resolver.client, None)

        return _Volume._from_loader(_load, "Volume()")

    @staticmethod
    def from_name(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
    ) -> "_Volume":
        """Create a reference to a persisted volume. Optionally create it lazily.

        **Example Usage**

        ```python
        import modal

        volume = modal.Volume.from_name("my-volume", create_if_missing=True)

        stub = modal.Stub()

        # Volume refers to the same object, even across instances of `stub`.
        @stub.function(volumes={"/vol": volume})
        def f():
            pass
        ```
        """

        async def _load(self: _Volume, resolver: Resolver, existing_object_id: Optional[str]):
            req = api_pb2.VolumeGetOrCreateRequest(
                deployment_name=label,
                namespace=namespace,
                environment_name=_get_environment_name(environment_name, resolver),
                object_creation_type=(api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING if create_if_missing else None),
            )
            response = await resolver.client.stub.VolumeGetOrCreate(req)
            self._hydrate(response.volume_id, resolver.client, None)

        return _Volume._from_loader(_load, "Volume()")

    @staticmethod
    def persisted(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        environment_name: Optional[str] = None,
        cloud: Optional[str] = None,
    ) -> "_Volume":
        """Deprecated! Use `Volume.from_name(name, create_if_missing=True)`."""
        deprecation_warning((2024, 3, 1), _Volume.persisted.__doc__)
        return _Volume.from_name(label, namespace, environment_name, create_if_missing=True)

    @staticmethod
    async def lookup(
        label: str,
        namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        client: Optional[_Client] = None,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
    ) -> "_Volume":
        """Lookup a volume with a given name

        ```python
        n = modal.Volume.lookup("my-volume")
        print(n.listdir("/"))
        ```
        """
        obj = _Volume.from_name(
            label, namespace=namespace, environment_name=environment_name, create_if_missing=create_if_missing
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
    ) -> str:
        """mdmd:hidden"""
        if client is None:
            client = await _Client.from_env()
        request = api_pb2.VolumeGetOrCreateRequest(
            deployment_name=deployment_name,
            namespace=namespace,
            environment_name=_get_environment_name(environment_name),
            object_creation_type=api_pb2.OBJECT_CREATION_TYPE_CREATE_FAIL_IF_EXISTS,
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
        """Commit changes to the volume and fetch any other changes made to the volume by other containers.

        Unless background commits are enabled, committing always triggers a reload after saving changes.

        If successful, the changes made are now persisted in durable storage and available to other containers accessing the volume.

        Committing will fail if there are open files for the volume.
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

        Uncommitted changes to the volume, such as new or modified files, will be preserved during reload. Uncommitted
        changes will shadow any changes made by other writers - e.g. if you have an uncommitted modified a file that was
        also updated by another writer you will not see the other change.

        Reloading will fail if there are open files for the volume.
        """
        try:
            await self._do_reload()
        except GRPCError as exc:
            raise RuntimeError(exc.message) if exc.status in (Status.FAILED_PRECONDITION, Status.NOT_FOUND) else exc

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
    async def read_file_into_fileobj(self, path: Union[str, bytes], fileobj: IO[bytes], progress: bool = False) -> int:
        """mdmd:hidden

        Read volume file into file-like IO object, with support for progress display.
        Used by modal CLI. In future will replace current generator implementation of `read_file` method.
        """
        if isinstance(path, str):
            path = path.encode("utf-8")

        if progress:
            from ._output import download_progress_bar

            progress_bar = download_progress_bar()
            task_id = progress_bar.add_task("download", path=path.decode(), start=False)
            progress_bar.console.log(f"Requesting {path.decode()}")
        else:
            progress_bar = nullcontext()
            task_id = None

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
            if progress:
                progress_bar.console.log(f"Wrote {n} bytes to '{path.decode()}'")
            return response.size
        elif n > response.size:
            raise RuntimeError(f"length of returned data exceeds reported filesize: {n} > {response.size}")
        # else: there's more data to read. continue reading with further ranged GET requests.
        start = n
        file_size = response.size
        written = n

        if progress:
            progress_bar.update(task_id, total=int(file_size))
            progress_bar.start_task(task_id)

        with progress_bar:
            while True:
                req = api_pb2.VolumeGetFileRequest(
                    volume_id=self.object_id, path=path, start=start, len=chunk_size_bytes
                )
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
                start += n
                written += n
                if progress:
                    progress_bar.update(task_id, advance=n)
                if written == file_size:
                    break

        if progress:
            progress_bar.console.log(f"Wrote {written} bytes to '{path.decode()}'")
        return written

    @live_method
    async def remove_file(self, path: Union[str, bytes], recursive: bool = False) -> None:
        """Remove a file or directory from a volume."""
        if isinstance(path, str):
            path = path.encode("utf-8")
        req = api_pb2.VolumeRemoveFileRequest(volume_id=self.object_id, path=path, recursive=recursive)
        await retry_transient_errors(self._client.stub.VolumeRemoveFile, req)

    @live_method
    async def copy_files(self, src_paths: Sequence[Union[str, bytes]], dst_path: Union[str, bytes]) -> None:
        """
        Copy files within the volume from src_paths to dst_path.
        The semantics of the copy operation follow those of the UNIX cp command.
        """
        src_paths = [path.encode("utf-8") for path in src_paths if isinstance(path, str)]
        if isinstance(dst_path, str):
            dst_path = dst_path.encode("utf-8")

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


class _VolumeUploadContextManager:
    """Context manager for batch-uploading files to a Volume."""

    _volume_id: str
    _client: _Client
    _force: bool
    _upload_generators: List[Generator[Callable[[], FileUploadSpec], None, None]]

    def __init__(self, volume_id: str, client: _Client, force: bool = False):
        """mdmd:hidden"""
        self._volume_id = volume_id
        self._client = client
        self._upload_generators = []
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

        request = api_pb2.MountPutFileRequest(sha256_hex=file_spec.sha256_hex)
        response = await retry_transient_errors(self._client.stub.MountPutFile, request, base_delay=1)

        if not response.exists:
            if file_spec.use_blob:
                logger.debug(f"Creating blob file for {file_spec.source_description} ({file_spec.size} bytes)")
                with file_spec.source() as fp:
                    blob_id = await blob_upload_file(fp, self._client.stub)
                logger.debug(f"Uploading blob file {file_spec.source_description} as {remote_filename}")
                request2 = api_pb2.MountPutFileRequest(data_blob_id=blob_id, sha256_hex=file_spec.sha256_hex)
            else:
                logger.debug(
                    f"Uploading file {file_spec.source_description} to {remote_filename} ({file_spec.size} bytes)"
                )
                request2 = api_pb2.MountPutFileRequest(data=file_spec.content, sha256_hex=file_spec.sha256_hex)

            start_time = time.monotonic()
            while time.monotonic() - start_time < VOLUME_PUT_FILE_CLIENT_TIMEOUT:
                response = await retry_transient_errors(self._client.stub.MountPutFile, request2, base_delay=1)
                if response.exists:
                    break

            if not response.exists:
                raise VolumeUploadTimeoutError(f"Uploading of {file_spec.source_description} timed out")

        return api_pb2.MountFile(
            filename=remote_filename,
            sha256_hex=file_spec.sha256_hex,
            mode=file_spec.mode,
        )


Volume = synchronize_api(_Volume)
VolumeUploadContextManager = synchronize_api(_VolumeUploadContextManager)
