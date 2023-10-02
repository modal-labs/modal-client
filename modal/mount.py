# Copyright Modal Labs 2022
import abc
import asyncio
import concurrent.futures
import dataclasses
import functools
import os
import time
import typing
from datetime import date
from pathlib import Path, PurePosixPath
from typing import AsyncGenerator, Callable, List, Optional, Sequence, Tuple, Union

import aiostream
from google.protobuf.message import Message

import modal.exception
from modal._types import typechecked
from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api
from modal_utils.grpc_utils import retry_transient_errors
from modal_utils.package_utils import get_module_mount_info, module_mount_condition
from modal_version import __version__

from ._blob_utils import FileUploadSpec, blob_upload_file, get_file_upload_spec
from ._resolver import Resolver
from .config import config, logger
from .exception import NotFoundError
from .object import _Object

MOUNT_PUT_FILE_CLIENT_TIMEOUT = 10 * 60  # 10 min max for transferring files

# Supported releases and versions for python-build-standalone.
#
# These can be updated safely, but changes will trigger a rebuild for all images
# that rely on `add_python()` in their constructor.
PYTHON_STANDALONE_VERSIONS: typing.Dict[str, typing.Tuple[str, str]] = {
    "3.8": ("20230826", "3.8.17"),
    "3.9": ("20230826", "3.9.18"),
    "3.10": ("20230826", "3.10.13"),
    "3.11": ("20230826", "3.11.5"),
}


def client_mount_name() -> str:
    """Get the deployed name of the client package mount."""
    return f"modal-client-mount-{__version__}"


def python_standalone_mount_name(version: str) -> str:
    """Get the deployed name of the python-build-standalone mount."""
    if "-" in version:  # default to glibc
        version, libc = version.split("-")
    else:
        libc = "gnu"
    if version not in PYTHON_STANDALONE_VERSIONS:
        raise modal.exception.InvalidError(
            f"Unsupported standalone python version: {version}, supported values are {list(PYTHON_STANDALONE_VERSIONS.keys())}"
        )
    if libc not in ("gnu", "musl"):
        raise modal.exception.InvalidError(f"Unsupported libc identifier: {libc}")
    release, full_version = PYTHON_STANDALONE_VERSIONS[version]
    return f"python-build-standalone.{release}.{full_version}-{libc}"


class _MountEntry(metaclass=abc.ABCMeta):
    remote_path: PurePosixPath

    def description(self) -> str:
        ...

    def get_files_to_upload(self) -> typing.Iterator[Tuple[Path, str]]:
        ...

    def watch_entry(self) -> Tuple[Path, Path]:
        ...


@dataclasses.dataclass
class _MountFile(_MountEntry):
    local_file: Path
    remote_path: PurePosixPath

    def description(self) -> str:
        return str(self.local_file)

    def get_files_to_upload(self):
        local_file = self.local_file.expanduser()
        if not local_file.exists():
            raise FileNotFoundError(local_file)

        rel_filename = self.remote_path
        yield local_file, rel_filename.as_posix()

    def watch_entry(self):
        parent = self.local_file.parent
        return parent, self.local_file


@dataclasses.dataclass
class _MountDir(_MountEntry):
    local_dir: Path
    remote_path: PurePosixPath
    condition: Callable[[str], bool]
    recursive: bool

    def description(self):
        return str(self.local_dir)

    def get_files_to_upload(self):
        local_dir = self.local_dir.expanduser()

        if not local_dir.exists():
            raise FileNotFoundError(local_dir)

        if not local_dir.is_dir():
            raise NotADirectoryError(local_dir)

        if self.recursive:
            gen = (os.path.join(root, name) for root, dirs, files in os.walk(local_dir) for name in files)
        else:
            gen = (dir_entry.path for dir_entry in os.scandir(local_dir) if dir_entry.is_file())

        for local_filename in gen:
            if self.condition(local_filename):
                local_relpath = Path(local_filename).relative_to(local_dir)
                mount_path = self.remote_path / local_relpath.as_posix()
                yield local_filename, mount_path.as_posix()

    def watch_entry(self):
        return self.local_dir, None


class _Mount(_Object, type_prefix="mo"):
    """Create a mount for a local directory or file that can be attached
    to one or more Modal functions.

    **Usage**

    ```python
    import modal
    import os
    stub = modal.Stub()

    @stub.function(mounts=[modal.Mount.from_local_dir("~/foo", remote_path="/root/foo")])
    def f():
        # `/root/foo` has the contents of `~/foo`.
        print(os.listdir("/root/foo/"))
    ```

    Modal syncs the contents of the local directory every time the app runs, but uses the hash of
    the file's contents to skip uploading files that have been uploaded before.
    """

    _entries: List[_MountEntry]

    _content_checksum_sha256_hex: Optional[str]

    @staticmethod
    def _from_entries(*entries: _MountEntry) -> "_Mount":
        rep = f"Mount({entries})"
        load = functools.partial(_Mount._load_mount, entries)
        obj = _Mount._from_loader(load, rep)
        obj._entries = entries
        obj._is_local = True
        return obj

    @staticmethod
    def new() -> "_Mount":
        """mdmd:hidden"""
        return _Mount._from_entries()

    @property
    def entries(self):
        """mdmd:hidden"""
        return self._entries

    def _hydrate_metadata(self, handle_metadata: Optional[Message]):
        assert isinstance(handle_metadata, api_pb2.MountHandleMetadata)
        self._content_checksum_sha256_hex = handle_metadata.content_checksum_sha256_hex

    def is_local(self) -> bool:
        """mdmd:hidden"""
        # TODO(erikbern): since any remote ref bypasses the constructor,
        # we can't rely on it to be set. Let's clean this up later.
        return getattr(self, "_is_local", False)

    @typechecked
    def add_local_dir(
        self,
        local_path: Union[str, Path],
        *,
        # Where the directory is placed within in the mount
        remote_path: Union[str, PurePosixPath, None] = None,
        # Filter function for file selection; defaults to including all files
        condition: Optional[Callable[[str], bool]] = None,
        # add files from subdirectories as well
        recursive: bool = True,
    ) -> "_Mount":
        """
        Add a local directory to the `Mount` object.
        """
        local_path = Path(local_path)
        if remote_path is None:
            remote_path = local_path.name
        remote_path = PurePosixPath("/", remote_path)
        if condition is None:

            def include_all(path):
                return True

            condition = include_all

        return _Mount._from_entries(
            *self._entries,
            _MountDir(
                local_dir=local_path,
                condition=condition,
                remote_path=remote_path,
                recursive=recursive,
            ),
        )

    @staticmethod
    @typechecked
    def from_local_dir(
        local_path: Union[str, Path],
        *,
        # Where the directory is placed within in the mount
        remote_path: Union[str, PurePosixPath, None] = None,
        # Filter function for file selection - default all files
        condition: Optional[Callable[[str], bool]] = None,
        # add files from subdirectories as well
        recursive: bool = True,
    ) -> "_Mount":
        """
        Create a `Mount` from a local directory.

        **Usage**

        ```python
        assets = modal.Mount.from_local_dir(
            "~/assets",
            condition=lambda pth: not ".venv" in pth,
            remote_path="/assets",
        )
        ```
        """
        return _Mount._from_entries().add_local_dir(
            local_path, remote_path=remote_path, condition=condition, recursive=recursive
        )

    @typechecked
    def add_local_file(
        self, local_path: Union[str, Path], remote_path: Union[str, PurePosixPath, None] = None
    ) -> "_Mount":
        """
        Add a local file to the `Mount` object.
        """
        local_path = Path(local_path)
        if remote_path is None:
            remote_path = local_path.name
        remote_path = PurePosixPath("/", remote_path)
        return _Mount._from_entries(
            *self._entries,
            _MountFile(
                local_file=local_path,
                remote_path=PurePosixPath(remote_path),
            ),
        )

    @staticmethod
    @typechecked
    def from_local_file(local_path: Union[str, Path], remote_path: Union[str, PurePosixPath, None] = None) -> "_Mount":
        """
        Create a `Mount` mounting a single local file.

        **Usage**

        ```python
        # Mount the DBT profile in user's home directory into container.
        dbt_profiles = modal.Mount.from_local_file(
            local_path="~/profiles.yml",
            remote_path="/root/dbt_profile/profiles.yml"),
        )
        ```
        """
        return _Mount._from_entries().add_local_file(local_path, remote_path=remote_path)

    @staticmethod
    def _description(entries: List[_MountEntry]) -> str:
        local_contents = [e.description() for e in entries]
        return ", ".join(local_contents)

    @staticmethod
    async def _get_files(entries: List[_MountEntry]) -> AsyncGenerator[FileUploadSpec, None]:
        all_files: List[Tuple[Path, str]] = []
        for entry in entries:
            all_files += list(entry.get_files_to_upload())

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as exe:
            futs = []
            for local_filename, remote_filename in all_files:
                futs.append(loop.run_in_executor(exe, get_file_upload_spec, local_filename, remote_filename))

            logger.debug(f"Computing checksums for {len(futs)} files using {exe._max_workers} workers")
            for fut in asyncio.as_completed(futs):
                try:
                    yield await fut
                except FileNotFoundError as exc:
                    # Can happen with temporary files (e.g. emacs will write temp files and delete them quickly)
                    logger.info(f"Ignoring file not found: {exc}")

    @staticmethod
    async def _load_mount(
        entries: List[_MountEntry],
        provider: "_Mount",
        resolver: Resolver,
        existing_object_id: Optional[str],
    ):
        # Run a threadpool to compute hash values, and use concurrent coroutines to register files.
        t0 = time.time()
        n_concurrent_uploads = 16

        n_files = 0
        uploaded_hashes: set[str] = set()
        total_bytes = 0
        message_label = _Mount._description(entries)
        status_row = resolver.add_status_row()

        async def _put_file(file_spec: FileUploadSpec) -> api_pb2.MountFile:
            nonlocal n_files, uploaded_hashes, total_bytes
            status_row.message(
                f"Creating mount {message_label}: Uploaded {len(uploaded_hashes)}/{n_files} inspected files"
            )

            remote_filename = file_spec.mount_filename
            mount_file = api_pb2.MountFile(
                filename=remote_filename,
                sha256_hex=file_spec.sha256_hex,
                mode=file_spec.mode,
            )

            if file_spec.sha256_hex in uploaded_hashes:
                return mount_file

            request = api_pb2.MountPutFileRequest(sha256_hex=file_spec.sha256_hex)
            response = await retry_transient_errors(resolver.client.stub.MountPutFile, request, base_delay=1)

            n_files += 1
            if response.exists:
                return mount_file

            uploaded_hashes.add(file_spec.sha256_hex)
            total_bytes += file_spec.size

            if file_spec.use_blob:
                logger.debug(f"Creating blob file for {file_spec.filename} ({file_spec.size} bytes)")
                with open(file_spec.filename, "rb") as fp:
                    blob_id = await blob_upload_file(fp, resolver.client.stub)
                logger.debug(f"Uploading blob file {file_spec.filename} as {remote_filename}")
                request2 = api_pb2.MountPutFileRequest(data_blob_id=blob_id, sha256_hex=file_spec.sha256_hex)
            else:
                logger.debug(f"Uploading file {file_spec.filename} to {remote_filename} ({file_spec.size} bytes)")
                request2 = api_pb2.MountPutFileRequest(data=file_spec.content, sha256_hex=file_spec.sha256_hex)

            start_time = time.monotonic()
            while time.monotonic() - start_time < MOUNT_PUT_FILE_CLIENT_TIMEOUT:
                response = await retry_transient_errors(resolver.client.stub.MountPutFile, request2, base_delay=1)
                if response.exists:
                    return mount_file

            raise modal.exception.MountUploadTimeoutError(f"Mounting of {file_spec.filename} timed out")

        logger.debug(f"Uploading mount using {n_concurrent_uploads} uploads")

        # Create async generator
        files_stream = aiostream.stream.iterate(_Mount._get_files(entries))

        # Upload files
        uploads_stream = aiostream.stream.map(files_stream, _put_file, task_limit=n_concurrent_uploads)
        files: List[api_pb2.MountFile] = await aiostream.stream.list(uploads_stream)
        if not files:
            logger.warning(f"Mount of '{message_label}' is empty.")

        # Build mounts
        status_row.message(f"Creating mount {message_label}: Building mount")
        req = api_pb2.MountBuildRequest(app_id=resolver.app_id, existing_mount_id=existing_object_id, files=files)
        resp = await retry_transient_errors(resolver.client.stub.MountBuild, req, base_delay=1)
        status_row.finish(f"Created mount {message_label}")

        logger.debug(f"Uploaded {len(uploaded_hashes)}/{n_files} files and {total_bytes} bytes in {time.time() - t0}s")
        provider._hydrate(resp.mount_id, resolver.client, resp.handle_metadata)

    @staticmethod
    def from_local_python_packages(*module_names: str) -> "_Mount":
        """Returns a `modal.Mount` that makes local modules listed in `module_names` available inside the container.
        This works by mounting the local path of each module's package to a directory inside the container that's on `PYTHONPATH`.

        **Usage**

        ```python notest
        import modal
        import my_local_module

        stub = modal.Stub()

        @stub.function(mounts=[
            modal.Mount.from_local_python_packages("my_local_module", "my_other_module"),
        ])
        def f():
            my_local_module.do_stuff()
        ```
        """
        from modal.app import is_local

        # Don't re-run inside container.

        mount = _Mount.new()
        if not is_local():
            return mount

        for module_name in module_names:
            mount_infos = get_module_mount_info(module_name)

            if mount_infos == []:
                raise NotFoundError(f"Module {module_name} not found.")

            for mount_info in mount_infos:
                is_package, base_path, module_mount_condition = mount_info
                if is_package:
                    mount = mount.add_local_dir(
                        base_path,
                        remote_path=f"/pkg/{module_name}",
                        condition=module_mount_condition,
                        recursive=True,
                    )
                else:
                    remote_path = PurePosixPath("/pkg") / Path(base_path).name
                    mount = mount.add_local_file(
                        base_path,
                        remote_path=remote_path,
                    )
        return mount


Mount = synchronize_api(_Mount)


def _create_client_mount():
    # TODO(erikbern): make this a static method on the Mount class
    import synchronicity

    import modal

    # Get the base_path because it also contains `modal_utils` and `modal_proto`.
    base_path, _ = os.path.split(modal.__path__[0])

    # TODO(erikbern): this is incredibly dumb, but we only want to include packages that start with "modal"
    # TODO(erikbern): merge functionality with _function_utils._is_modal_path
    prefix = os.path.join(base_path, "modal")

    def condition(arg):
        return module_mount_condition(arg) and arg.startswith(prefix)

    return (
        _Mount.from_local_dir(base_path, remote_path="/pkg/", condition=condition, recursive=True)
        # Mount synchronicity, so version changes don't trigger image rebuilds for users.
        .add_local_dir(
            synchronicity.__path__[0],
            remote_path="/pkg/synchronicity",
            condition=module_mount_condition,
            recursive=True,
        )
    )


create_client_mount = synchronize_api(_create_client_mount)


def _get_client_mount():
    # TODO(erikbern): make this a static method on the Mount class
    if config["sync_entrypoint"]:
        return _create_client_mount()
    else:
        return _Mount.from_name(client_mount_name(), namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)


_create_package_mounts_deprecation_msg = (
    "modal.create_package_mounts() is being deprecated, use modal.Mount.from_local_python_packages() instead"
)


@typechecked
def _create_package_mounts(module_names: Sequence[str]) -> List[_Mount]:
    f"""{_create_package_mounts_deprecation_msg}"""
    modal.exception.deprecation_warning(
        date(2023, 7, 19),
        _create_package_mounts_deprecation_msg,
        pending=True,
    )
    return [_Mount.from_local_python_packages(*module_names)]


create_package_mounts = synchronize_api(_create_package_mounts)
