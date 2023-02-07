# Copyright Modal Labs 2022
import asyncio
import concurrent.futures
import dataclasses
import os
import time
from pathlib import Path
from typing import Callable, Collection, List, Optional, Union, Tuple

import aiostream

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.grpc_utils import retry_transient_errors
from modal_utils.package_utils import get_module_mount_info, module_mount_condition
from modal_version import __version__

from ._blob_utils import FileUploadSpec, blob_upload_file, get_file_upload_spec
from ._resolver import Resolver
from .config import config, logger
from .exception import InvalidError, NotFoundError
from .object import Handle, Provider


def client_mount_name():
    return f"modal-client-mount-{__version__}"


@dataclasses.dataclass
class _MountFile:
    local_file: Path
    remote_path: Path

    def description(self) -> str:
        return self.local_file.as_posix()

    def get_files_to_upload(self):
        local_file = self.local_file.expanduser()
        if not local_file.exists():
            raise FileNotFoundError(local_file)
        remote_filename = self.remote_path / self.local_file.name
        yield local_file, remote_filename


@dataclasses.dataclass
class _MountDir:
    local_dir: Path
    remote_path: Path
    condition: Callable[[str], bool]
    recursive: bool

    def description(self):
        return self.local_dir.as_posix()

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
                remote_filename = self.remote_path / Path(local_filename).relative_to(local_dir)
                yield local_filename, remote_filename


_MountEntry = Union[_MountFile, _MountDir]


class _MountHandle(Handle, type_prefix="mo"):
    pass


class _Mount(Provider[_MountHandle]):
    """Create a mount for a local directory or file that can be attached
    to one or more Modal functions.

    **Usage**

    ```python
    import modal
    import os
    stub = modal.Stub()

    @stub.function(mounts=[modal.Mount(remote_dir="/root/foo", local_dir="~/foo")])
    def f():
        # `/root/foo` has the contents of `~/foo`.
        print(os.listdir("/root/foo/"))
    ```

    Modal syncs the contents of the local directory every time the app runs, but uses the hash of
    the file's contents to skip uploading files that have been uploaded before.
    """

    def __init__(
        self,
        # Mount path within the container.
        remote_dir: Union[str, Path] = "/",
        *,
        # Local directory to mount.
        local_dir: Optional[Union[str, Path]] = None,
        # Local file to mount, if only a single file needs to be mounted. Note that exactly one of `local_dir` and `local_file` can be provided.
        local_file: Optional[Union[str, Path]] = None,
        # Optional predicate to filter files while creating the mount. `condition` is any function that accepts an absolute local file path, and returns `True` if it should be mounted, and `False` otherwise.
        condition: Callable[[str], bool] = lambda path: True,
        # Optional flag to toggle if subdirectories should be mounted recursively.
        recursive: bool = True,
    ):
        self._entries: List[_MountEntry] = []
        if local_file or local_dir:
            # TODO: add deprecation warning here for legacy API
            if local_file is not None and local_dir is not None:
                raise InvalidError("Cannot specify both local_file and local_dir as arguments to Mount.")

            if local_dir:
                self.add_local_dir(
                    local_path=local_dir,
                    remote_path=remote_dir,
                    condition=condition,
                    recursive=recursive,
                )
            elif local_file:
                self.add_local_file(local_path=local_file, remote_path=remote_dir)

        self._is_local = True
        rep = f"Mount({self._entries})"
        super().__init__(self._load, rep)

    def is_local(self):
        # TODO(erikbern): since any remote ref bypasses the constructor,
        # we can't rely on it to be set. Let's clean this up later.
        return getattr(self, "_is_local", False)

    def add_local_dir(
        self,
        local_path: Union[str, Path],
        *,
        condition: Callable[[str], bool] = lambda path: True,  # Filter function for file selection
        remote_path: Union[str, Path],  # Where the directory is placed within in the mount
        recursive: bool = True,  # add files from subdirectories as well
    ):
        self._entries.append(
            _MountDir(
                local_dir=Path(local_path),
                condition=condition,
                remote_path=Path(remote_path),
                recursive=recursive,
            )
        )
        return self

    def add_local_file(self, local_path: Union[str, Path], remote_path: Union[str, Path]):
        self._entries.append(
            _MountFile(
                local_file=Path(local_path),
                remote_path=Path(remote_path),
            )
        )
        return self

    def __repr__(self):
        return f"Mount({self._description()}"

    def _description(self):
        local_contents = [e.description() for e in self._entries]
        return ", ".join(local_contents)

    async def _get_files(self):
        all_files: List[Tuple[Path, Path]] = []
        for entry in self._entries:
            all_files += list(entry.get_files_to_upload())

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as exe:
            futs = []
            for local_filename, remote_filename in all_files:
                futs.append(loop.run_in_executor(exe, get_file_upload_spec, local_filename, remote_filename))

            logger.debug(f"Computing checksums for {len(futs)} files using {exe._max_workers} workers")
            for i, fut in enumerate(asyncio.as_completed(futs)):
                try:
                    yield await fut
                except FileNotFoundError as exc:
                    # Can happen with temporary files (e.g. emacs will write temp files and delete them quickly)
                    logger.info(f"Ignoring file not found: {exc}")

    async def _load(self, resolver: Resolver):
        # Run a threadpool to compute hash values, and use concurrent coroutines to register files.
        t0 = time.time()
        n_concurrent_uploads = 16

        n_files = 0
        uploaded_hashes: set[str] = set()
        files: list[api_pb2.MountFile] = []
        total_bytes = 0
        message_label = self._description()

        async def _put_file(mount_file: FileUploadSpec):
            nonlocal n_files, uploaded_hashes, total_bytes
            resolver.set_message(
                f"Creating mount {message_label}: Uploaded {len(uploaded_hashes)}/{n_files} inspected files"
            )

            remote_filename = mount_file.mount_filename.as_posix()
            files.append(api_pb2.MountFile(filename=remote_filename, sha256_hex=mount_file.sha256_hex))

            request = api_pb2.MountPutFileRequest(sha256_hex=mount_file.sha256_hex)
            response = await retry_transient_errors(resolver.client.stub.MountPutFile, request, base_delay=1)

            n_files += 1
            if response.exists or mount_file.sha256_hex in uploaded_hashes:
                return
            uploaded_hashes.add(mount_file.sha256_hex)
            total_bytes += mount_file.size

            if mount_file.use_blob:
                logger.debug(f"Creating blob file for {mount_file.filename} ({mount_file.size} bytes)")
                with open(mount_file.filename, "rb") as fp:
                    blob_id = await blob_upload_file(fp, resolver.client.stub)
                logger.debug(f"Uploading blob file {mount_file.filename} as {remote_filename}")
                request2 = api_pb2.MountPutFileRequest(data_blob_id=blob_id, sha256_hex=mount_file.sha256_hex)
            else:
                logger.debug(f"Uploading file {mount_file.filename} to {remote_filename} ({mount_file.size} bytes)")
                request2 = api_pb2.MountPutFileRequest(data=mount_file.content, sha256_hex=mount_file.sha256_hex)
            await retry_transient_errors(resolver.client.stub.MountPutFile, request2, base_delay=1)

        logger.debug(f"Uploading mount using {n_concurrent_uploads} uploads")

        # Create async generator
        files_stream = aiostream.stream.iterate(self._get_files())

        # Upload files
        uploads_stream = aiostream.stream.map(files_stream, _put_file, task_limit=n_concurrent_uploads)
        try:
            await uploads_stream
        except aiostream.StreamEmpty:
            logger.warning(f"Mount of '{message_label}' is empty.")

        resolver.set_message(f"Creating mount {message_label}: Building mount")
        req = api_pb2.MountBuildRequest(
            app_id=resolver.app_id, existing_mount_id=resolver.existing_object_id, files=files
        )
        resp = await retry_transient_errors(resolver.client.stub.MountBuild, req, base_delay=1)
        resolver.set_message(f"Created mount {message_label}")

        logger.debug(f"Uploaded {len(uploaded_hashes)}/{n_files} files and {total_bytes} bytes in {time.time() - t0}s")
        return _MountHandle._from_id(resp.mount_id, resolver.client, None)


Mount, AioMount = synchronize_apis(_Mount)


def _create_client_mount():
    import modal

    # Get the base_path because it also contains `modal_utils` and `modal_proto`.
    base_path, _ = os.path.split(modal.__path__[0])

    # TODO(erikbern): this is incredibly dumb, but we only want to include packages that start with "modal"
    prefix = os.path.join(base_path, "modal")

    def condition(arg):
        return module_mount_condition(arg) and arg.startswith(prefix)

    return _Mount(local_dir=base_path, remote_dir="/pkg/", condition=condition, recursive=True)


_, aio_create_client_mount = synchronize_apis(_create_client_mount)


def _get_client_mount():
    if config["sync_entrypoint"]:
        return _create_client_mount()
    else:
        return _Mount.from_name(client_mount_name(), namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)


async def _create_package_mounts(module_names: Collection[str]) -> List[_Mount]:
    """Returns a `modal.Mount` that makes local modules listed in `module_names` available inside the container.
    This works by mounting the local path of each module's package to a directory inside the container that's on `PYTHONPATH`.

    **Usage**

    ```python notest
    import modal
    import my_local_module

    stub = modal.Stub()

    @stub.function(mounts=[
        *modal.create_package_mounts(["my_local_module", "my_other_module"]),
        modal.Mount(local_dir="/my_local_dir", remote_dir="/"),
    ])
    def f():
        my_local_module.do_stuff()
    ```
    """
    from modal import is_local

    # Don't re-run inside container.
    if not is_local():
        return []

    mounts = []
    for module_name in module_names:
        mount_infos = get_module_mount_info(module_name)

        if mount_infos == []:
            raise NotFoundError(f"Module {module_name} not found.")

        for mount_info in mount_infos:
            is_package, base_path, module_mount_condition = mount_info
            if is_package:
                mounts.append(
                    _Mount(
                        local_dir=base_path,
                        remote_dir=f"/pkg/{module_name}",
                        condition=module_mount_condition,
                        recursive=True,
                    )
                )
            else:
                mounts.append(
                    _Mount(
                        local_file=base_path,
                        remote_dir="/pkg",
                        condition=module_mount_condition,
                    )
                )
    return mounts


create_package_mounts, aio_create_package_mounts = synchronize_apis(_create_package_mounts)
