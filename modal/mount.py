import asyncio
import concurrent.futures
import os
import time
import warnings
from pathlib import Path
from typing import AsyncIterable, Callable, Collection, Optional, Union

import aiostream

import modal._blob_utils
from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.grpc_utils import retry_transient_errors
from modal_utils.package_utils import get_module_mount_info, module_mount_condition

from ._blob_utils import FileUploadSpec, get_file_upload_spec
from .config import logger
from .exception import InvalidError
from .object import Object
from .version import __version__


def client_mount_name():
    return f"modal-client-mount-{__version__}"


class _Mount(Object, type_prefix="mo"):
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
        remote_dir: Union[str, Path],
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
        if local_file is not None and local_dir is not None:
            raise InvalidError("Cannot specify both local_file and local_dir as arguments to Mount.")

        if local_file is None and local_dir is None:
            raise InvalidError("Must provide at least one of local_file and local_dir to Mount.")

        self._local_dir = local_dir
        self._local_file = local_file
        self._remote_dir = remote_dir
        self._condition = condition
        self._recursive = recursive
        super().__init__()

    def _get_creating_message(self):
        label = getattr(self, "_local_dir", None) or getattr(self, "_local_file", None)
        if label is None:
            return None
        return f"Mounting {label}..."

    def _get_created_message(self):
        label = getattr(self, "_local_dir", None) or getattr(self, "_local_file", None)
        if label is None:
            return None
        return f"Mounted {label}."

    async def _get_files(self):
        if self._local_file:
            relpath = os.path.basename(self._local_file)
            yield get_file_upload_spec(self._local_file, relpath)
            return

        local_dir = os.path.expanduser(self._local_dir)
        if not os.path.exists(local_dir):
            raise FileNotFoundError(local_dir)
        if not os.path.isdir(local_dir):
            raise NotADirectoryError(local_dir)

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as exe:
            futs = []
            if self._recursive:
                gen = (os.path.join(root, name) for root, dirs, files in os.walk(local_dir) for name in files)
            else:
                gen = (dir_entry.path for dir_entry in os.scandir(local_dir) if dir_entry.is_file())

            for filename in gen:
                rel_filename = os.path.relpath(filename, local_dir)
                if self._condition(filename):
                    futs.append(loop.run_in_executor(exe, get_file_upload_spec, filename, rel_filename))
            logger.debug(f"Computing checksums for {len(futs)} files using {exe._max_workers} workers")
            for fut in asyncio.as_completed(futs):
                yield await fut

    async def _load(self, client, app_id, loader, existing_mount_id):
        # Run a threadpool to compute hash values, and use concurrent coroutines to register files.
        t0 = time.time()
        n_concurrent_uploads = 16

        n_files = 0
        uploaded_hashes: set[str] = set()
        files: list[api_pb2.MountFile] = []
        total_bytes = 0

        async def _put_file(mount_file: FileUploadSpec):
            nonlocal n_files, uploaded_hashes, total_bytes

            remote_filename = (Path(self._remote_dir) / Path(mount_file.rel_filename)).as_posix()
            files.append(api_pb2.MountFile(filename=remote_filename, sha256_hex=mount_file.sha256_hex))

            request = api_pb2.MountPutFileRequest(sha256_hex=mount_file.sha256_hex)
            response = await retry_transient_errors(client.stub.MountPutFile, request, base_delay=1)

            n_files += 1
            if response.exists or mount_file.sha256_hex in uploaded_hashes:
                return
            uploaded_hashes.add(mount_file.sha256_hex)
            total_bytes += mount_file.size

            if mount_file.use_blob:
                logger.debug(f"Creating blob file for {mount_file.filename} ({mount_file.size} bytes)")
                blob_id = await modal._blob_utils.blob_upload_file(mount_file.filename, client.stub)
                logger.debug(f"Uploading blob file {mount_file.filename} as {remote_filename}")
                request2 = api_pb2.MountPutFileRequest(data_blob_id=blob_id, sha256_hex=mount_file.sha256_hex)
            else:
                logger.debug(f"Uploading file {mount_file.filename} to {remote_filename} ({mount_file.size} bytes)")
                request2 = api_pb2.MountPutFileRequest(data=mount_file.content, sha256_hex=mount_file.sha256_hex)
            await retry_transient_errors(client.stub.MountPutFile, request2, base_delay=1)

        logger.debug(f"Uploading mount using {n_concurrent_uploads} uploads")

        # Create async generator
        files_stream = aiostream.stream.iterate(self._get_files())

        # Upload files
        uploads_stream = aiostream.stream.map(files_stream, _put_file, task_limit=n_concurrent_uploads)
        try:
            await uploads_stream
        except aiostream.StreamEmpty:
            logger.warning("Mount is empty.")

        req = api_pb2.MountBuildRequest(app_id=app_id, existing_mount_id=existing_mount_id, files=files)
        resp = await retry_transient_errors(client.stub.MountBuild, req, base_delay=1)

        logger.debug(f"Uploaded {len(uploaded_hashes)}/{n_files} files and {total_bytes} bytes in {time.time() - t0}s")
        return resp.mount_id


Mount, AioMount = synchronize_apis(_Mount)


def _create_client_mount():
    import modal

    # Get the base_path because it also contains `modal_utils` and `modal_proto`.
    base_path, _ = os.path.split(modal.__path__[0])

    mount = _Mount(local_dir=base_path, remote_dir="/pkg/", condition=module_mount_condition, recursive=True)
    return mount


_, aio_create_client_mount = synchronize_apis(_create_client_mount)


async def _create_package_mounts(module_names: Collection[str]) -> AsyncIterable[_Mount]:
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
    for module_name in module_names:
        mount_infos = get_module_mount_info(module_name)

        for mount_info in mount_infos:
            _, base_path, module_mount_condition = mount_info
            yield _Mount(
                local_dir=base_path, remote_dir=f"/pkg/{module_name}", condition=module_mount_condition, recursive=True
            )


async def _create_package_mount(module_name: str):
    warnings.warn(
        "`create_package_mount` is deprecated. Please use `create_package_mounts` instead.", DeprecationWarning
    )
    mounts = [m async for m in _create_package_mounts([module_name])]
    assert len(mounts) == 1
    return mounts[0]


create_package_mount, aio_create_package_mount = synchronize_apis(_create_package_mount)
create_package_mounts, aio_create_package_mounts = synchronize_apis(_create_package_mounts)
