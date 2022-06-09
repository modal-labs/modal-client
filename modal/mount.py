import asyncio
import concurrent.futures
import os
import time
from pathlib import Path
from typing import Callable, Optional, Union

import aiostream

import modal._blob_utils
from modal_proto import api_pb2
from modal_utils.async_utils import retry, synchronize_apis
from modal_utils.package_utils import get_module_mount_info, module_mount_condition

from ._blob_utils import FileUploadSpec, get_file_upload_spec
from .config import logger
from .exception import InvalidError
from .object import Object
from .version import __version__


def client_mount_name():
    return f"modal-client-mount-{__version__}"


class _Mount(Object, type_prefix="mo"):
    def __init__(
        self,
        remote_dir: Union[str, Path],
        *,
        local_dir: Optional[Union[str, Path]] = None,
        local_file: Optional[Union[str, Path]] = None,
        condition: Callable[[str], bool] = lambda path: True,
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

    def get_creating_message(self):
        label = getattr(self, "_local_dir", None) or getattr(self, "_local_file", None)
        if label is None:
            return None
        return f"Mounting {label}..."

    def get_created_message(self):
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

    async def load(self, client, app_id, existing_mount_id):
        # Run a threadpool to compute hash values, and use n coroutines to put files
        # TODO(erikbern): this is not ideal when mounts are created in-place, because it
        # creates a brief period where the files are reset to an empty list.
        # A better way to do it is to upload the files before we create the mount.
        # Let's consider doing this as a part of refactoring how we store mounts and files.

        t0 = time.time()
        n_concurrent_uploads = 16

        n_files = 0
        uploaded_hashes: set[str] = set()
        total_bytes = 0

        async def _put_file(mount_file: FileUploadSpec):
            nonlocal n_files, uploaded_hashes, total_bytes

            remote_filename = (Path(self._remote_dir) / Path(mount_file.rel_filename)).as_posix()

            request = api_pb2.MountRegisterFileRequest(
                filename=remote_filename, sha256_hex=mount_file.sha256_hex, mount_id=mount_id
            )
            response = await retry(client.stub.MountRegisterFile, base_delay=1)(request)

            n_files += 1
            if response.exists or mount_file.sha256_hex in uploaded_hashes:
                return
            uploaded_hashes.add(mount_file.sha256_hex)
            total_bytes += mount_file.size

            if mount_file.use_blob:
                logger.debug(f"Creating blob file for {mount_file.filename} ({mount_file.size} bytes)")
                blob_id = await modal._blob_utils.blob_upload_file(mount_file.filename, client.stub)
                logger.debug(f"Uploading blob file {mount_file.filename} as {remote_filename}")
                request2 = api_pb2.MountUploadFileRequest(
                    data_blob_id=blob_id, sha256_hex=mount_file.sha256_hex, size=mount_file.size, mount_id=mount_id
                )
            else:
                logger.debug(f"Uploading file {mount_file.filename} to {remote_filename} ({mount_file.size} bytes)")
                request2 = api_pb2.MountUploadFileRequest(
                    data=mount_file.content,
                    sha256_hex=mount_file.sha256_hex,
                    size=mount_file.size,
                    mount_id=mount_id,
                )
            await retry(client.stub.MountUploadFile, base_delay=1)(request2)

        req = api_pb2.MountCreateRequest(app_id=app_id, existing_mount_id=existing_mount_id)
        resp = await retry(client.stub.MountCreate, base_delay=1)(req)
        mount_id = resp.mount_id

        logger.debug(f"Uploading mount {mount_id} using {n_concurrent_uploads} uploads")

        # Create async generator
        files_stream = aiostream.stream.iterate(self._get_files())

        # Upload files
        uploads_stream = aiostream.stream.map(files_stream, _put_file, task_limit=n_concurrent_uploads)
        try:
            await uploads_stream
        except aiostream.StreamEmpty:
            logger.warn("Mount is empty.")

        logger.debug(f"Uploaded {len(uploaded_hashes)}/{n_files} files and {total_bytes} bytes in {time.time() - t0}s")

        # Set the mount to done
        req_done = api_pb2.MountDoneRequest(mount_id=mount_id)
        await retry(client.stub.MountDone, base_delay=1)(req_done)

        return mount_id


Mount, AioMount = synchronize_apis(_Mount)


def _create_client_mount():
    import modal

    # Get the base_path because it also contains `modal_utils` and `modal_proto`.
    base_path, _ = os.path.split(modal.__path__[0])

    mount = _Mount(local_dir=base_path, remote_dir="/pkg/", condition=module_mount_condition, recursive=True)
    return mount


_, aio_create_client_mount = synchronize_apis(_create_client_mount)


async def _create_package_mount(module_name):
    mount_infos = get_module_mount_info(module_name)

    assert len(mount_infos) == 1

    _, base_path, module_mount_condition = mount_infos[0]
    return _Mount(
        local_dir=base_path, remote_dir=f"/pkg/{module_name}", condition=module_mount_condition, recursive=True
    )


create_package_mount, aio_create_package_mount = synchronize_apis(_create_package_mount)
