import asyncio
import concurrent.futures
import os
import time
from pathlib import Path

import aiostream

from modal_proto import api_pb2
from modal_utils.async_utils import retry, synchronize_apis
from modal_utils.package_utils import (
    get_module_mount_info,
    get_sha256_hex_from_filename,
    module_mount_condition,
)

from ._factory import _factory
from .config import logger
from .exception import InvalidError
from .object import Object


class _Mount(Object, type_prefix="mo"):
    def __init__(
        self, app, remote_dir, *, local_dir=None, local_file=None, condition=lambda path: True, recursive=True
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
        super().__init__(app=app)

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
            yield get_sha256_hex_from_filename(self._local_file, relpath)
            return

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as exe:
            futs = []
            if self._recursive:
                gen = (os.path.join(root, name) for root, dirs, files in os.walk(self._local_dir) for name in files)
            else:
                gen = (dir_entry.path for dir_entry in os.scandir(self._local_dir) if dir_entry.is_file())

            for filename in gen:
                rel_filename = os.path.relpath(filename, self._local_dir)
                if self._condition(filename):
                    futs.append(loop.run_in_executor(exe, get_sha256_hex_from_filename, filename, rel_filename))
            logger.debug(f"Computing checksums for {len(futs)} files using {exe._max_workers} workers")
            for fut in asyncio.as_completed(futs):
                yield await fut

    async def load(self, app):
        # Run a threadpool to compute hash values, and use n coroutines to put files
        n_files = 0
        n_missing_files = 0
        total_bytes = 0
        t0 = time.time()
        n_concurrent_uploads = 16

        async def _put_file(client, mount_id, filename, rel_filename, data, sha256_hex):
            nonlocal n_files, n_missing_files, total_bytes

            remote_filename = (Path(self._remote_dir) / Path(rel_filename)).as_posix()

            request = api_pb2.MountRegisterFileRequest(
                filename=remote_filename, sha256_hex=sha256_hex, mount_id=mount_id
            )
            response = await retry(client.stub.MountRegisterFile, base_delay=1)(request)
            n_files += 1
            if not response.exists:
                # TODO: use S3 for large files.
                n_missing_files += 1
                total_bytes += len(data)
                logger.debug(f"Uploading file {filename} to {remote_filename} ({len(data)} bytes)")

                request2 = api_pb2.MountUploadFileRequest(
                    data=data, sha256_hex=sha256_hex, size=len(data), mount_id=mount_id
                )
                await retry(client.stub.MountUploadFile, base_delay=1)(request2)

        req = api_pb2.MountCreateRequest(app_id=app.app_id)
        resp = await retry(app.client.stub.MountCreate, base_delay=1)(req)
        mount_id = resp.mount_id

        logger.debug(f"Uploading mount {mount_id} using {n_concurrent_uploads} uploads")

        # Create async generator
        files_stream = aiostream.stream.iterate(self._get_files())

        async def put_file_tupled(tup):
            filename, rel_filename, content, sha256_hex = tup
            await _put_file(app.client, mount_id, filename, rel_filename, content, sha256_hex)

        # Upload files
        uploads_stream = aiostream.stream.map(files_stream, put_file_tupled, task_limit=n_concurrent_uploads)
        await uploads_stream

        logger.debug(f"Uploaded {n_missing_files}/{n_files} files and {total_bytes} bytes in {time.time() - t0}s")

        # Set the mount to done
        req_done = api_pb2.MountDoneRequest(mount_id=mount_id)
        await retry(app.client.stub.MountDone, base_delay=1)(req_done)

        return mount_id


Mount, AioMount = synchronize_apis(_Mount)


async def _create_client_mount(app):
    import modal

    # Get the base_path because it also contains `modal_utils` and `modal_proto`.
    base_path, _ = os.path.split(modal.__path__[0])

    mount = _Mount(app, local_dir=base_path, remote_dir="/pkg/", condition=module_mount_condition, recursive=True)
    await app.create_object(mount)
    return mount


_, aio_create_client_mount = synchronize_apis(_create_client_mount)


@_factory(_Mount)
async def _create_package_mount(app, module_name):
    mount_infos = get_module_mount_info(module_name)

    assert len(mount_infos) == 1

    _, base_path, module_mount_condition = mount_infos[0]
    return _Mount(
        app, local_dir=base_path, remote_dir=f"/pkg/{module_name}", condition=module_mount_condition, recursive=True
    )


create_package_mount, aio_create_package_mount = synchronize_apis(_create_package_mount)
