import asyncio
import concurrent.futures
import os
import time
from pathlib import Path

import aiostream

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.package_utils import (
    get_module_mount_info,
    get_sha256_hex_from_filename,
    module_mount_condition,
)

from ._factory import _factory
from .config import logger
from .object import Object


async def _get_files(local_dir, condition, recursive):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as exe:
        futs = []
        if recursive:
            gen = (os.path.join(root, name) for root, dirs, files in os.walk(local_dir) for name in files)
        else:
            gen = (dir_entry.path for dir_entry in os.scandir(local_dir) if dir_entry.is_file())

        for filename in gen:
            rel_filename = os.path.relpath(filename, local_dir)
            if condition(filename):
                futs.append(loop.run_in_executor(exe, get_sha256_hex_from_filename, filename, rel_filename))
        logger.debug(f"Computing checksums for {len(futs)} files using {exe._max_workers} workers")
        for fut in asyncio.as_completed(futs):
            filename, rel_filename, sha256_hex = await fut
            yield filename, rel_filename, sha256_hex


class _Mount(Object, type_prefix="mo"):
    def __init__(self, app, local_dir, remote_dir, condition=lambda path: True, recursive=True):
        self._local_dir = local_dir
        self._remote_dir = remote_dir
        self._condition = condition
        self._recursive = recursive
        super().__init__(app=app)

    async def load(self, app):
        # Run a threadpool to compute hash values, and use n coroutines to put files
        n_files = 0
        n_missing_files = 0
        total_bytes = 0
        t0 = time.time()
        n_concurrent_uploads = 16

        async def _put_file(client, mount_id, filename, rel_filename, sha256_hex):
            nonlocal n_files, n_missing_files, total_bytes

            remote_filename = (Path(self._remote_dir) / Path(rel_filename)).as_posix()

            request = api_pb2.MountRegisterFileRequest(
                filename=remote_filename, sha256_hex=sha256_hex, mount_id=mount_id
            )
            response = await client.stub.MountRegisterFile(request)
            n_files += 1
            if not response.exists:
                # TODO: this will be moved to S3 soon
                data = open(filename, "rb").read()
                n_missing_files += 1
                total_bytes += len(data)
                logger.debug(f"Uploading file {filename} to {remote_filename} ({len(data)} bytes)")

                request2 = api_pb2.MountUploadFileRequest(
                    data=data, sha256_hex=sha256_hex, size=len(data), mount_id=mount_id
                )
                await client.stub.MountUploadFile(request2)

        req = api_pb2.MountCreateRequest(app_id=app.app_id)
        resp = await app.client.stub.MountCreate(req)
        mount_id = resp.mount_id

        logger.debug(f"Uploading mount {mount_id} using {n_concurrent_uploads} uploads")

        # Create async generator
        files = _get_files(self._local_dir, self._condition, self._recursive)
        files_stream = aiostream.stream.iterate(files)

        async def put_file_tupled(tup):
            filename, rel_filename, sha256_hex = tup
            await _put_file(app.client, mount_id, filename, rel_filename, sha256_hex)

        # Upload files
        uploads_stream = aiostream.stream.map(files_stream, put_file_tupled, task_limit=n_concurrent_uploads)
        await uploads_stream

        logger.debug(f"Uploaded {n_missing_files}/{n_files} files and {total_bytes} bytes in {time.time() - t0}s")

        # Set the mount to done
        req_done = api_pb2.MountDoneRequest(mount_id=mount_id)
        await app.client.stub.MountDone(req_done)

        return mount_id


Mount, AioMount = synchronize_apis(_Mount)


async def _create_client_mount(app):
    import modal

    # Get the base_path because it also contains `modal_utils` and `modal_proto`.
    base_path, _ = os.path.split(modal.__path__[0])

    mount = _Mount(app, base_path, "/pkg/", module_mount_condition, recursive=True)
    await app.create_object(mount)
    return mount


_, aio_create_client_mount = synchronize_apis(_create_client_mount)


@_factory(_Mount)
async def _create_package_mount(app, module_name):
    mount_infos = get_module_mount_info(module_name)

    assert len(mount_infos) == 1

    _, base_path, module_mount_condition = mount_infos[0]
    return _Mount(app, base_path, f"/pkg/{module_name}", module_mount_condition, recursive=True)


create_package_mount, aio_create_package_mount = synchronize_apis(_create_package_mount)
