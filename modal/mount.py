import asyncio
import concurrent.futures
import hashlib
import os
import time

import aiostream

from ._package_utils import get_module_mount_info
from .config import logger
from .object import Object
from .proto import api_pb2


def _get_sha256_hex_from_content(content):
    m = hashlib.sha256()
    m.update(content)
    return m.hexdigest()


def get_sha256_hex_from_filename(filename, rel_filename):
    # Somewhat CPU intensive, so we run it in a thread/process
    content = open(filename, "rb").read()
    return filename, rel_filename, _get_sha256_hex_from_content(content)


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


class Mount(Object, type_prefix="mo"):
    @classmethod
    async def create(cls, local_dir, remote_dir, condition, app=None, recursive=True):
        # Run a threadpool to compute hash values, and use n coroutines to put files
        app = cls._get_app(app)

        n_files = 0
        n_missing_files = 0
        total_bytes = 0
        t0 = time.time()
        n_concurrent_uploads = 16

        async def _put_file(client, mount_id, filename, rel_filename, sha256_hex):
            nonlocal n_files, n_missing_files, total_bytes
            remote_filename = os.path.join(remote_dir, rel_filename)  # won't work on windows
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
                request = api_pb2.MountUploadFileRequest(
                    data=data, sha256_hex=sha256_hex, size=len(data), mount_id=mount_id
                )
                response = await client.stub.MountUploadFile(request)

        req = api_pb2.MountCreateRequest(app_id=app.app_id)
        resp = await app.client.stub.MountCreate(req)
        mount_id = resp.mount_id

        logger.debug(f"Uploading mount {mount_id} using {n_concurrent_uploads} uploads")

        # Create async generator
        files = _get_files(local_dir, condition, recursive)
        files_stream = aiostream.stream.iterate(files)

        async def put_file_tupled(tup):
            filename, rel_filename, sha256_hex = tup
            await _put_file(app.client, mount_id, filename, rel_filename, sha256_hex)

        # Upload files
        uploads_stream = aiostream.stream.map(files_stream, put_file_tupled, task_limit=n_concurrent_uploads)
        await uploads_stream

        logger.debug(f"Uploaded {n_missing_files}/{n_files} files and {total_bytes} bytes in {time.time() - t0}s")

        # Set the mount to done
        req = api_pb2.MountDoneRequest(mount_id=mount_id)
        await app.client.stub.MountDone(req)

        return cls._create_object_instance(mount_id, app)


async def create_package_mounts(package_name):
    mount_infos = get_module_mount_info(package_name)
    return [await Mount.create(path, f"/pkg/{name}", condition) for (name, path, condition) in mount_infos]
