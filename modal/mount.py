import asyncio
import concurrent.futures
import hashlib
import os
import time

import aiostream

from ._async_utils import retry
from ._grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from ._package_utils import get_package_deps_mount_info
from .config import config, logger
from .object import Object
from .proto import api_pb2


def get_sha256_hex_from_content(content):
    m = hashlib.sha256()
    m.update(content)
    return m.hexdigest()


def get_sha256_hex_from_filename(filename, rel_filename):
    # Somewhat CPU intensive, so we run it in a thread/process
    content = open(filename, "rb").read()
    return filename, rel_filename, get_sha256_hex_from_content(content)


async def get_files(local_dir, condition, recursive):
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


class Mount(Object):
    def __init__(self, local_dir, remote_dir, condition, session=None, recursive=True):
        super().__init__(session=session)
        self.local_dir = local_dir
        self.remote_dir = remote_dir
        self.condition = condition
        self.recursive = recursive

    async def _put_file(self, client, mount_id, filename, rel_filename, sha256_hex):
        remote_filename = os.path.join(self.remote_dir, rel_filename)  # won't work on windows
        request = api_pb2.MountRegisterFileRequest(filename=remote_filename, sha256_hex=sha256_hex, mount_id=mount_id)
        response = await client.stub.MountRegisterFile(request)
        self.n_files += 1
        if not response.exists:
            # TODO: this will be moved to S3 soon
            data = open(filename, "rb").read()
            self.n_missing_files += 1
            self.total_bytes += len(data)
            logger.debug(f"Uploading file {filename} to {remote_filename} ({len(data)} bytes)")
            request = api_pb2.MountUploadFileRequest(
                data=data, sha256_hex=sha256_hex, size=len(data), mount_id=mount_id
            )
            response = await client.stub.MountUploadFile(request)

    async def _create_impl(self, session):
        req = api_pb2.MountCreateRequest(session_id=session.session_id)
        resp = await session.client.stub.MountCreate(req)
        mount_id = resp.mount_id

        # Run a threadpool to compute hash values, and use n coroutines to put files
        self.n_files = 0
        self.n_missing_files = 0
        self.total_bytes = 0
        t0 = time.time()
        n_concurrent_uploads = 16

        logger.debug(f"Uploading mount {mount_id} using {n_concurrent_uploads} uploads")

        # Create async generator
        files = get_files(self.local_dir, self.condition, self.recursive)
        files_stream = aiostream.stream.iterate(files)

        async def put_file_tupled(tup):
            filename, rel_filename, sha256_hex = tup
            await self._put_file(session.client, mount_id, filename, rel_filename, sha256_hex)

        # Upload files
        uploads_stream = aiostream.stream.map(files_stream, put_file_tupled, task_limit=n_concurrent_uploads)
        await uploads_stream

        logger.debug(
            f"Uploaded {self.n_missing_files}/{self.n_files} files and {self.total_bytes} bytes in {time.time() - t0}s"
        )

        # Set the mount to done
        req = api_pb2.MountDoneRequest(mount_id=mount_id)
        await session.client.stub.MountDone(req)

        return mount_id


def create_package_mounts(package_name):
    mount_infos = get_package_deps_mount_info(package_name)
    return [Mount(path, f"/pkg/{name}", condition) for (name, path, condition) in mount_infos]
