import asyncio
import concurrent.futures
import hashlib
import os
import time

from .async_utils import retry
from .config import config, logger
from .grpc_utils import BLOCKING_REQUEST_TIMEOUT, GRPC_REQUEST_TIMEOUT
from .object import Object
from .package_utils import get_package_deps_mount_info
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
    def __init__(self, local_dir, remote_dir, condition, recursive=True, session=None):
        super().__init__(tag=None, session=session)
        self.local_dir = local_dir
        self.remote_dir = remote_dir
        self.condition = condition
        self.recursive = recursive

    async def _register_file_requests(self, mount_id, hashes, filenames):
        async for filename, rel_filename, sha256_hex in get_files(self.local_dir, self.condition, self.recursive):
            remote_filename = os.path.join(self.remote_dir, rel_filename)  # won't work on windows
            filenames[remote_filename] = filename
            request = api_pb2.MountRegisterFileRequest(
                filename=remote_filename, sha256_hex=sha256_hex, mount_id=mount_id
            )
            hashes[filename] = sha256_hex
            yield request

    async def _upload_file_requests(self, client, mount_id, hashes, filenames):
        t0 = time.time()
        n_files, n_missing_files, total_bytes = 0, 0, 0
        async for response in client.stub.MountRegisterFile(self._register_file_requests(mount_id, hashes, filenames)):
            n_files += 1
            if not response.exists:
                filename = filenames[response.filename]
                data = open(filename, "rb").read()
                n_missing_files += 1
                total_bytes += len(data)
                logger.debug(f"Uploading file {filename} to {response.filename} ({len(data)} bytes)")
                request = api_pb2.MountUploadFileRequest(
                    data=data, sha256_hex=hashes[filename], size=len(data), mount_id=mount_id
                )
                yield request

        logger.info(f"Uploaded {n_missing_files}/{n_files} files and {total_bytes} bytes in {time.time() - t0}s")

    async def create_or_get(self, session):
        # TODO: I think in theory we could split the get_files iterator and launch multiple concurrent
        # calls to MountRegisterFileRequest and MountUploadFileRequest. This would speed up a lot of the
        # serial operations on the server side (like hitting Redis for every file serially).
        # Another option is to parallelize more on the server side.

        hashes = {}
        filenames = {}

        req = api_pb2.MountCreateRequest(session_id=session.session_id)
        resp = await session.client.stub.MountCreate(req)
        mount_id = resp.mount_id

        logger.debug(f"Uploading mount {mount_id}")
        await session.client.stub.MountUploadFile(self._upload_file_requests(session.client, mount_id, hashes, filenames))

        req = api_pb2.MountDoneRequest(mount_id=mount_id)
        await session.client.stub.MountDone(req)

        return mount_id


def create_package_mounts(package_name):
    mount_infos = get_package_deps_mount_info(package_name)
    return [Mount(path, f"/pkg/{name}", condition) for (name, path, condition) in mount_infos]
