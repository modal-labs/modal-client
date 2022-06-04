import time
from pathlib import Path
from typing import Optional, Union

import aiostream

from modal_proto import api_pb2
from modal_utils.async_utils import retry, synchronize_apis

from ._blob_utils import FileUploadSpec, blob_upload_file, get_file_upload_specs
from .config import logger
from .object import Object


class _SharedVolume(Object, type_prefix="sv"):
    def __init__(
        self,
        *,
        local_init_dir: Optional[Union[str, Path]] = None,
    ):
        self._local_init_dir = local_init_dir
        super().__init__()

    def get_creating_message(self):
        return f"Creating shared volume..."

    def get_created_message(self):
        return f"Created shared volume."

    async def load(self, client, app_id, existing_shared_volume_id):
        if existing_shared_volume_id:
            # Volume already exists; do nothing.
            return existing_shared_volume_id

        t0 = time.time()
        n_concurrent_uploads = 16

        n_files = 0
        total_bytes = 0

        async def _put_file(mount_file: FileUploadSpec):
            nonlocal n_files, total_bytes

            remote_filename = mount_file.rel_filename

            n_files += 1
            total_bytes += mount_file.size

            if mount_file.use_blob:
                logger.debug(f"Creating blob file for {mount_file.filename} ({mount_file.size} bytes)")
                blob_id = await blob_upload_file(mount_file.filename, client.stub)
                logger.debug(f"Uploading blob file {mount_file.filename} as {remote_filename}")
                request = api_pb2.SharedVolumeUploadFileRequest(
                    filename=remote_filename,
                    data_blob_id=blob_id,
                    sha256_hex=mount_file.sha256_hex,
                    size=mount_file.size,
                    shared_volume_id=shared_volume_id,
                )
            else:
                logger.debug(f"Uploading file {mount_file.filename} to {remote_filename} ({mount_file.size} bytes)")
                request = api_pb2.SharedVolumeUploadFileRequest(
                    filename=remote_filename,
                    data=mount_file.content,
                    sha256_hex=mount_file.sha256_hex,
                    size=mount_file.size,
                    shared_volume_id=shared_volume_id,
                )
            await retry(client.stub.SharedVolumeUploadFile, base_delay=1)(request)

        req = api_pb2.SharedVolumeCreateRequest(app_id=app_id)
        resp = await retry(client.stub.SharedVolumeCreate, base_delay=1)(req)
        shared_volume_id = resp.shared_volume_id

        logger.debug(f"Uploading shared volume {shared_volume_id} using {n_concurrent_uploads} uploads")

        # Create async generator
        files_stream = aiostream.stream.iterate(get_file_upload_specs(self._local_init_dir))

        # Upload files
        uploads_stream = aiostream.stream.map(files_stream, _put_file, task_limit=n_concurrent_uploads)
        await uploads_stream

        logger.debug(f"Uploaded {n_files} files and {total_bytes} bytes in {time.time() - t0}s")

        # Set the mount to done
        req_done = api_pb2.SharedVolumeDoneRequest(shared_volume_id=shared_volume_id)
        await retry(client.stub.SharedVolumeDone, base_delay=1)(req_done)

        return shared_volume_id


SharedVolume, AioSharedVolume = synchronize_apis(_SharedVolume)
