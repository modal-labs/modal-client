# Copyright Modal Labs 2022
import asyncio
import dataclasses
import hashlib
import io
import math
import os
import platform
import time
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Optional, Union
from urllib.parse import urlparse

from aiohttp import BytesIOPayload
from aiohttp.abc import AbstractStreamWriter

from modal_proto import api_pb2, modal_api_grpc
from modal_proto.modal_api_grpc import ModalClientModal

from ..exception import ExecutionError
from .async_utils import TaskContext, retry
from .grpc_utils import retry_transient_errors
from .hash_utils import get_sha256_hex
from .http_utils import ClientSessionRegistry
from .logger import logger

# Max size for function inputs and outputs.
MAX_OBJECT_SIZE_BYTES = 2 * 1024 * 1024  # 2 MiB

#  If a file is LARGE_FILE_LIMIT bytes or larger, it's uploaded to blob store (s3) instead of going through grpc
#  It will also make sure to chunk the hash calculation to avoid reading the entire file into memory
LARGE_FILE_LIMIT = 4 * 1024 * 1024  # 4 MiB

# Max parallelism during map calls
BLOB_MAX_PARALLELISM = 10

# read ~16MiB chunks by default
DEFAULT_SEGMENT_CHUNK_SIZE = 2**24


class BytesIOSegmentPayload(BytesIOPayload):
    """Modified bytes payload for concurrent sends of chunks from the same file.

    Adds:
    * read limit using remaining_bytes, in order to split files across streams
    * larger read chunk (to prevent excessive read contention between parts)
    * calculates an md5 for the segment

    Feels like this should be in some standard lib...
    """

    def __init__(
        self,
        bytes_io: BinaryIO,  # should *not* be shared as IO position modification is not locked
        segment_start: int,
        segment_length: int,
        chunk_size: int = DEFAULT_SEGMENT_CHUNK_SIZE,
        progress_report_cb: Optional[Callable] = None,
    ):
        # not thread safe constructor!
        super().__init__(bytes_io)
        self.initial_seek_pos = bytes_io.tell()
        self.segment_start = segment_start
        self.segment_length = segment_length
        # seek to start of file segment we are interested in, in order to make .size() evaluate correctly
        self._value.seek(self.initial_seek_pos + segment_start)
        assert self.segment_length <= super().size
        self.chunk_size = chunk_size
        self.progress_report_cb = progress_report_cb or (lambda *_, **__: None)
        self.reset_state()

    def reset_state(self):
        self._md5_checksum = hashlib.md5()
        self.num_bytes_read = 0
        self._value.seek(self.initial_seek_pos)

    @contextmanager
    def reset_on_error(self):
        try:
            yield
        except Exception as exc:
            try:
                self.progress_report_cb(reset=True)
            except Exception as cb_exc:
                raise cb_exc from exc
            raise exc
        finally:
            self.reset_state()

    @property
    def size(self) -> int:
        return self.segment_length

    def md5_checksum(self):
        return self._md5_checksum

    async def write(self, writer: AbstractStreamWriter):
        loop = asyncio.get_event_loop()

        async def safe_read():
            read_start = self.initial_seek_pos + self.segment_start + self.num_bytes_read
            self._value.seek(read_start)
            num_bytes = min(self.chunk_size, self.remaining_bytes())
            chunk = await loop.run_in_executor(None, self._value.read, num_bytes)

            await loop.run_in_executor(None, self._md5_checksum.update, chunk)
            self.num_bytes_read += len(chunk)
            return chunk

        chunk = await safe_read()
        while chunk and self.remaining_bytes() > 0:
            await writer.write(chunk)
            self.progress_report_cb(advance=len(chunk))
            chunk = await safe_read()
        if chunk:
            await writer.write(chunk)
            self.progress_report_cb(advance=len(chunk))

    def remaining_bytes(self):
        return self.segment_length - self.num_bytes_read


@retry(n_attempts=5, base_delay=0.5, timeout=None)
async def _upload_with_request_details(
    request_details: api_pb2.BlobUploadRequestDetails,
    payload: BytesIOSegmentPayload,
):
    with payload.reset_on_error():  # ensure retries read the same data
        method = request_details.method
        uri = request_details.uri

        headers = {}
        for header in request_details.headers:
            headers[header.name] = header.value

        async with ClientSessionRegistry.get_session().request(
            method,
            uri,
            data=payload,
            headers=headers,
        ) as resp:
            # S3 signal to slow down request rate.
            if resp.status == 503:
                logger.warning("Received SlowDown signal from S3, sleeping for 1 second before retrying.")
                await asyncio.sleep(1)

            if resp.status not in [200, 204]:
                try:
                    text = await resp.text()
                except Exception:
                    text = "<no body>"
                raise ExecutionError(f"{method} to url {uri} failed with status {resp.status}: {text}")


async def _stage_and_upload(
    stub: modal_api_grpc.ModalClientModal,
    session_token: bytes,
    part: int,
    payload: BytesIOSegmentPayload,
):
    req = api_pb2.BlobStagePartRequest(session_token=session_token, part=part)
    resp = await retry_transient_errors(stub.BlobStagePart, req)
    request_details = resp.upload_request
    return await _upload_with_request_details(request_details, payload)


async def _perform_multipart_upload(
    data_file: Union[BinaryIO, io.BytesIO, io.FileIO],
    *,
    stub: modal_api_grpc.ModalClientModal,
    session_token: bytes,
    blob_size: int,
    max_part_size: int,
    upload_chunk_size: int = DEFAULT_SEGMENT_CHUNK_SIZE,
    progress_report_cb: Optional[Callable] = None,
):
    def ceildiv(a, b):
        return -(a // -b)

    upload_coros = []
    file_offset = 0
    num_parts = ceildiv(blob_size, max_part_size)
    num_bytes_left = blob_size

    # Give each part its own IO reader object to avoid needing to
    # lock access to the reader's position pointer.
    data_file_readers: list[BinaryIO]
    if isinstance(data_file, io.BytesIO):
        view = data_file.getbuffer()  # does not copy data
        data_file_readers = [io.BytesIO(view) for _ in range(num_parts)]
    else:
        filename = data_file.name
        data_file_readers = [open(filename, "rb") for _ in range(num_parts)]

    for part, data_file_rdr in enumerate(data_file_readers):
        part_length_bytes = min(num_bytes_left, max_part_size)
        part_payload = BytesIOSegmentPayload(
            data_file_rdr,
            segment_start=file_offset,
            segment_length=part_length_bytes,
            chunk_size=upload_chunk_size,
            progress_report_cb=progress_report_cb,
        )
        upload_coros.append(_stage_and_upload(stub, session_token, part, part_payload))
        num_bytes_left -= part_length_bytes
        file_offset += part_length_bytes

    await TaskContext.gather(*upload_coros)


def _get_blob_size(data: BinaryIO) -> int:
    # *Remaining* length of file from current seek position
    pos = data.tell()
    data.seek(0, os.SEEK_END)
    content_length = data.tell()
    data.seek(pos)
    return content_length - pos


async def _blob_upload(
    sha256_hex: str,
    data: Union[bytes, BinaryIO],
    stub: modal_api_grpc.ModalClientModal,
    progress_report_cb: Optional[Callable] = None
) -> str:
    if isinstance(data, bytes):
        data = io.BytesIO(data)

    blob_size = _get_blob_size(data)

    create_req = api_pb2.BlobCreateUploadRequest(
        blob_hash=sha256_hex,
        blob_size=blob_size,
    )
    create_resp = await retry_transient_errors(stub.BlobCreateUpload, create_req)

    session_token = create_resp.session_token

    which_oneof = create_resp.WhichOneof("upload_status")
    if which_oneof == "already_exists":
        return sha256_hex
    elif which_oneof == "multi_part_upload":
        await _perform_multipart_upload(
            data,
            stub=stub,
            session_token=session_token,
            blob_size=blob_size,
            max_part_size=create_resp.multi_part_upload.part_size,
            upload_chunk_size=DEFAULT_SEGMENT_CHUNK_SIZE,
            progress_report_cb=progress_report_cb,
        )
    elif which_oneof == "single_part_upload":
        request_details = create_resp.single_part_upload.upload_request
        payload = BytesIOSegmentPayload(
            data, segment_start=0, segment_length=blob_size, progress_report_cb=progress_report_cb
        )
        await _upload_with_request_details(
            request_details,
            payload,
        )
    else:
        raise NotImplementedError(f"unsupported upload mode from CreateBlobUploadResponse: {which_oneof}")

    commit_req = api_pb2.BlobCommitUploadRequest(session_token=session_token)
    commit_resp = await retry_transient_errors(stub.BlobCommitUpload, commit_req)

    if progress_report_cb:
        progress_report_cb(complete=True)

    return commit_resp.blob_hash


async def blob_upload(payload: bytes, stub: modal_api_grpc.ModalClientModal) -> str:
    size_mib = len(payload) / 1024 / 1024
    logger.debug(f"Uploading large blob of size {size_mib:.2f} MiB")
    t0 = time.time()
    if isinstance(payload, str):
        logger.warning("Blob uploading string, not bytes - auto-encoding as utf8")
        payload = payload.encode("utf8")
    sha256_hex = get_sha256_hex(payload)
    blob_id = await _blob_upload(sha256_hex, payload, stub)
    dur_s = max(time.time() - t0, 0.001)  # avoid division by zero
    throughput_mib_s = size_mib / dur_s
    logger.debug(f"Uploaded large blob of size {size_mib:.2f} MiB ({throughput_mib_s:.2f} MiB/s)." f" {blob_id}")
    return blob_id


async def blob_upload_file(
    file_obj: BinaryIO, stub: modal_api_grpc.ModalClientModal, progress_report_cb: Optional[Callable] = None
) -> str:
    sha256_hex = get_sha256_hex(file_obj)
    return await _blob_upload(sha256_hex, file_obj, stub, progress_report_cb)


@retry(n_attempts=5, base_delay=0.1, timeout=None)
async def _download_from_url(download_url: str) -> bytes:
    async with ClientSessionRegistry.get_session().get(download_url) as s3_resp:
        # S3 signal to slow down request rate.
        if s3_resp.status == 503:
            logger.warning("Received SlowDown signal from S3, sleeping for 1 second before retrying.")
            await asyncio.sleep(1)

        if s3_resp.status != 200:
            text = await s3_resp.text()
            raise ExecutionError(f"Get from url failed with status {s3_resp.status}: {text}")
        return await s3_resp.read()


async def blob_download(blob_id: str, stub: ModalClientModal) -> bytes:
    """Convenience function for reading all of the downloaded file into memory."""
    logger.debug(f"Downloading large blob {blob_id}")
    t0 = time.time()
    req = api_pb2.BlobGetRequest(blob_id=blob_id)
    resp = await retry_transient_errors(stub.BlobGet, req)
    data = await _download_from_url(resp.download_url)
    size_mib = len(data) / 1024 / 1024
    dur_s = max(time.time() - t0, 0.001)  # avoid division by zero
    throughput_mib_s = size_mib / dur_s
    logger.debug(f"Downloaded large blob {blob_id} of size {size_mib:.2f} MiB ({throughput_mib_s:.2f} MiB/s)")
    return data


async def blob_iter(blob_id: str, stub: ModalClientModal) -> AsyncIterator[bytes]:
    req = api_pb2.BlobGetRequest(blob_id=blob_id)
    resp = await retry_transient_errors(stub.BlobGet, req)
    download_url = resp.download_url
    async with ClientSessionRegistry.get_session().get(download_url) as s3_resp:
        # S3 signal to slow down request rate.
        if s3_resp.status == 503:
            logger.warning("Received SlowDown signal from S3, sleeping for 1 second before retrying.")
            await asyncio.sleep(1)

        if s3_resp.status != 200:
            text = await s3_resp.text()
            raise ExecutionError(f"Get from url failed with status {s3_resp.status}: {text}")

        async for chunk in s3_resp.content.iter_any():
            yield chunk


@dataclasses.dataclass
class FileUploadSpec:
    source: Callable[[], Union[AbstractContextManager, BinaryIO]]
    source_description: Any
    mount_filename: str

    use_blob: bool
    content: Optional[bytes]  # typically None if using blob, required otherwise
    sha256_hex: str
    mode: int  # file permission bits (last 12 bits of st_mode)
    size: int


def _get_file_upload_spec(
    source: Callable[[], Union[AbstractContextManager, BinaryIO]],
    source_description: Any,
    mount_filename: PurePosixPath,
    mode: int,
) -> FileUploadSpec:
    with source() as fp:
        # Current position is ignored - we always upload from position 0
        fp.seek(0, os.SEEK_END)
        size = fp.tell()
        fp.seek(0)

        if size >= LARGE_FILE_LIMIT:
            use_blob = True
            content = None
            sha256_hex = get_sha256_hex(fp)
        else:
            use_blob = False
            content = fp.read()
            sha256_hex = get_sha256_hex(content)

    return FileUploadSpec(
        source=source,
        source_description=source_description,
        mount_filename=mount_filename.as_posix(),
        use_blob=use_blob,
        content=content,
        sha256_hex=sha256_hex,
        mode=mode & 0o7777,
        size=size,
    )


def get_file_upload_spec_from_path(
    filename: Path, mount_filename: PurePosixPath, mode: Optional[int] = None
) -> FileUploadSpec:
    # Python appears to give files 0o666 bits on Windows (equal for user, group, and global),
    # so we mask those out to 0o755 for compatibility with POSIX-based permissions.
    mode = mode or os.stat(filename).st_mode & (0o7777 if platform.system() != "Windows" else 0o7755)
    return _get_file_upload_spec(
        lambda: open(filename, "rb"),
        filename,
        mount_filename,
        mode,
    )


def get_file_upload_spec_from_fileobj(fp: BinaryIO, mount_filename: PurePosixPath, mode: int) -> FileUploadSpec:
    @contextmanager
    def source():
        # We ignore position in stream and always upload from position 0
        fp.seek(0)
        yield fp

    return _get_file_upload_spec(
        source,
        str(fp),
        mount_filename,
        mode,
    )


def use_md5(url: str) -> bool:
    """This takes an upload URL in S3 and returns whether we should attach a checksum.

    It's only a workaround for missing functionality in moto.
    https://github.com/spulec/moto/issues/816
    """
    host = urlparse(url).netloc.split(":")[0]
    if host.endswith(".amazonaws.com") or host.endswith(".r2.cloudflarestorage.com"):
        return True
    elif host in ["127.0.0.1", "localhost", "172.21.0.1"]:
        return False
    else:
        raise Exception(f"Unknown S3 host: {host}")
