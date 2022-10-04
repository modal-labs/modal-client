import asyncio
import dataclasses
import io
import os
from typing import AsyncIterator, BinaryIO, Optional, Union

from modal.exception import ExecutionError
from modal_proto import api_pb2
from modal_utils.async_utils import retry
from modal_utils.blob_utils import use_md5
from modal_utils.grpc_utils import retry_transient_errors
from modal_utils.hash_utils import get_md5_base64, get_sha256_hex
from modal_utils.http_utils import http_client_with_tls
from modal_utils.logger import logger

# Max size for function inputs and outputs.
MAX_OBJECT_SIZE_BYTES = 64 * 1024  # 64 kb

#  If a file is LARGE_FILE_LIMIT bytes or larger, it's uploaded to blob store (s3) instead of going through grpc
#  It will also make sure to chunk the hash calculation to avoid reading the entire file into memory
LARGE_FILE_LIMIT = 1024 * 1024  # 1MB

# Max parallelism during map calls
BLOB_MAX_PARALLELISM = 10


@retry(n_attempts=5, base_delay=0.5, timeout=None)
async def _upload_to_url(upload_url: str, content_md5: str, payload: Union[bytes, BinaryIO]) -> None:
    async with http_client_with_tls(timeout=None) as session:
        headers = {"content-type": "application/octet-stream"}

        if use_md5(upload_url):
            headers["Content-MD5"] = content_md5

        wrapped_payload: Union[bytes, BinaryIO]
        if isinstance(payload, bytes) and len(payload) > 100_000:
            wrapped_payload = io.BytesIO(payload)
        else:
            wrapped_payload = payload

        async with session.put(upload_url, data=wrapped_payload, headers=headers) as resp:
            # S3 signal to slow down request rate.
            if resp.status == 503:
                logger.warning("Received SlowDown signal from S3, sleeping for 1 second before retrying.")
                await asyncio.sleep(1)

            if resp.status != 200:
                text = await resp.text()
                raise ExecutionError(f"Put to url failed with status {resp.status}: {text}")


async def _blob_upload(content_md5: str, payload: Union[bytes, BinaryIO], stub) -> str:
    req = api_pb2.BlobCreateRequest(content_md5=content_md5)
    resp = await retry_transient_errors(stub.BlobCreate, req)

    blob_id = resp.blob_id
    target = resp.upload_url

    await _upload_to_url(target, content_md5, payload)

    return blob_id


async def blob_upload(payload: bytes, stub) -> str:
    if isinstance(payload, str):
        logger.warning("Blob uploading string, not bytes - auto-encoding as utf8")
        payload = payload.encode("utf8")
    content_md5 = get_md5_base64(payload)
    return await _blob_upload(content_md5, payload, stub)


async def blob_upload_file(file_obj: BinaryIO, stub) -> str:
    content_md5 = get_md5_base64(file_obj)
    file_obj.seek(0)
    return await _blob_upload(content_md5, file_obj, stub)


@retry(n_attempts=5, base_delay=0.1, timeout=None)
async def _download_from_url(download_url) -> bytes:
    async with http_client_with_tls(timeout=None) as session:
        async with session.get(download_url) as resp:
            # S3 signal to slow down request rate.
            if resp.status == 503:
                logger.warning("Received SlowDown signal from S3, sleeping for 1 second before retrying.")
                await asyncio.sleep(1)

            if resp.status != 200:
                text = await resp.text()
                raise ExecutionError(f"Get from url failed with status {resp.status}: {text}")
            return await resp.read()


async def blob_download(blob_id, stub) -> bytes:
    # convenience function reading all of the downloaded file into memory
    req = api_pb2.BlobGetRequest(blob_id=blob_id)
    resp = await retry_transient_errors(stub.BlobGet, req)

    return await _download_from_url(resp.download_url)


async def blob_iter(blob_id, stub) -> AsyncIterator[bytes]:
    req = api_pb2.BlobGetRequest(blob_id=blob_id)
    resp = await retry_transient_errors(stub.BlobGet, req)
    download_url = resp.download_url
    async with http_client_with_tls(timeout=None) as session:
        async with session.get(download_url) as resp:
            # S3 signal to slow down request rate.
            if resp.status == 503:
                logger.warning("Received SlowDown signal from S3, sleeping for 1 second before retrying.")
                await asyncio.sleep(1)

            if resp.status != 200:
                text = await resp.text()
                raise ExecutionError(f"Get from url failed with status {resp.status}: {text}")

            async for chunk in resp.content.iter_any():
                yield chunk


@dataclasses.dataclass
class FileUploadSpec:
    filename: str
    rel_filename: str

    use_blob: bool
    content: Optional[bytes]  # typically None if using blob, required otherwise
    sha256_hex: str
    size: int


def get_file_upload_spec(filename: str, rel_filename: str) -> FileUploadSpec:
    # Somewhat CPU intensive, so we run it in a thread/process
    size = os.path.getsize(filename)
    if size >= LARGE_FILE_LIMIT:
        use_blob = True
        content = None
        with open(filename, "rb") as fp:
            sha256_hex = get_sha256_hex(fp)
    else:
        use_blob = False
        with open(filename, "rb") as fp:
            content = fp.read()
        sha256_hex = get_sha256_hex(content)
    return FileUploadSpec(filename, rel_filename, use_blob=use_blob, content=content, sha256_hex=sha256_hex, size=size)
