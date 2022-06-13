import base64
import dataclasses
import hashlib
import os
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlparse

import aiohttp

from modal_proto import api_pb2
from modal_utils.async_utils import retry

# Max size for function inputs and outputs.
MAX_OBJECT_SIZE_BYTES = 64 * 1024  # 64 kb
HASH_CHUNK_SIZE = 4096

#  If a file is LARGE_FILE_LIMIT bytes or larger, it's uploaded to blob store (s3) instead of going through grpc
#  It will also make sure to chunk the hash calculation to avoid reading the entire file into memory
LARGE_FILE_LIMIT = 1024 * 1024  # 1MB


def base64_md5(md5) -> str:
    return base64.b64encode(md5.digest()).decode("utf-8")


def check_md5(url):
    # Turned off in tests because of an open issue in moto: https://github.com/spulec/moto/issues/816
    host = urlparse(url).netloc.split(":")[0]
    if host.endswith(".amazonaws.com"):
        return True
    elif host == "127.0.0.1":
        return False
    else:
        raise Exception(f"Unknown S3 host: {host}")


async def blob_upload(payload: bytes, stub):
    content_md5 = base64_md5(hashlib.md5(payload))
    return await _blob_upload(content_md5, payload, stub)


async def blob_upload_file(filename: str, stub):
    md5 = hashlib.md5()
    with open(filename, "rb") as fp:
        # don't read entire file into memory, in case it's a big one
        while 1:
            chunk = fp.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            md5.update(chunk)
        content_md5 = base64_md5(md5)

    with open(filename, "rb") as fp:
        return await _blob_upload(content_md5, fp, stub)


@retry(n_attempts=5, base_delay=0.1, timeout=None)
async def _upload_to_url(upload_url, content_md5, aiohttp_payload):
    async with aiohttp.ClientSession() as session:
        headers = {"content-type": "application/octet-stream"}

        if check_md5(upload_url):
            headers["Content-MD5"] = content_md5

        async with session.put(upload_url, data=aiohttp_payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Put to {upload_url} failed with status {resp.status}: {text}")


async def _blob_upload(content_md5, aiohttp_payload, stub):
    req = api_pb2.BlobCreateRequest(content_md5=content_md5)
    resp = await stub.BlobCreate(req)

    blob_id = resp.blob_id
    target = resp.upload_url

    await _upload_to_url(target, content_md5, aiohttp_payload)

    return blob_id


@asynccontextmanager
async def _blob_download_file(download_url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(download_url) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Get from {download_url} failed with status {resp.status}: {text}")

            yield resp


@retry(n_attempts=5, base_delay=0.1, timeout=None)
async def _download_from_url(download_url):
    async with _blob_download_file(download_url) as download_response:
        return await download_response.read()


async def blob_download(blob_id, stub):
    # convenience function reading all of the downloaded file into memory
    req = api_pb2.BlobGetRequest(blob_id=blob_id)
    resp = await stub.BlobGet(req)

    return await _download_from_url(resp.download_url)


@dataclasses.dataclass
class FileUploadSpec:
    filename: str
    rel_filename: str

    use_blob: bool
    content: Optional[bytes]  # typically None if using blob, required otherwise
    sha256_hex: str
    size: int


def get_file_upload_spec(filename, rel_filename):
    # Somewhat CPU intensive, so we run it in a thread/process
    filesize = os.path.getsize(filename)

    if filesize >= LARGE_FILE_LIMIT:
        sha256 = hashlib.sha256()
        size = 0
        with open(filename, "rb") as fp:
            while 1:
                chunk = fp.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                sha256.update(chunk)

        sha256_hex = sha256.hexdigest()
        return FileUploadSpec(filename, rel_filename, use_blob=True, content=None, sha256_hex=sha256_hex, size=size)
    else:
        with open(filename, "rb") as fp:
            content = fp.read()
        sha256_hex = hashlib.sha256(content).hexdigest()
        return FileUploadSpec(
            filename, rel_filename, use_blob=False, content=content, sha256_hex=sha256_hex, size=len(content)
        )
