import base64
import hashlib
from contextlib import asynccontextmanager

import aiohttp

from modal_proto import api_pb2

# Max size for function inputs and outputs.
MAX_OBJECT_SIZE_BYTES = 64 * 1024  # 64 kb
# Turned off in tests because of an open issue in moto: https://github.com/spulec/moto/issues/816
CHECK_MD5 = True
HASH_CHUNK_SIZE = 4096


def base64_md5(md5) -> str:
    return base64.b64encode(md5.digest()).decode("utf-8")


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


async def _blob_upload(content_md5, aiohttp_payload, stub):
    req = api_pb2.BlobCreateRequest(content_md5=content_md5)
    resp = await stub.BlobCreate(req)

    blob_id = resp.blob_id
    target = resp.upload_url

    async with aiohttp.ClientSession() as session:
        headers = {"content-type": "application/octet-stream"}

        if CHECK_MD5:
            headers["Content-MD5"] = content_md5

        async with session.put(target, data=aiohttp_payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Put to {target} failed with status {resp.status}: {text}")

    return blob_id


@asynccontextmanager
async def _blob_download_file(download_url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(download_url) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Get from {download_url} failed with status {resp.status}: {text}")

            yield resp


async def blob_download(blob_id, stub):
    # convenience function reading all of the downloaded file into memory
    req = api_pb2.BlobGetRequest(blob_id=blob_id)
    resp = await stub.BlobGet(req)

    async with _blob_download_file(resp.download_url) as download_response:
        return await download_response.read()
