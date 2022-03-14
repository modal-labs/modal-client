import base64
import hashlib

import aiohttp
from modal_proto.proto import api_pb2

# Max size for function inputs and outputs.
MAX_OBJECT_SIZE_BYTES = 64 * 1024  # 64 kb
# Turned off in tests because of an open issue in moto: https://github.com/spulec/moto/issues/816
CHECK_MD5 = True


def base64_md5(value) -> str:
    m = hashlib.md5()
    m.update(value)
    return base64.b64encode(m.digest()).decode("utf-8")


async def blob_upload(payload, client):
    content_md5 = base64_md5(payload)

    req = api_pb2.BlobCreateRequest(content_md5=content_md5)
    resp = await client.stub.BlobCreate(req)

    blob_id = resp.blob_id
    target = resp.upload_url

    async with aiohttp.ClientSession() as session:
        headers = {"content-type": "application/octet-stream"}

        if CHECK_MD5:
            headers["Content-MD5"] = content_md5

        async with session.put(target, data=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Put to {target} failed with status {resp.status}: {text}")

    return blob_id


async def blob_download(blob_id, client):
    req = api_pb2.BlobGetRequest(blob_id=blob_id)
    resp = await client.stub.BlobGet(req)
    target = resp.download_url

    async with aiohttp.ClientSession() as session:
        async with session.get(target) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Get from {target} failed with status {resp.status}: {text}")

            data = await resp.read()
            return data
