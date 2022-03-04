import base64
import hashlib

import aiohttp

from .proto import api_pb2

# Max size for function inputs and outputs.
# SERIALIZED_SIZE_THRESHOLD = 64 * 1024 # 64 kb
SERIALIZED_SIZE_THRESHOLD = 0


def base64_md5(value) -> str:
    m = hashlib.md5()
    m.update(value)
    return base64.b64encode(m.digest()).decode("utf-8")


async def blob_upload(payload, client):
    content_md5 = base64_md5(payload)

    req = api_pb2.BlobCreateRequest(content_md5=content_md5)
    resp = await client.stub.BlobCreate(req)
    target = resp.presigned_url

    async with aiohttp.ClientSession() as session:
        async with session.put(
            target, data=payload, headers={"content-type": "application/octet-stream", "Content-MD5": content_md5}
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Put to {target} failed with status {resp.status}: {text}")
