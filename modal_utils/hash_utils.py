import base64
import hashlib
import io
from typing import Union

HASH_CHUNK_SIZE = 4096


def _update(hasher, data: Union[bytes, io.BufferedIOBase]):
    if isinstance(data, bytes):
        hasher.update(data)
    elif isinstance(data, io.BufferedIOBase):
        while 1:
            chunk = data.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)


def get_sha256_hex(data: Union[bytes, io.BufferedIOBase]) -> str:
    hasher = hashlib.sha256()
    _update(hasher, data)
    return hasher.hexdigest()


def get_md5_base64(data: Union[bytes, io.BufferedIOBase]) -> str:
    hasher = hashlib.md5()
    _update(hasher, data)
    return base64.b64encode(hasher.digest()).decode("utf-8")
