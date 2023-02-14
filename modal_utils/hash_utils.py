# Copyright Modal Labs 2022
import base64
import dataclasses
import hashlib
from typing import IO, Union

HASH_CHUNK_SIZE = 4096


def _update(hashers, data: Union[bytes, IO[bytes]]):
    if isinstance(data, bytes):
        for hasher in hashers:
            hasher.update(data)
    else:
        pos = data.tell()
        while 1:
            chunk = data.read(HASH_CHUNK_SIZE)
            if not isinstance(chunk, bytes):
                raise ValueError(f"Only accepts bytes or byte buffer objects, not {type(chunk)} buffers")
            if not chunk:
                break
            for hasher in hashers:
                hasher.update(chunk)
        data.seek(pos)


def get_sha256_hex(data: Union[bytes, IO[bytes]]) -> str:
    hasher = hashlib.sha256()
    _update([hasher], data)
    return hasher.hexdigest()


def get_sha256_base64(data: Union[bytes, IO[bytes]]) -> str:
    hasher = hashlib.sha256()
    _update([hasher], data)
    return base64.b64encode(hasher.digest()).decode("ascii")


def get_md5_base64(data: Union[bytes, IO[bytes]]) -> str:
    hasher = hashlib.md5()
    _update([hasher], data)
    return base64.b64encode(hasher.digest()).decode("utf-8")


@dataclasses.dataclass
class UploadHashes:
    md5_base64: str
    sha256_base64: str


def get_upload_hashes(data: Union[bytes, IO[bytes]]) -> UploadHashes:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    _update([md5, sha256], data)
    return UploadHashes(
        md5_base64=base64.b64encode(md5.digest()).decode("ascii"),
        sha256_base64=base64.b64encode(sha256.digest()).decode("ascii"),
    )
