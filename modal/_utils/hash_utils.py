# Copyright Modal Labs 2022
import base64
import dataclasses
import hashlib
import time
from typing import BinaryIO, Callable, Union

from modal.config import logger

HASH_CHUNK_SIZE = 4096


def _update(hashers: list[Callable[[bytes], None]], data: Union[bytes, BinaryIO]) -> None:
    if isinstance(data, bytes):
        for hasher in hashers:
            hasher(data)
    else:
        assert not isinstance(data, (bytearray, memoryview))  # https://github.com/microsoft/pyright/issues/5697
        pos = data.tell()
        while True:
            chunk = data.read(HASH_CHUNK_SIZE)
            if not isinstance(chunk, bytes):
                raise ValueError(f"Only accepts bytes or byte buffer objects, not {type(chunk)} buffers")
            if not chunk:
                break
            for hasher in hashers:
                hasher(chunk)
        data.seek(pos)


def get_sha256_hex(data: Union[bytes, BinaryIO]) -> str:
    t0 = time.monotonic()
    hasher = hashlib.sha256()
    _update([hasher.update], data)
    logger.debug("get_sha256_hex took %.3fs", time.monotonic() - t0)
    return hasher.hexdigest()


def get_sha256_base64(data: Union[bytes, BinaryIO]) -> str:
    t0 = time.monotonic()
    hasher = hashlib.sha256()
    _update([hasher.update], data)
    logger.debug("get_sha256_base64 took %.3fs", time.monotonic() - t0)
    return base64.b64encode(hasher.digest()).decode("ascii")


def get_md5_base64(data: Union[bytes, BinaryIO]) -> str:
    t0 = time.monotonic()
    hasher = hashlib.md5()
    _update([hasher.update], data)
    logger.debug("get_md5_base64 took %.3fs", time.monotonic() - t0)
    return base64.b64encode(hasher.digest()).decode("utf-8")


@dataclasses.dataclass
class UploadHashes:
    md5_base64: str
    sha256_base64: str


def get_upload_hashes(data: Union[bytes, BinaryIO]) -> UploadHashes:
    t0 = time.monotonic()
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    _update([md5.update, sha256.update], data)
    hashes = UploadHashes(
        md5_base64=base64.b64encode(md5.digest()).decode("ascii"),
        sha256_base64=base64.b64encode(sha256.digest()).decode("ascii"),
    )
    logger.debug("get_upload_hashes took %.3fs", time.monotonic() - t0)
    return hashes
