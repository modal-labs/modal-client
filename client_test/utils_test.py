# Copyright Modal Labs 2022
import asyncio
import io
import random

import pytest

from modal_utils.app_utils import is_valid_app_name, is_valid_subdomain_label
from modal_utils.blob_utils import BytesIOSegmentPayload
from modal_utils.hash_utils import get_multipart_upload_hash, Part, get_md5_base64, get_sha256_hex, get_sha256_base64


def test_subdomain_label():
    assert is_valid_subdomain_label("banana")
    assert is_valid_subdomain_label("foo-123-456")
    assert not is_valid_subdomain_label("BaNaNa")
    assert not is_valid_subdomain_label(" ")
    assert not is_valid_subdomain_label("ban/ana")


def test_app_name():
    assert is_valid_app_name("baNaNa")
    assert is_valid_app_name("foo-123_456")
    assert is_valid_app_name("a" * 64)
    assert not is_valid_app_name("hello world")
    assert not is_valid_app_name("a" * 65)


@pytest.mark.asyncio
async def test_multihash():
    data_bytes = random.randbytes(25000)
    data_io = io.BytesIO(data_bytes)
    res = await get_multipart_upload_hash(data_io, max_part_size=10000)

    assert res.parts == [
        Part(0, 8192, get_md5_base64(data_bytes[:8192])),
        Part(8192, 8192, get_md5_base64(data_bytes[8192 : 8192 * 2])),
        Part(8192 * 2, 25000 - 8192 * 2, get_md5_base64(data_bytes[8192 * 2 :])),
    ]
    assert res.sha256_hex == get_sha256_hex(data_bytes)
    assert res.sha256_b64 == get_sha256_base64(data_bytes)


@pytest.mark.asyncio
async def test_file_segment_payloads():
    data = io.BytesIO(b"abc123")
    lock = asyncio.Lock()

    class DummyOutput:  # AbstractStreamWriter
        def __init__(self):
            self.value = b""

        async def write(self, chunk: bytes):
            self.value += chunk

    out1 = DummyOutput()
    out2 = DummyOutput()
    p1 = BytesIOSegmentPayload(data, lock, 0, 3)
    p2 = BytesIOSegmentPayload(data, lock, 3, 3)

    # "out of order" writes
    await p2.write(out2)  # noqa
    await p1.write(out1)  # noqa
    assert out1.value == b"abc"
    assert out2.value == b"123"


@pytest.mark.asyncio
async def test_file_segment_payloads_concurrency():
    data = io.BytesIO(random.randbytes(1024 * 1024))  # 1 MiB
    lock = asyncio.Lock()

    class DummyOutput:  # AbstractStreamWriter
        def __init__(self):
            self.value = b""

        async def write(self, chunk: bytes):
            self.value += chunk

    out1 = DummyOutput()
    out2 = DummyOutput()
    p1 = BytesIOSegmentPayload(data, lock, 0, len(data.getvalue()) // 2, chunk_size=100 * 1024)  # 100 KiB chunks
    p2 = BytesIOSegmentPayload(data, lock, len(data.getvalue()) // 2, len(data.getvalue()) // 2, chunk_size=100 * 1024)
    await asyncio.gather(p2.write(out2), p1.write(out1))  # noqa
    assert out1.value + out2.value == data.getvalue()
