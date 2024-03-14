# Copyright Modal Labs 2022
import asyncio
import hashlib
import io
import pytest

from modal._utils.app_utils import is_valid_app_name, is_valid_subdomain_label
from modal._utils.blob_utils import BytesIOSegmentPayload


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
    await p2.write(out2)  # type: ignore
    await p1.write(out1)  # type: ignore
    assert out1.value == b"abc"
    assert out2.value == b"123"
    assert p1.md5_checksum().digest() == hashlib.md5(b"abc").digest()
    assert p2.md5_checksum().digest() == hashlib.md5(b"123").digest()

    assert data.read() == b"abc123"
    data.seek(0)

    # test reset_on_error
    all_data = BytesIOSegmentPayload(data, lock, 0, 6)

    class DummyExc(Exception):
        pass

    try:
        with all_data.reset_on_error():
            await all_data.write(DummyOutput())  # type: ignore
    except DummyExc:
        pass

    out = DummyOutput()
    await all_data.write(out)  # type: ignore
    assert out.value == b"abc123"


@pytest.mark.asyncio
async def test_file_segment_payloads_concurrency():
    data = io.BytesIO((b"123" * 1024 * 350)[: 1024 * 1024])  # 1 MiB
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
    await asyncio.gather(p2.write(out2), p1.write(out1))  # type: ignore
    assert out1.value + out2.value == data.getvalue()
