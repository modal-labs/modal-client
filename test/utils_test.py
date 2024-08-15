# Copyright Modal Labs 2022
import asyncio
import hashlib
import io
import pytest

from modal._utils.blob_utils import BytesIOSegmentPayload
from modal._utils.name_utils import (
    check_object_name,
    is_valid_environment_name,
    is_valid_object_name,
    is_valid_subdomain_label,
    is_valid_tag,
)
from modal._utils.package_utils import parse_major_minor_version
from modal.exception import InvalidError


def test_subdomain_label():
    assert is_valid_subdomain_label("banana")
    assert is_valid_subdomain_label("foo-123-456")
    assert not is_valid_subdomain_label("BaNaNa")
    assert not is_valid_subdomain_label(" ")
    assert not is_valid_subdomain_label("ban/ana")


def test_object_name():
    assert is_valid_object_name("baNaNa")
    assert is_valid_object_name("foo-123_456")
    assert is_valid_object_name("a" * 64)
    assert not is_valid_object_name("hello world")
    assert not is_valid_object_name("a" * 65)
    assert not is_valid_object_name("ap-abcdefghABCDEFGH012345")
    with pytest.raises(InvalidError, match="Invalid Volume name: 'foo/bar'"):
        check_object_name("foo/bar", "Volume")


def test_environment_name():
    assert is_valid_object_name("a" * 64)
    assert not is_valid_object_name("a" * 65)
    assert not is_valid_environment_name("--help")
    assert not is_valid_environment_name(":env")
    assert not is_valid_environment_name("env:env")
    assert not is_valid_environment_name("/env")
    assert not is_valid_environment_name("env/env")
    assert not is_valid_environment_name("")


def test_tag():
    assert is_valid_tag("v1.0.0")
    assert is_valid_tag("a38298githash39920bk")
    assert not is_valid_tag("v1 .0.0-alpha")
    assert not is_valid_tag("$$$build")


@pytest.mark.asyncio
async def test_file_segment_payloads():
    data = io.BytesIO(b"abc123")
    data2 = io.BytesIO(data.getbuffer())

    class DummyOutput:  # AbstractStreamWriter
        def __init__(self):
            self.value = b""

        async def write(self, chunk: bytes):
            self.value += chunk

    out1 = DummyOutput()
    out2 = DummyOutput()
    p1 = BytesIOSegmentPayload(data, 0, 3)
    p2 = BytesIOSegmentPayload(data2, 3, 3)

    # "out of order" writes
    await p2.write(out2)  # type: ignore
    await p1.write(out1)  # type: ignore
    assert out1.value == b"abc"
    assert out2.value == b"123"
    assert p1.md5_checksum().digest() == hashlib.md5(b"abc").digest()
    assert p2.md5_checksum().digest() == hashlib.md5(b"123").digest()

    data = io.BytesIO(b"abc123")

    # test reset_on_error
    all_data = BytesIOSegmentPayload(data, 0, 6)

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
    data2 = io.BytesIO(data.getbuffer())

    class DummyOutput:  # AbstractStreamWriter
        def __init__(self):
            self.value = b""

        async def write(self, chunk: bytes):
            self.value += chunk

    out1 = DummyOutput()
    out2 = DummyOutput()
    p1 = BytesIOSegmentPayload(data, 0, len(data.getvalue()) // 2, chunk_size=100 * 1024)  # 100 KiB chunks
    p2 = BytesIOSegmentPayload(data2, len(data.getvalue()) // 2, len(data.getvalue()) // 2, chunk_size=100 * 1024)
    await asyncio.gather(p2.write(out2), p1.write(out1))  # type: ignore
    assert out1.value + out2.value == data.getvalue()


def test_parse_major_minor_version():
    assert parse_major_minor_version("3.8") == (3, 8)
    assert parse_major_minor_version("3.9.1") == (3, 9)
    assert parse_major_minor_version("3.10.1rc0") == (3, 10)
    with pytest.raises(ValueError, match="at least an 'X.Y' format"):
        parse_major_minor_version("123")
    with pytest.raises(ValueError, match="at least an 'X.Y' format with integral"):
        parse_major_minor_version("x.y")
