# Copyright Modal Labs 2022

import pytest
import random

from modal._utils.async_utils import synchronize_api
from modal._utils.blob_utils import (
    blob_download as _blob_download,
    blob_upload as _blob_upload,
    blob_upload_file as _blob_upload_file,
)
from modal.exception import ExecutionError

from .supports.skip import skip_old_py

blob_upload = synchronize_api(_blob_upload)
blob_download = synchronize_api(_blob_download)
blob_upload_file = synchronize_api(_blob_upload_file)


@pytest.mark.asyncio
async def test_blob_put_get(servicer, blob_server, client):
    # Upload
    blob_id = await blob_upload.aio(b"Hello, world", client.stub)

    # Download
    data = await blob_download.aio(blob_id, client.stub)
    assert data == b"Hello, world"


@pytest.mark.asyncio
async def test_blob_put_failure(servicer, blob_server, client):
    with pytest.raises(ExecutionError):
        await blob_upload.aio(b"FAILURE", client.stub)


@pytest.mark.asyncio
async def test_blob_get_failure(servicer, blob_server, client):
    with pytest.raises(ExecutionError):
        await blob_download.aio("bl-failure", client.stub)


@pytest.mark.asyncio
async def test_blob_large(servicer, blob_server, client):
    data = b"*" * 10_000_000
    blob_id = await blob_upload.aio(data, client.stub)
    assert await blob_download.aio(blob_id, client.stub) == data


@skip_old_py("random.randbytes() was introduced in python 3.9", (3, 9))
@pytest.mark.asyncio
async def test_blob_multipart(servicer, blob_server, client, monkeypatch, tmp_path):
    monkeypatch.setattr("modal._utils.blob_utils.DEFAULT_SEGMENT_CHUNK_SIZE", 128)
    multipart_threshold = 1024
    servicer.blob_multipart_threshold = multipart_threshold
    # - set high # of parts, to test concurrency correctness
    # - make last part significantly shorter than rest, creating uneven upload time.
    data_len = (256 * multipart_threshold) + (multipart_threshold // 2)
    data = random.randbytes(data_len)  # random data will not hide byte re-ordering corruption
    blob_id = await blob_upload.aio(data, client.stub)
    assert await blob_download.aio(blob_id, client.stub) == data

    data_len = (256 * multipart_threshold) + (multipart_threshold // 2)
    data = random.randbytes(data_len)  # random data will not hide byte re-ordering corruption
    data_filepath = tmp_path / "temp.bin"
    data_filepath.write_bytes(data)
    blob_id = await blob_upload_file.aio(data_filepath.open("rb"), client.stub)
    assert await blob_download.aio(blob_id, client.stub) == data


def test_sync(blob_server, client):
    # just tests that tests running blocking calls that upload to blob storage don't deadlock
    blob_upload(b"adsfadsf", client.stub)
