# Copyright Modal Labs 2022

import asyncio
import pytest
import random

import modal
from modal._utils.async_utils import synchronize_api
from modal._utils.blob_utils import (
    MULTIPART_INFLIGHT_BYTES_MIN,
    _ByteBudget,
    _get_multipart_inflight_budget,
    blob_download as _blob_download,
    blob_upload as _blob_upload,
    blob_upload_file as _blob_upload_file,
)
from modal.exception import ExecutionError

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
async def test_blob_put_failure(servicer, blob_server, client, monkeypatch):
    monkeypatch.setattr(modal._utils.async_utils, "RETRY_N_ATTEMPTS_OVERRIDE", 1)
    with pytest.raises(ExecutionError):
        await blob_upload.aio(b"FAILURE", client.stub)


@pytest.mark.asyncio
async def test_blob_get_failure(servicer, blob_server, client, monkeypatch):
    monkeypatch.setattr(modal._utils.async_utils, "RETRY_N_ATTEMPTS_OVERRIDE", 1)
    with pytest.raises(ExecutionError):
        await blob_download.aio("bl-failure", client.stub)


@pytest.mark.asyncio
async def test_blob_large(servicer, blob_server, client):
    data = b"*" * 10_000_000
    blob_id = await blob_upload.aio(data, client.stub)
    assert await blob_download.aio(blob_id, client.stub) == data


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
    with data_filepath.open("rb") as f:
        blob_id = await blob_upload_file.aio(f, client.stub)
    assert await blob_download.aio(blob_id, client.stub) == data


@pytest.mark.asyncio
async def test_blob_multipart_inflight_bytes_bounded(servicer, blob_server, client, monkeypatch, tmp_path):
    """Verify that multipart upload respects the byte budget when one is set."""
    chunk_size = 128
    monkeypatch.setattr("modal._utils.blob_utils.DEFAULT_SEGMENT_CHUNK_SIZE", chunk_size)
    multipart_threshold = 1024
    servicer.blob_multipart_threshold = multipart_threshold
    file_size = 128 * multipart_threshold
    byte_budget = file_size // 2
    budget = _ByteBudget(byte_budget)

    min_available = budget._available
    assert min_available > 0
    original_upload_to_s3 = modal._utils.blob_utils._upload_to_s3_url

    async def tracking_upload(*args, **kwargs):
        nonlocal min_available
        min_available = min(min_available, budget._available)
        await asyncio.sleep(0.01)
        return await original_upload_to_s3(*args, **kwargs)

    monkeypatch.setattr("modal._utils.blob_utils._upload_to_s3_url", tracking_upload)

    # Upload file
    data = random.randbytes(file_size)
    path = tmp_path / "temp.bin"
    path.write_bytes(data)
    with path.open("rb") as f:
        blob_id = await blob_upload_file.aio(f, client.stub, byte_budget=budget)

    assert await blob_download.aio(blob_id, client.stub) == data

    assert min_available == 0, "test did not exercise concurrent uploads"


@pytest.mark.parametrize("exc_type", [KeyError, TypeError, RuntimeError, OSError])
def test_get_multipart_inflight_budget_psutil_exception_fallback(monkeypatch, exc_type):
    """psutil.virtual_memory() can raise exceptions beyond ImportError/AttributeError
    (e.g. KeyError on malformed /proc/meminfo). The budget function must still return a
    valid value via the os.sysconf fallback or the hard-coded minimum."""
    import psutil

    original_virtual_memory = psutil.virtual_memory

    def broken_virtual_memory():
        raise exc_type("simulated failure")

    monkeypatch.setattr(psutil, "virtual_memory", broken_virtual_memory)
    result = _get_multipart_inflight_budget()
    assert result >= MULTIPART_INFLIGHT_BYTES_MIN

    monkeypatch.setattr(psutil, "virtual_memory", original_virtual_memory)


def test_sync(blob_server, client):
    # just tests that tests running blocking calls that upload to blob storage don't deadlock
    blob_upload(b"adsfadsf", client.stub)
