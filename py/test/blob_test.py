# Copyright Modal Labs 2022

import asyncio
import pytest
import random

import modal
from modal._utils.async_utils import synchronize_api
from modal._utils.blob_utils import (
    MULTIPART_INFLIGHT_BYTES_MIN,
    _blob_upload_with_fallback,
    _ByteBudget,
    _get_multipart_inflight_budget,
    blob_download as _blob_download,
    blob_upload as _blob_upload,
    blob_upload_file as _blob_upload_file,
)
from modal.exception import ExecutionError
from modal_proto import api_pb2

blob_upload = synchronize_api(_blob_upload)
blob_download = synchronize_api(_blob_download)
blob_upload_file = synchronize_api(_blob_upload_file)


@pytest.mark.asyncio
async def test_blob_put_get(servicer, blob_server, client):
    # Upload
    blob_id = await blob_upload.aio(b"Hello, world", client._stub)

    # Download
    data = await blob_download.aio(blob_id, client._stub)
    assert data == b"Hello, world"


@pytest.mark.asyncio
async def test_blob_upload_with_fallback_results():
    async def upload(item):
        if item == "r2":
            raise RuntimeError("r2 down")

    blob_id, results = await _blob_upload_with_fallback(
        ["r2", "s3"], ["bl-123:r2", "bl-123"], upload, content_length=1000
    )
    assert blob_id == "bl-123"
    assert [(r.blob_id, r.outcome) for r in results] == [
        ("bl-123:r2", api_pb2.BlobUploadResult.OUTCOME_FAILURE),
        ("bl-123", api_pb2.BlobUploadResult.OUTCOME_SUCCESS),
    ]
    assert results[0].throughput_bytes_s == 0
    assert results[1].throughput_bytes_s > 0

    async def ok(item):
        pass

    blob_id, results = await _blob_upload_with_fallback(["r2", "s3"], ["bl-456:r2", "bl-456"], ok, content_length=1000)
    assert blob_id == "bl-456:r2"
    assert [(r.blob_id, r.outcome) for r in results] == [
        ("bl-456:r2", api_pb2.BlobUploadResult.OUTCOME_SUCCESS),
    ]

    async def fail(item):
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        await _blob_upload_with_fallback(["r2", "s3"], ["bl-789:r2", "bl-789"], fail, content_length=1000)


@pytest.mark.asyncio
async def test_blob_put_failure(servicer, blob_server, client, monkeypatch):
    monkeypatch.setattr(modal._utils.async_utils, "RETRY_N_ATTEMPTS_OVERRIDE", 1)
    with pytest.raises(ExecutionError):
        await blob_upload.aio(b"FAILURE", client._stub)


@pytest.mark.asyncio
async def test_blob_get_failure(servicer, blob_server, client, monkeypatch):
    monkeypatch.setattr(modal._utils.async_utils, "RETRY_N_ATTEMPTS_OVERRIDE", 1)
    with pytest.raises(ExecutionError):
        await blob_download.aio("bl-failure", client._stub)


@pytest.mark.asyncio
async def test_blob_large(servicer, blob_server, client):
    data = b"*" * 10_000_000
    blob_id = await blob_upload.aio(data, client._stub)
    assert await blob_download.aio(blob_id, client._stub) == data


@pytest.mark.asyncio
async def test_blob_multipart(servicer, blob_server, client, monkeypatch, tmp_path):
    monkeypatch.setattr("modal._utils.blob_utils.DEFAULT_SEGMENT_CHUNK_SIZE", 128)
    multipart_threshold = 1024
    servicer.blob_multipart_threshold = multipart_threshold
    # - set high # of parts, to test concurrency correctness
    # - make last part significantly shorter than rest, creating uneven upload time.
    data_len = (256 * multipart_threshold) + (multipart_threshold // 2)
    data = random.randbytes(data_len)  # random data will not hide byte re-ordering corruption
    blob_id = await blob_upload.aio(data, client._stub)
    assert await blob_download.aio(blob_id, client._stub) == data

    data_len = (256 * multipart_threshold) + (multipart_threshold // 2)
    data = random.randbytes(data_len)  # random data will not hide byte re-ordering corruption
    data_filepath = tmp_path / "temp.bin"
    data_filepath.write_bytes(data)
    with data_filepath.open("rb") as f:
        blob_id = await blob_upload_file.aio(f, client._stub)
    assert await blob_download.aio(blob_id, client._stub) == data


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
        blob_id = await blob_upload_file.aio(f, client._stub, byte_budget=budget)

    assert await blob_download.aio(blob_id, client._stub) == data

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
    blob_upload(b"adsfadsf", client._stub)
