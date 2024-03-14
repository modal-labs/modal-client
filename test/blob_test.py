# Copyright Modal Labs 2022
import pytest

from modal._utils.async_utils import synchronize_api
from modal._utils.blob_utils import blob_download as _blob_download, blob_upload as _blob_upload
from modal.exception import ExecutionError

blob_upload = synchronize_api(_blob_upload)
blob_download = synchronize_api(_blob_download)


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


@pytest.mark.asyncio
async def test_blob_multipart(servicer, blob_server, client):
    servicer.blob_multipart_threshold = 2_000_000
    data = b"*" * 10_000_020
    blob_id = await blob_upload.aio(data, client.stub)
    assert await blob_download.aio(blob_id, client.stub) == data
