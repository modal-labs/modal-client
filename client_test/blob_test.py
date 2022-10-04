import pytest

from modal._blob_utils import blob_download, blob_upload
from modal.exception import ExecutionError
from modal_utils.async_utils import synchronize_apis

_, aio_blob_upload = synchronize_apis(blob_upload)
_, aio_blob_download = synchronize_apis(blob_download)


@pytest.mark.asyncio
async def test_blob_put_get(servicer, blob_server, aio_client):
    # Upload
    blob_id = await aio_blob_upload(b"Hello, world", aio_client.stub)

    # Download
    data = await aio_blob_download(blob_id, aio_client.stub)
    assert data == b"Hello, world"


@pytest.mark.asyncio
async def test_blob_put_failure(servicer, blob_server, aio_client):
    with pytest.raises(ExecutionError):
        await aio_blob_upload(b"FAILURE", aio_client.stub)


@pytest.mark.asyncio
async def test_blob_get_failure(servicer, blob_server, aio_client):
    with pytest.raises(ExecutionError):
        await aio_blob_download("bl-failure", aio_client.stub)


@pytest.mark.asyncio
async def test_blob_large(servicer, blob_server, aio_client):
    data = b"*" * 10_000_000
    blob_id = await aio_blob_upload(data, aio_client.stub)
    assert await aio_blob_download(blob_id, aio_client.stub) == data
