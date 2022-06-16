import pytest

from modal._blob_utils import blob_download, blob_upload
from modal_utils.async_utils import synchronize_apis

_, aio_blob_upload = synchronize_apis(blob_upload)
_, aio_blob_download = synchronize_apis(blob_download)


@pytest.mark.asyncio
async def test_blob_put_get(servicer, blob_server, aio_client):
    # Upload
    blob_id = await aio_blob_upload("Hello, world", aio_client.stub)

    # Download
    data = await aio_blob_download(blob_id, aio_client.stub)
    assert data == b"Hello, world"
