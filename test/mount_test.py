import os

import pytest

from modal.mount import _get_files


@pytest.mark.asyncio
async def test_get_files():
    files = {}
    async for tup in _get_files(os.path.dirname(__file__), lambda fn: fn.endswith(".py"), True):
        filename, rel_filename, sha256_hex = tup
        files[filename] = sha256_hex

    assert __file__ in files
