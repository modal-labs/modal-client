import asyncio
import os

import pytest

from polyester.mount import get_files


@pytest.mark.asyncio
async def test_get_files():
    q = asyncio.Queue()
    n_sentinels = 10
    await get_files(os.path.dirname(__file__), lambda fn: fn.endswith(".py"), True, q, n_sentinels)
    files = {}
    while not q.empty():
        z = await q.get()
        if z is None:
            n_sentinels -= 1
        else:
            filename, rel_filename, sha256_hex = z
        files[filename] = sha256_hex

    assert n_sentinels == 0
    assert __file__ in files
