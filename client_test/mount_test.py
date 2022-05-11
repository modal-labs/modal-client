import os
import pytest

from modal import App
from modal.aio import AioApp
from modal.mount import AioMount, Mount


@pytest.mark.asyncio
async def test_get_files(servicer, client):
    files = {}
    app = AioApp()
    async with app.run(client=client):
        m = AioMount("/", local_dir=os.path.dirname(__file__), condition=lambda fn: fn.endswith(".py"), recursive=True)
        await m.load(app, None)
        async for tup in m._get_files():
            filename, rel_filename, content, sha256_hex = tup
            files[filename] = sha256_hex

        assert __file__ in files


def test_create_mount(servicer, client):
    app = App()
    with app.run(client=client):
        local_dir, cur_filename = os.path.split(__file__)
        remote_dir = "/foo"

        def condition(fn):
            return fn.endswith(".py")

        m = Mount(local_dir=local_dir, remote_dir=remote_dir, condition=condition)
        m.load(app, None) == "mo-123"
        assert f"/foo/{cur_filename}" in servicer.files_name2sha
        sha256_hex = servicer.files_name2sha[f"/foo/{cur_filename}"]
        assert sha256_hex in servicer.files_sha2data
        assert servicer.files_sha2data[sha256_hex] == open(__file__, "rb").read()
