import os
import pytest

from modal import Session
from modal.mount import Mount, _get_files


@pytest.mark.asyncio
async def test_get_files():
    files = {}
    async for tup in _get_files(os.path.dirname(__file__), lambda fn: fn.endswith(".py"), True):
        filename, rel_filename, sha256_hex = tup
        files[filename] = sha256_hex

    assert __file__ in files


def test_create_mount(servicer, client):
    session = Session()
    with session.run(client=client):
        local_dir, cur_filename = os.path.split(__file__)
        remote_dir = "/foo"

        def condition(fn):
            return fn.endswith(".py")

        m = Mount.create(local_dir, remote_dir, condition, session=session)
        assert m.object_id == "mo-123"
        assert f"/foo/{cur_filename}" in servicer.files_name2sha
        sha256_hex = servicer.files_name2sha[f"/foo/{cur_filename}"]
        assert sha256_hex in servicer.files_sha2data
        assert servicer.files_sha2data[sha256_hex] == open(__file__, "rb").read()
