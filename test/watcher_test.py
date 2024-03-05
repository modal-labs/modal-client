# Copyright Modal Labs 2023
import pytest
import random
import string
from pathlib import Path

from watchfiles import Change

from modal._watcher import _watch_args_from_mounts
from modal.mount import _Mount


@pytest.mark.asyncio
async def test__watch_args_from_mounts(monkeypatch, test_dir):
    paths, watch_filter = _watch_args_from_mounts(
        mounts=[
            _Mount.from_local_file("/x/foo.py", remote_path="/foo.py"),
            _Mount.from_local_dir("/one/two/bucklemyshoe", remote_path="/"),
            _Mount.from_local_dir("/x/z", remote_path="/z"),
        ]
    )

    assert paths == {Path("/x").absolute(), Path("/one/two/bucklemyshoe").absolute(), Path("/x/z").absolute()}
    assert watch_filter(Change.modified, "/x/foo.py")
    assert not watch_filter(Change.modified, "/x/notwatched.py")
    assert not watch_filter(Change.modified, "/x/y/foo.py")
    assert watch_filter(Change.modified, "/x/z/bar.py")
    random_filename = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    assert watch_filter(Change.modified, f"/one/two/bucklemyshoe/{random_filename}")
    assert not watch_filter(Change.modified, "/one/two/bucklemyshoe/.DS_Store")
