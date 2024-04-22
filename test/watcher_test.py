# Copyright Modal Labs 2023
import pytest
import random
import string
import sys
from pathlib import Path

from watchfiles import Change

import modal
from modal._watcher import _watch_args_from_mounts
from modal.mount import Mount, _Mount


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


def dummy():
    pass


@pytest.fixture()
def clean_sys_modules(monkeypatch):
    # run test assuming no user-defined modules have been loaded
    module_names = set()
    for name, mod in sys.modules.items():
        if getattr(mod, "__file__", None) and not ("/lib/" in mod.__file__ or "/site-packages/" in mod.__file__):
            module_names.add(name)

    for m in module_names:
        monkeypatch.delitem(sys.modules, m)


@pytest.mark.usefixtures("clean_sys_modules")
@pytest.mark.skip("not working in ci for some reason. deactivating for now")  # TODO(elias) fix
def test_watch_mounts_ignore_local():
    app = modal.App()
    app.function(mounts=[Mount.from_name("some-published-mount")])(dummy)

    mounts = app._get_watch_mounts()
    assert len(mounts) == 0
