# Copyright Modal Labs 2023
import pytest
import random
import string
from pathlib import Path

from watchfiles import Change

import modal
from modal._watcher import _watch_args_from_mounts
from modal.exception import ExecutionError
from modal.mount import _Mount


@pytest.mark.asyncio
async def test__watch_args_from_mounts(monkeypatch, test_dir):
    paths, watch_filter = _watch_args_from_mounts(
        mounts=[
            _Mount._from_local_file("/x/foo.py", remote_path="/foo.py"),
            _Mount._from_local_dir("/one/two/bucklemyshoe", remote_path="/"),
            _Mount._from_local_dir("/x/z", remote_path="/z"),
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


def test_watch_mounts_requires_running_app():
    # Arguably a bit strange to test this, as the exception should never
    # happen unless there is a bug in the client, since _get_watch_mounts
    # is not a public function, and should only ever be called from "safe"
    # contexts...

    # requires running app to make sure the mounts have been loaded
    app = modal.App()
    with pytest.raises(ExecutionError):
        # _get_watch_mounts needs to be called on a hydrated app
        app._get_watch_mounts()


def test_watch_mounts_ignore_non_local(client, servicer):
    app = modal.App()

    # uses the published client mount - should not be included in watch items
    # serialized=True avoids auto-mounting the entrypoint
    @app.function(serialized=True)
    def dummy():
        pass

    with app.run(client=client):
        mounts = app._get_watch_mounts()

    assert len(mounts) == 0


def test_add_local_mount_included_in_serve_watchers(servicer, client, supports_on_path):
    deb_slim = modal.Image.debian_slim()
    img = deb_slim.add_local_python_source("pkg_a")
    app = modal.App()

    @app.function(serialized=True, image=img)
    def f():
        pass

    with app.run(client=client):
        watch_mounts = app._get_watch_mounts()
    assert len(watch_mounts) == 1
