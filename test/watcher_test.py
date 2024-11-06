# Copyright Modal Labs 2023
import pytest
import random
import string
from pathlib import Path

from watchfiles import Change

import modal
from modal import method
from modal._watcher import _watch_args_from_mounts
from modal.exception import InvalidError
from modal.mount import _get_client_mount, _Mount


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


def test_watch_mounts_requires_running_app():
    # requires running app to make sure the mounts have been loaded
    app = modal.App()
    with pytest.raises(InvalidError):
        # _get_watch_mounts needs to be called on a hydrated app
        app._get_watch_mounts()


def test_watch_mounts_includes_function_mounts(client, supports_dir, monkeypatch, disable_auto_mount):
    monkeypatch.syspath_prepend(supports_dir)
    app = modal.App()
    pkg_a_mount = modal.Mount.from_local_python_packages("pkg_a")

    @app.function(mounts=[pkg_a_mount], serialized=True)
    def f():
        pass

    with app.run(client=client):
        watch_mounts = app._get_watch_mounts()
    assert watch_mounts == [pkg_a_mount]


def test_watch_mounts_includes_cls_mounts(client, supports_dir, monkeypatch, disable_auto_mount):
    monkeypatch.syspath_prepend(supports_dir)
    app = modal.App()
    pkg_a_mount = modal.Mount.from_local_python_packages("pkg_a")

    @app.cls(mounts=[pkg_a_mount], serialized=True)
    class A:
        @method()
        def foo(self):
            pass

    with app.run(client=client):
        watch_mounts = app._get_watch_mounts()
    assert watch_mounts == [pkg_a_mount]


def test_watch_mounts_includes_image_mounts(client, supports_dir, monkeypatch, disable_auto_mount):
    monkeypatch.syspath_prepend(supports_dir)
    app = modal.App()
    pkg_a_mount = modal.Mount.from_local_python_packages("pkg_a")
    image = modal.Image.debian_slim().copy_mount(pkg_a_mount)

    @app.function(image=image, serialized=True)
    def f():
        pass

    with app.run(client=client):
        watch_mounts = app._get_watch_mounts()
    assert watch_mounts == [pkg_a_mount]


def test_watch_mounts_ignore_non_local(disable_auto_mount, client, servicer):
    app = modal.App()

    # uses the published client mount - should not be included in watch items
    # serialized=True avoids auto-mounting the entrypoint
    @app.function(mounts=[_get_client_mount()], serialized=True)
    def dummy():
        pass

    with app.run(client=client):
        mounts = app._get_watch_mounts()

    assert len(mounts) == 0


def test_add_local_mount_included_in_serve_watchers(servicer, client, supports_on_path, disable_auto_mount):
    deb_slim = modal.Image.debian_slim()
    img = deb_slim.add_local_python_packages("pkg_a")
    app = modal.App()

    @app.function(serialized=True, image=img)
    def f():
        pass

    with app.run(client=client):
        watch_mounts = app._get_watch_mounts()
    assert len(watch_mounts) == 1
