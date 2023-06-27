# Copyright Modal Labs 2023

import pytest

import modal
from modal.exception import InvalidError
from modal.volume import VolumeHandle

from .supports.skip import skip_windows


def dummy():
    pass


def test_volume_mount(client, servicer):
    stub = modal.Stub()

    dummy_modal = stub.function(
        volumes={"/root/foo": modal.Volume()},
    )(dummy)

    with stub.run(client=client):
        dummy_modal.call()


@skip_windows("TODO: implement client-side path check on Windows.")
def test_volume_bad_paths(client, servicer):
    stub = modal.Stub()

    dummy_modal = stub.function(volumes={"/root/../../foo": modal.Volume()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.call()

    dummy_modal = stub.function(volumes={"/": modal.Volume()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.call()

    dummy_modal = stub.function(volumes={"/tmp/": modal.Volume()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.call()


def test_volume_duplicate_mount(client, servicer):
    stub = modal.Stub()

    volume = modal.Volume()
    dummy_modal = stub.function(volumes={"/foo": volume, "/bar": volume})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.call()


def test_volume_commit(client, servicer):
    stub = modal.Stub()
    stub.vol = modal.Volume()

    with stub.run(client=client) as app:
        handle = app.vol
        assert isinstance(handle, VolumeHandle)
        # Note that in practice this will not work unless run in a task.
        handle.commit()

    assert servicer.volume_commits[handle.object_id] == 1
    # commit should implicitly reload on successful commit
    assert servicer.volume_reloads[handle.object_id] == 1


def test_volume_reload(client, servicer):
    stub = modal.Stub()
    stub.vol = modal.Volume()

    with stub.run(client=client) as app:
        handle = app.vol
        assert isinstance(handle, VolumeHandle)
        # Note that in practice this will not work unless run in a task.
        handle.reload()

    assert servicer.volume_reloads[handle.object_id] == 1
