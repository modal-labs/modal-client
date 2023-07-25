# Copyright Modal Labs 2023

import pytest

import modal
from modal.exception import InvalidError
from modal.runner import deploy_stub
from modal.volume import VolumeHandle

from .supports.skip import skip_windows


def dummy():
    pass


def test_volume_mount(client, servicer):
    stub = modal.Stub()

    _ = stub.function(
        volumes={"/root/foo": modal.Volume.new()},
    )(dummy)

    with stub.run(client=client):
        pass


@skip_windows("TODO: implement client-side path check on Windows.")
def test_volume_bad_paths(client, servicer):
    stub = modal.Stub()

    _ = stub.function(volumes={"/root/../../foo": modal.Volume.new()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            pass

    _ = stub.function(volumes={"/": modal.Volume.new()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            pass

    _ = stub.function(volumes={"/tmp/": modal.Volume.new()})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            pass


def test_volume_duplicate_mount(client, servicer):
    stub = modal.Stub()

    volume = modal.Volume.new()
    _ = stub.function(volumes={"/foo": volume, "/bar": volume})(dummy)
    with pytest.raises(InvalidError):
        with stub.run(client=client):
            pass


def test_volume_commit(client, servicer):
    stub = modal.Stub()
    stub.vol = modal.Volume.new()

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
    stub.vol = modal.Volume.new()

    with stub.run(client=client) as app:
        handle = app.vol
        assert isinstance(handle, VolumeHandle)
        # Note that in practice this will not work unless run in a task.
        handle.reload()

    assert servicer.volume_reloads[handle.object_id] == 1


def test_redeploy(servicer, client):
    stub = modal.Stub()
    stub.n1 = modal.Volume.new()
    stub.n2 = modal.Volume.new()
    stub.n3 = modal.Volume.new()

    # Deploy app once
    app1 = deploy_stub(stub, "my-app", client=client)
    app1_ids = [app1.n1.object_id, app1.n2.object_id, app1.n3.object_id]

    # Deploy app again
    app2 = deploy_stub(stub, "my-app", client=client)
    app2_ids = [app2.n1.object_id, app2.n2.object_id, app2.n3.object_id]

    # Make sure ids are stable
    assert app1_ids == app2_ids

    # Make sure ids are unique
    assert len(set(app1_ids)) == 3
    assert len(set(app2_ids)) == 3
