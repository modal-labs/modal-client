# Copyright Modal Labs 2023

import pytest

import modal
from modal.exception import InvalidError
from modal.runner import deploy_stub

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

    with stub.run(client=client):
        # Note that in practice this will not work unless run in a task.
        stub.vol.commit()

        # Make sure we can commit through the provider too
        stub.vol.commit()

        assert servicer.volume_commits[stub.vol.object_id] == 2
        # commit should implicitly reload on successful commit
        assert servicer.volume_reloads[stub.vol.object_id] == 2


def test_volume_reload(client, servicer):
    stub = modal.Stub()
    stub.vol = modal.Volume.new()

    with stub.run(client=client):
        # Note that in practice this will not work unless run in a task.
        stub.vol.reload()

        assert servicer.volume_reloads[stub.vol.object_id] == 1


def test_redeploy(servicer, client):
    stub = modal.Stub()
    stub.v1 = modal.Volume.new()
    stub.v2 = modal.Volume.new()
    stub.v3 = modal.Volume.new()

    # Deploy app once
    deploy_stub(stub, "my-app", client=client)
    app1_ids = [stub.v1.object_id, stub.v2.object_id, stub.v3.object_id]

    # Deploy app again
    deploy_stub(stub, "my-app", client=client)
    app2_ids = [stub.v1.object_id, stub.v2.object_id, stub.v3.object_id]

    # Make sure ids are stable
    assert app1_ids == app2_ids

    # Make sure ids are unique
    assert len(set(app1_ids)) == 3
    assert len(set(app2_ids)) == 3

    # Deploy to a different app
    deploy_stub(stub, "my-other-app", client=client)
    app3_ids = [stub.v1.object_id, stub.v2.object_id, stub.v3.object_id]

    # Should be unique and different
    assert len(set(app3_ids)) == 3
    assert set(app1_ids) & set(app3_ids) == set()
