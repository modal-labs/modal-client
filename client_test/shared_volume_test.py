# Copyright Modal Labs 2022
import platform
import pytest

import modal
from modal.exception import InvalidError


def dummy():
    pass


def test_shared_volume_files(client, test_dir, servicer):
    stub = modal.Stub()

    dummy_modal = stub.function(
        shared_volumes={"/root/foo": modal.SharedVolume()},
    )(dummy)

    with stub.run(client=client):
        dummy_modal.call()


@pytest.mark.skipif(platform.system() == "Windows", reason="TODO: implement client-side path check on Windows.")
def test_shared_volume_bad_paths(client, test_dir, servicer):
    stub = modal.Stub()

    def _f():
        pass

    dummy_modal = stub.function(dummy, shared_volumes={"/root/../../foo": modal.SharedVolume()})

    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.call()

    dummy_modal = stub.function(dummy, shared_volumes={"/": modal.SharedVolume()})

    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.call()

    dummy_modal = stub.function(dummy, shared_volumes={"/tmp/": modal.SharedVolume()})

    with pytest.raises(InvalidError):
        with stub.run(client=client):
            dummy_modal.call()
