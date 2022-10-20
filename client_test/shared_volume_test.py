# Copyright Modal Labs 2022
import platform
import pytest

import modal
from modal.exception import InvalidError


def test_shared_volume_files(client, test_dir, servicer):
    stub = modal.Stub()

    @stub.function(
        shared_volumes={"/root/foo": modal.SharedVolume()},
    )
    def f():
        pass

    with stub.run(client=client):
        f()


@pytest.mark.skipif(platform.system() == "Windows", reason="TODO: implement client-side path check on Windows.")
def test_shared_volume_bad_paths(client, test_dir, servicer):
    stub = modal.Stub()

    def _f():
        pass

    f = stub.function(shared_volumes={"/root/../../foo": modal.SharedVolume()})(_f)

    with pytest.raises(InvalidError):
        with stub.run(client=client):
            f()

    f = stub.function(
        shared_volumes={"/": modal.SharedVolume()},
    )(_f)

    with pytest.raises(InvalidError):
        with stub.run(client=client):
            f()

    f = stub.function(shared_volumes={"/tmp/": modal.SharedVolume()})(_f)

    with pytest.raises(InvalidError):
        with stub.run(client=client):
            f()
