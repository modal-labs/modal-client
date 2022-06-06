import pytest

import modal
from modal.exception import InvalidError


def test_shared_volume_files(client, test_dir, servicer):
    app = modal.App()

    @app.function(
        shared_volumes={"/root/foo": modal.SharedVolume(local_init_dir=test_dir / "supports/init_volume")},
    )
    def f():
        pass

    with app.run(client=client):
        f()

    assert len(servicer.shared_volume_files) == 4
    assert set(servicer.shared_volume_files) == set(["a.txt", "b.txt", "c/a.txt", "c/d.txt"])


def test_shared_volume_empty(client, test_dir, servicer):
    app = modal.App()

    @app.function(
        shared_volumes={"/root/foo": modal.SharedVolume(local_init_dir=test_dir / "supports/empty_volume")},
    )
    def f():
        pass

    with app.run(client=client):
        f()

    assert len(servicer.shared_volume_files) == 0


def test_shared_volume_bad_path(client, test_dir, servicer):
    app = modal.App()

    @app.function(
        shared_volumes={"/root/../../foo": modal.SharedVolume(local_init_dir=test_dir / "supports/init_volume")},
    )
    def f():
        pass

    with pytest.raises(InvalidError):
        with app.run(client=client):
            f()
