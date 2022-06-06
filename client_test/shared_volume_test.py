import platform
import pytest

import modal
from modal.exception import InvalidError


def test_shared_volume_files(client, test_dir, servicer):
    app = modal.App()

    @app.function(
        shared_volumes={"/root/foo": modal.SharedVolume()},
    )
    def f():
        pass

    with app.run(client=client):
        f()


@pytest.mark.skipif(platform.system() == "Windows", reason="TODO: implement client-side path check on Windows.")
def test_shared_volume_bad_path(client, test_dir, servicer):
    app = modal.App()

    @app.function(
        shared_volumes={"/root/../../foo": modal.SharedVolume()},
    )
    def f():
        pass

    with pytest.raises(InvalidError):
        with app.run(client=client):
            f()
