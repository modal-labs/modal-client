import os
import pytest
import sys

from modal import App, DebianSlim
from modal.exception import InvalidError
from modal.image import _dockerhub_python_version


def test_python_version():
    assert _dockerhub_python_version("3.9.1") == "3.9.9"
    assert _dockerhub_python_version("3.9") == "3.9.9"
    v = _dockerhub_python_version().split(".")
    assert len(v) == 3
    assert (int(v[0]), int(v[1])) == sys.version_info[:2]


def test_debian_slim_python_packages(client):
    app = App()
    image = DebianSlim(python_packages=["numpy"])
    with app.run(client=client):
        assert app.create_object(image) == "im-123"


def test_debian_slim_with_apps(client):
    app = App()
    with pytest.raises(InvalidError):
        DebianSlim(app, python_packages=["numpy"])


def test_debian_slim_requirements_txt(servicer, client):

    requirements_txt = os.path.join(os.path.dirname(__file__), "test-requirements.txt")

    app = App()
    image = DebianSlim(requirements_txt=requirements_txt)
    with app.run(client=client):
        assert app.create_object(image) == "im-123"
        assert any(
            "COPY /.requirements.txt /.requirements.txt" in cmd for cmd in servicer.last_image.dockerfile_commands
        )
        assert any("pip install -r /.requirements.txt" in cmd for cmd in servicer.last_image.dockerfile_commands)
        assert any(b"banana" in f.data for f in servicer.last_image.context_files)
