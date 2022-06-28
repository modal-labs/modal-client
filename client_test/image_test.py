import os
import sys

from modal import DebianSlim, Stub
from modal.image import _dockerhub_python_version


def test_python_version():
    assert _dockerhub_python_version("3.9.1") == "3.9.9"
    assert _dockerhub_python_version("3.9") == "3.9.9"
    v = _dockerhub_python_version().split(".")
    assert len(v) == 3
    assert (int(v[0]), int(v[1])) == sys.version_info[:2]


def test_debian_slim_python_packages(client):
    stub = Stub()
    stub["image"] = DebianSlim().pip_install(["numpy"])
    with stub.run(client=client) as running_app:
        assert running_app["image"].object_id == "im-123"


def test_debian_slim_requirements_txt(servicer, client):
    requirements_txt = os.path.join(os.path.dirname(__file__), "test-requirements.txt")

    stub = Stub()
    stub["image"] = DebianSlim().pip_install_from_requirements(requirements_txt)
    with stub.run(client=client) as running_app:
        assert running_app["image"].object_id == "im-123"
        assert any(
            "COPY /.requirements.txt /.requirements.txt" in cmd for cmd in servicer.last_image.dockerfile_commands
        )
        assert any("pip install -r /.requirements.txt" in cmd for cmd in servicer.last_image.dockerfile_commands)
        assert any(b"banana" in f.data for f in servicer.last_image.context_files)


def test_debian_slim_apt_install(servicer, client):
    stub = Stub(image=DebianSlim().pip_install(["numpy"]).apt_install(["git", "ssh"]))

    with stub.run(client=client) as running_app:
        assert running_app["image"].object_id == "im-123"
        assert any("apt-get install -y git ssh" in cmd for cmd in servicer.last_image.dockerfile_commands)
