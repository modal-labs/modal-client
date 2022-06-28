import os
import sys
from typing import List

from modal import Conda, DebianSlim, Stub
from modal.image import _dockerhub_python_version
from modal_proto import api_pb2


def test_python_version():
    assert _dockerhub_python_version("3.9.1") == "3.9.9"
    assert _dockerhub_python_version("3.9") == "3.9.9"
    v = _dockerhub_python_version().split(".")
    assert len(v) == 3
    assert (int(v[0]), int(v[1])) == sys.version_info[:2]


def get_image_layers(image_id: str, servicer) -> List[api_pb2.Image]:
    """Follow pointers to the previous image recursively in the servicer's list of images,
    and return a list of image layers from top to bottom."""

    result = []

    while True:
        idx = int(image_id.split("-")[1])

        if idx not in servicer.images:
            break

        image = servicer.images[idx]
        image_id = image.base_images[0].image_id

        result.append(servicer.images[idx])

    return result


def test_image_python_packages(client, servicer):
    stub = Stub()
    stub["image"] = DebianSlim().pip_install(["numpy"])
    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)

        assert any("pip install numpy" in cmd for cmd in layers[0].dockerfile_commands)


def test_image_requirements_txt(servicer, client):
    requirements_txt = os.path.join(os.path.dirname(__file__), "test-requirements.txt")

    stub = Stub()
    stub["image"] = DebianSlim().pip_install_from_requirements(requirements_txt)
    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)

        assert any("COPY /.requirements.txt /.requirements.txt" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("pip install -r /.requirements.txt" in cmd for cmd in layers[0].dockerfile_commands)
        assert any(b"banana" in f.data for f in layers[0].context_files)


def test_debian_slim_apt_install(servicer, client):
    stub = Stub(image=DebianSlim().pip_install(["numpy"]).apt_install(["git", "ssh"]).pip_install(["scikit-learn"]))

    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)

        assert any("pip install scikit-learn" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("apt-get install -y git ssh" in cmd for cmd in layers[1].dockerfile_commands)
        assert any("pip install numpy" in cmd for cmd in layers[2].dockerfile_commands)


def test_conda_install(servicer, client):
    stub = Stub(image=Conda().pip_install(["numpy"]).conda_install(["pymc3", "theano"]).pip_install(["scikit-learn"]))

    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)
        print(layers, servicer.images)

        assert any("pip install scikit-learn" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("conda install pymc3 theano --yes" in cmd for cmd in layers[1].dockerfile_commands)
        assert any("pip install numpy" in cmd for cmd in layers[2].dockerfile_commands)
