import os
import pytest
import sys
from typing import List

from modal import Conda, DebianSlim, Image, Stub
from modal.exception import DeprecationError, InvalidError
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

        result.append(servicer.images[idx])

        if not image.base_images:
            break

        image_id = image.base_images[0].image_id

    return result


def test_image_python_packages(client, servicer):
    stub = Stub()
    stub["image"] = Image.debian_slim().pip_install(["numpy"])
    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)

        assert any("pip install numpy" in cmd for cmd in layers[0].dockerfile_commands)


def test_debian_slim_deprecated(servicer, client):
    with pytest.warns(DeprecationError):
        DebianSlim()


def test_wrong_type(servicer, client):
    image = Image.debian_slim()
    for method in [image.pip_install, image.apt_install, image.run_commands]:
        method(["xyz"])
        with pytest.raises(InvalidError):
            method("xyz")


def test_image_requirements_txt(servicer, client):
    requirements_txt = os.path.join(os.path.dirname(__file__), "supports/test-requirements.txt")

    stub = Stub()
    stub["image"] = Image.debian_slim().pip_install_from_requirements(requirements_txt)
    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)

        assert any("COPY /.requirements.txt /.requirements.txt" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("pip install -r /.requirements.txt" in cmd for cmd in layers[0].dockerfile_commands)
        assert any(b"banana" in f.data for f in layers[0].context_files)


def test_empty_install(servicer, client):
    with pytest.raises(TypeError):
        Image.debian_slim().pip_install()  # Missing positional argument `packages`

    # Install functions with no packages should be ignored.
    stub = Stub(image=Image.debian_slim().pip_install([]).apt_install([]))

    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)
        assert len(layers) == 1


def test_debian_slim_apt_install(servicer, client):
    stub = Stub(
        image=Image.debian_slim().pip_install(["numpy"]).apt_install(["git", "ssh"]).pip_install(["scikit-learn"])
    )

    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)

        assert any("pip install scikit-learn" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("apt-get install -y git ssh" in cmd for cmd in layers[1].dockerfile_commands)
        assert any("pip install numpy" in cmd for cmd in layers[2].dockerfile_commands)


def test_conda_install(servicer, client):
    stub = Stub(
        image=Image.conda().pip_install(["numpy"]).conda_install(["pymc3", "theano"]).pip_install(["scikit-learn"])
    )

    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)

        assert any("pip install scikit-learn" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("conda install pymc3 theano --yes" in cmd for cmd in layers[1].dockerfile_commands)
        assert any("pip install numpy" in cmd for cmd in layers[2].dockerfile_commands)


def test_conda_deprecated(servicer, client):
    with pytest.warns(DeprecationError) as record:
        Conda()

    # Make sure it has the right filename
    assert record[0].filename == __file__


def test_dockerfile_image(servicer, client):
    path = os.path.join(os.path.dirname(__file__), "supports/test-dockerfile")

    stub = Stub(image=Image.from_dockerfile(path))

    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)

        assert any("RUN pip install numpy" in cmd for cmd in layers[1].dockerfile_commands)


def test_conda_update_from_environment(servicer, client):
    path = os.path.join(os.path.dirname(__file__), "supports/test-conda-environment.yml")

    stub = Stub(image=Image.conda().conda_update_from_environment(path))

    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)

        assert any("RUN conda env update" in cmd for cmd in layers[0].dockerfile_commands)
        assert any(b"foo=1.0" in f.data for f in layers[0].context_files)
        assert any(b"bar=2.1" in f.data for f in layers[0].context_files)


def test_dockerhub_install(servicer, client):
    stub = Stub(image=Image.from_dockerhub("gisops/valhalla:latest", setup_commands=["apt-get update"]))

    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)

        assert any("FROM gisops/valhalla:latest" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("apt-get update" in cmd for cmd in layers[0].dockerfile_commands)
