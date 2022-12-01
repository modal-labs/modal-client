# Copyright Modal Labs 2022
import os
import pytest
import sys
from typing import List

from modal import Conda, DebianSlim, Image, Secret, SharedVolume, Stub
from modal.exception import DeprecationError, InvalidError, NotFoundError
from modal.image import _dockerhub_python_version
from modal_proto import api_pb2


def test_python_version():
    assert _dockerhub_python_version("3.9.1") == "3.9.15"
    assert _dockerhub_python_version("3.9") == "3.9.15"
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
    with pytest.raises(DeprecationError):
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
    with pytest.raises(DeprecationError):
        Conda()


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


def run_f():
    print("foo!")


def test_image_run_function(client, servicer):
    stub = Stub()
    volume = SharedVolume().persist("test-vol")
    stub["image"] = (
        Image.debian_slim()
        .pip_install(["pandas"])
        .run_function(run_f, secrets=[Secret({"xyz": "123"})], shared_volumes={"/foo": volume})
    )

    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)
        assert "foo!" in layers[0].build_function_def
        assert "Secret([xyz])" in layers[0].build_function_def
        assert "Ref<SharedVolume()>(test-vol)" in layers[0].build_function_def

    function_id = servicer.image_build_function_ids[2]
    assert function_id
    assert servicer.app_functions[function_id].function_name == "run_f"
    assert len(servicer.app_functions[function_id].secret_ids) == 1


def test_poetry(client, servicer):
    path = os.path.join(os.path.dirname(__file__), "supports/pyproject.toml")

    # No lockfile provided and there's no lockfile found
    with pytest.raises(NotFoundError):
        Image.debian_slim().poetry_install_from_file(path)

    # Explicitly ignore lockfile - this should work
    Image.debian_slim().poetry_install_from_file(path, ignore_lockfile=True)

    # Provide lockfile explicitly - this should also work
    lockfile_path = os.path.join(os.path.dirname(__file__), "supports/special_poetry.lock")
    image = Image.debian_slim().poetry_install_from_file(path, lockfile_path)

    # Build iamge
    stub = Stub()
    stub["image"] = image
    with stub.run(client=client) as running_app:
        layers = get_image_layers(running_app["image"].object_id, servicer)
        context_files = {f.filename for layer in layers for f in layer.context_files}
        assert context_files == {"/.poetry.lock", "/.pyproject.toml", "/modal_requirements.txt"}
