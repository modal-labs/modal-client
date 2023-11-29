# Copyright Modal Labs 2022
import os
import pytest
import sys
import threading
from tempfile import NamedTemporaryFile
from typing import List
from unittest import mock

from modal import Image, Mount, NetworkFileSystem, Secret, Stub, gpu, method
from modal.exception import InvalidError, NotFoundError
from modal.image import _dockerhub_python_version
from modal_proto import api_pb2


def test_python_version():
    assert _dockerhub_python_version("3.9.1") == "3.9.1"
    assert _dockerhub_python_version("3.9") == "3.9.15"
    v = _dockerhub_python_version().split(".")
    assert len(v) == 3
    assert (int(v[0]), int(v[1])) == sys.version_info[:2]


def get_image_layers(image_id: str, servicer) -> List[api_pb2.Image]:
    """Follow pointers to the previous image recursively in the servicer's list of images,
    and return a list of image layers from top to bottom."""

    result = []

    while True:
        if image_id not in servicer.images:
            break

        image = servicer.images[image_id]
        result.append(servicer.images[image_id])

        if not image.base_images:
            break

        image_id = image.base_images[0].image_id

    return result


def test_image_python_packages(client, servicer):
    stub = Stub()
    stub["image"] = (
        Image.debian_slim()
        .pip_install("sklearn[xyz]")
        .pip_install("numpy", "scipy", extra_index_url="https://xyz", find_links="https://abc?q=123", pre=True)
    )
    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)
        assert any("pip install 'sklearn[xyz]'" in cmd for cmd in layers[1].dockerfile_commands)
        assert any(
            "pip install numpy scipy --find-links 'https://abc?q=123' --extra-index-url https://xyz --pre" in cmd
            for cmd in layers[0].dockerfile_commands
        )


def test_image_kwargs_validation(servicer, client):
    stub = Stub()
    stub["image"] = Image.debian_slim().run_commands(
        "echo hi", secrets=[Secret.from_dict({"xyz": "123"}), Secret.from_name("foo")]
    )
    with pytest.raises(InvalidError):
        stub["image"] = Image.debian_slim().run_commands(
            "echo hi",
            secrets=[
                Secret.from_dict({"xyz": "123"}),
                Secret.from_name("foo"),
                Mount.from_local_dir("/", remote_path="/"),  # type: ignore
            ],  # Mount is not a valid Secret
        )

    stub = Stub()
    stub["image"] = Image.debian_slim().copy_local_dir("/", remote_path="/dummy")
    stub["image"] = Image.debian_slim().copy_mount(Mount.from_name("foo"), remote_path="/dummy")
    with pytest.raises(InvalidError):
        # Secret is not a valid Mount
        stub["image"] = Image.debian_slim().copy_mount(Secret.from_dict({"xyz": "123"}), remote_path="/dummy")  # type: ignore


def test_wrong_type(servicer, client):
    image = Image.debian_slim()
    for m in [image.pip_install, image.apt_install, image.run_commands]:
        m(["xyz"])  # type: ignore
        m("xyz")  # type: ignore
        m("xyz", ["def", "foo"], "ghi")  # type: ignore
        with pytest.raises(InvalidError):
            m(3)  # type: ignore
        with pytest.raises(InvalidError):
            m([3])  # type: ignore
        with pytest.raises(InvalidError):
            m([["double-nested-package"]])  # type: ignore


def test_image_requirements_txt(servicer, client):
    requirements_txt = os.path.join(os.path.dirname(__file__), "supports/test-requirements.txt")

    stub = Stub()
    stub["image"] = Image.debian_slim().pip_install_from_requirements(requirements_txt)
    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)

        assert any("COPY /.requirements.txt /.requirements.txt" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("pip install -r /.requirements.txt" in cmd for cmd in layers[0].dockerfile_commands)
        assert any(b"banana" in f.data for f in layers[0].context_files)


def test_empty_install(servicer, client):
    # Install functions with no packages should be ignored.
    stub = Stub(
        image=Image.debian_slim()
        .pip_install()
        .pip_install([], [], [], [])
        .apt_install([])
        .run_commands()
        .conda_install()
    )

    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)
        assert len(layers) == 1


def test_debian_slim_apt_install(servicer, client):
    stub = Stub(image=Image.debian_slim().pip_install("numpy").apt_install("git", "ssh").pip_install("scikit-learn"))

    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)

        assert any("pip install scikit-learn" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("apt-get install -y git ssh" in cmd for cmd in layers[1].dockerfile_commands)
        assert any("pip install numpy" in cmd for cmd in layers[2].dockerfile_commands)


def test_image_pip_install_pyproject(servicer, client):
    pyproject_toml = os.path.join(os.path.dirname(__file__), "supports/test-pyproject.toml")

    stub = Stub()
    stub["image"] = Image.debian_slim().pip_install_from_pyproject(pyproject_toml)
    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)

        print(layers[0].dockerfile_commands)
        assert any("pip install 'banana >=1.2.0' 'potato >=0.1.0'" in cmd for cmd in layers[0].dockerfile_commands)


def test_image_pip_install_pyproject_with_optionals(servicer, client):
    pyproject_toml = os.path.join(os.path.dirname(__file__), "supports/test-pyproject.toml")

    stub = Stub()
    stub["image"] = Image.debian_slim().pip_install_from_pyproject(
        pyproject_toml, optional_dependencies=["dev", "test"]
    )
    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)

        print(layers[0].dockerfile_commands)
        assert any(
            "pip install 'banana >=1.2.0' 'linting-tool >=0.0.0' 'potato >=0.1.0' 'pytest >=1.2.0'" in cmd
            for cmd in layers[0].dockerfile_commands
        )
        assert not (any("'mkdocs >=1.4.2'" in cmd for cmd in layers[0].dockerfile_commands))


def test_image_pip_install_private_repos(servicer, client):
    stub = Stub()
    with pytest.raises(InvalidError):
        stub["image"] = Image.debian_slim().pip_install_private_repos(
            "github.com/ecorp/private-one@1.0.0",
            git_user="erikbern",
            secrets=[],  # Invalid: missing secret
        )

    bad_repo_refs = [
        "ecorp/private-one@1.0.0",
        "gitspace.com/corp/private-one@1.0.0",
    ]
    for invalid_ref in bad_repo_refs:
        with pytest.raises(InvalidError):
            stub["image"] = Image.debian_slim().pip_install_private_repos(
                invalid_ref,
                git_user="erikbern",
                secrets=[Secret.from_name("test-gh-read")],
            )

    stub["image"] = Image.debian_slim().pip_install_private_repos(
        "github.com/corp/private-one@1.0.0",
        "gitlab.com/corp2/private-two@0.0.2",
        git_user="erikbern",
        secrets=[
            Secret.from_dict({"GITHUB_TOKEN": "not-a-secret"}),
            Secret.from_dict({"GITLAB_TOKEN": "not-a-secret"}),
        ],
    )

    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)
        assert len(layers[0].secret_ids) == 2
        assert any(
            'pip install "git+https://erikbern:$GITHUB_TOKEN@github.com/corp/private-one@1.0.0"' in cmd
            for cmd in layers[0].dockerfile_commands
        )
        assert any(
            'pip install "git+https://erikbern:$GITLAB_TOKEN@gitlab.com/corp2/private-two@0.0.2"' in cmd
            for cmd in layers[0].dockerfile_commands
        )


def test_conda_install(servicer, client):
    stub = Stub(image=Image.conda().pip_install("numpy").conda_install("pymc3", "theano").pip_install("scikit-learn"))

    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)

        assert any("pip install scikit-learn" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("conda install pymc3 theano --yes" in cmd for cmd in layers[1].dockerfile_commands)
        assert any("pip install numpy" in cmd for cmd in layers[2].dockerfile_commands)


def test_dockerfile_image(servicer, client):
    path = os.path.join(os.path.dirname(__file__), "supports/test-dockerfile")

    stub = Stub(image=Image.from_dockerfile(path))

    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)

        assert any("RUN pip install numpy" in cmd for cmd in layers[1].dockerfile_commands)


def test_conda_update_from_environment(servicer, client):
    path = os.path.join(os.path.dirname(__file__), "supports/test-conda-environment.yml")

    stub = Stub(image=Image.conda().conda_update_from_environment(path))

    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)

        assert any("RUN conda env update" in cmd for cmd in layers[0].dockerfile_commands)
        assert any(b"foo=1.0" in f.data for f in layers[0].context_files)
        assert any(b"bar=2.1" in f.data for f in layers[0].context_files)


def test_dockerhub_install(servicer, client):
    stub = Stub(image=Image.from_registry("gisops/valhalla:latest", setup_dockerfile_commands=["RUN apt-get update"]))

    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)

        assert any("FROM gisops/valhalla:latest" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("RUN apt-get update" in cmd for cmd in layers[0].dockerfile_commands)


def test_ecr_install(servicer, client):
    image_tag = "000000000000.dkr.ecr.us-east-1.amazonaws.com/my-private-registry:latest"
    stub = Stub(
        image=Image.from_aws_ecr(
            image_tag,
            setup_dockerfile_commands=["RUN apt-get update"],
            secret=Secret.from_dict({"AWS_ACCESS_KEY_ID": "", "AWS_SECRET_ACCESS_KEY": ""}),
        )
    )

    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)

        assert any(f"FROM {image_tag}" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("RUN apt-get update" in cmd for cmd in layers[0].dockerfile_commands)


def run_f():
    print("foo!")


def test_image_run_function(client, servicer):
    stub = Stub()
    NetworkFileSystem.persisted("test-vol")
    stub["image"] = (
        Image.debian_slim().pip_install("pandas").run_function(run_f, secrets=[Secret.from_dict({"xyz": "123"})])
    )

    with stub.run(client=client):
        image_id = stub["image"].object_id
        layers = get_image_layers(image_id, servicer)
        assert "foo!" in layers[0].build_function_def
        assert "Secret.from_dict([xyz])" in layers[0].build_function_def
        # globals is none when no globals are referenced
        assert layers[0].build_function_globals == b""

    function_id = servicer.image_build_function_ids[image_id]
    assert function_id
    assert servicer.app_functions[function_id].function_name == "run_f"
    assert len(servicer.app_functions[function_id].secret_ids) == 1


def test_image_run_function_interactivity(client, servicer):
    stub = Stub()
    NetworkFileSystem.persisted("test-vol")
    stub["image"] = Image.debian_slim().pip_install("pandas").run_function(run_f)

    from modal.runner import run_stub

    with run_stub(stub, client=client, shell=True):
        image_id = stub["image"].object_id
        layers = get_image_layers(image_id, servicer)
        assert "foo!" in layers[0].build_function_def

    function_id = servicer.image_build_function_ids[image_id]
    assert function_id
    assert servicer.app_functions[function_id].function_name == "run_f"
    assert not servicer.app_functions[function_id].pty_info.enabled


VARIABLE_1 = 1
VARIABLE_2 = 3


def run_f_globals():
    print("foo!", VARIABLE_1)


def test_image_run_function_globals(client, servicer):
    global VARIABLE_1, VARIABLE_2

    stub = Stub()
    stub["image"] = Image.debian_slim().run_function(run_f_globals)

    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)
        old_globals = layers[0].build_function_globals
        assert b"VARIABLE_1" in old_globals
        assert b"VARIABLE_2" not in old_globals

    VARIABLE_1 = 3
    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)
        assert layers[0].build_function_globals != old_globals

    VARIABLE_1 = 1
    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)
        assert layers[0].build_function_globals == old_globals


VARIABLE_3 = threading.Lock()
VARIABLE_4 = "bar"


def run_f_unserializable_globals():
    print("foo!", VARIABLE_3, VARIABLE_4)


def test_image_run_unserializable_function(client, servicer):
    stub = Stub()
    stub["image"] = Image.debian_slim().run_function(run_f_unserializable_globals)

    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)
        old_globals = layers[0].build_function_globals
        assert b"VARIABLE_4" in old_globals


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
    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)
        context_files = {f.filename for layer in layers for f in layer.context_files}
        assert context_files == {"/.poetry.lock", "/.pyproject.toml", "/modal_requirements.txt"}


def test_image_build_with_context_mount(client, servicer, tmp_path):
    (tmp_path / "data.txt").write_text("hello")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sub").write_text("world")

    data_mount = Mount.from_local_dir(tmp_path, remote_path="/")

    stub = Stub()
    dockerfile = NamedTemporaryFile("w", delete=False)
    dockerfile.write("COPY . /dummy\n")
    dockerfile.close()

    stub["copy"] = Image.debian_slim().copy_local_dir(tmp_path, remote_path="/dummy")
    stub["from_dockerfile"] = Image.debian_slim().dockerfile_commands(["COPY . /dummy"], context_mount=data_mount)
    stub["dockerfile_commands"] = Image.debian_slim().from_dockerfile(dockerfile.name, context_mount=data_mount)

    with stub.run(client=client):
        for image_name, expected_layer in [("copy", 0), ("dockerfile_commands", 1), ("from_dockerfile", 0)]:
            layers = get_image_layers(stub[image_name].object_id, servicer)
            assert layers[expected_layer].context_mount_id == "mo-123", f"error in {image_name}"
            assert "COPY . /dummy" in layers[expected_layer].dockerfile_commands

        files = {f.mount_filename: f.content for f in Mount._get_files(data_mount.entries)}
        assert files == {"/data.txt": b"hello", "/data/sub": b"world"}


def test_image_env(client, servicer):
    stub = Stub(image=Image.debian_slim().env({"HELLO": "world!"}))

    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)
        assert any("ENV HELLO=" in cmd and "world!" in cmd for cmd in layers[0].dockerfile_commands)


def test_image_gpu(client, servicer):
    stub = Stub(image=Image.debian_slim().run_commands("echo 0"))
    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)
        assert layers[0].gpu_config.type == api_pb2.GPU_TYPE_UNSPECIFIED

    # TODO(erikbern): reenable this warning when we actually support different GPU types
    # with pytest.warns(DeprecationError):
    stub = Stub(image=Image.debian_slim().run_commands("echo 1", gpu=True))
    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)
        assert layers[0].gpu_config.type == api_pb2.GPU_TYPE_ANY

    stub = Stub(image=Image.debian_slim().run_commands("echo 2", gpu=gpu.A10G()))
    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)
        assert layers[0].gpu_config.type == api_pb2.GPU_TYPE_A10G


def test_image_force_build(client, servicer):
    stub = Stub()
    stub["image"] = (
        Image.debian_slim().run_commands("echo 1").pip_install("foo", force_build=True).run_commands("echo 2")
    )
    with stub.run(client=client):
        assert servicer.force_built_images == ["im-3", "im-4"]

    stub["image"] = (
        Image.from_gcp_artifact_registry("foo", force_build=True)
        .run_commands("python_packagesecho 1")
        .pip_install("foo", force_build=True)
        .run_commands("echo 2")
    )
    with stub.run(client=client):
        assert servicer.force_built_images == ["im-3", "im-4", "im-5", "im-6", "im-7", "im-8"]


def test_workdir(servicer, client):
    stub = Stub(image=Image.debian_slim().workdir("/foo/bar"))

    with stub.run(client=client):
        layers = get_image_layers(stub["image"].object_id, servicer)

        assert any("WORKDIR /foo/bar" in cmd for cmd in layers[0].dockerfile_commands)


cls_stub = Stub()

VARIABLE_5 = 1
VARIABLE_6 = 1


@cls_stub.cls(
    image=Image.debian_slim().pip_install("pandas"),
    secrets=[Secret.from_dict({"xyz": "123"})],
)
class Foo:
    def __build__(self):
        global VARIABLE_5

        print("foo!", VARIABLE_5)

    @method()
    def f(self):
        global VARIABLE_6

        print("bar!", VARIABLE_6)


def test_image_build_snapshot(client, servicer):
    with cls_stub.run(client=client):
        image_id = list(servicer.images.keys())[-1]
        layers = get_image_layers(image_id, servicer)

        assert "foo!" in layers[0].build_function_def
        assert "Secret.from_dict([xyz])" in layers[0].build_function_def
        assert any("pip install pandas" in cmd for cmd in layers[1].dockerfile_commands)

        globals = layers[0].build_function_globals
        assert b"VARIABLE_5" in globals

        # Globals and def for the main function should not affect __enter__.
        assert "bar!" not in layers[0].build_function_def
        assert b"VARIABLE_6" not in globals

    function_id = servicer.image_build_function_ids[image_id]
    assert function_id
    assert servicer.app_functions[function_id].function_name == "Foo.__build__"
    assert len(servicer.app_functions[function_id].secret_ids) == 1


def test_inside_ctx_unhydrated(client):
    image_1 = Image.debian_slim()
    image_2 = Image.debian_slim()

    with mock.patch.dict(os.environ, {"MODAL_IMAGE_ID": "im-123"}):
        # This should initially swallow the exception
        with image_1.run_inside():
            raise Exception("foo")

        # This one too
        with image_2.run_inside():
            raise Exception("bar")

        # Hydration of the image should raise the exception
        with pytest.raises(Exception, match="foo"):
            image_1._hydrate("im-123", client, None)

        # Should not raise since it's a different image
        image_2._hydrate("im-456", client, None)


def test_inside_ctx_hydrated(client):
    image_1 = Image.debian_slim()
    image_2 = Image.debian_slim()

    with mock.patch.dict(os.environ, {"MODAL_IMAGE_ID": "im-123"}):
        # Assign ids before the ctx mgr runs
        image_1._hydrate("im-123", client, None)
        image_2._hydrate("im-456", client, None)

        # Ctx manager should now raise right away
        with pytest.raises(Exception, match="baz"):
            with image_1.run_inside():
                raise Exception("baz")

        # We're not inside this image so this should be swallowed
        with image_2.run_inside():
            raise Exception("bar")
