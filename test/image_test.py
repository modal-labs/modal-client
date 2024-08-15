# Copyright Modal Labs 2022
import asyncio
import os
import pytest
import re
import sys
import threading
from hashlib import sha256
from tempfile import NamedTemporaryFile
from typing import List, Literal, get_args
from unittest import mock

from modal import App, Image, Mount, Secret, build, gpu, method
from modal._serialization import serialize
from modal.client import Client
from modal.exception import DeprecationError, InvalidError, VersionError
from modal.image import (
    SUPPORTED_PYTHON_SERIES,
    ImageBuilderVersion,
    _dockerhub_debian_codename,
    _dockerhub_python_version,
    _get_modal_requirements_path,
    _validate_python_version,
)
from modal.mount import PYTHON_STANDALONE_VERSIONS
from modal_proto import api_pb2

from .supports.skip import skip_windows


def dummy():
    ...


def test_supported_python_series():
    assert SUPPORTED_PYTHON_SERIES == PYTHON_STANDALONE_VERSIONS.keys()


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


def get_all_dockerfile_commands(image_id: str, servicer) -> str:
    layers = get_image_layers(image_id, servicer)
    return "\n".join([cmd for layer in layers for cmd in layer.dockerfile_commands])


@pytest.fixture(params=get_args(ImageBuilderVersion))
def builder_version(request, server_url_env, modal_config):
    version = request.param
    with modal_config():
        with mock.patch("test.conftest.ImageBuilderVersion", Literal[version]):  # type: ignore
            yield version


def test_python_version_validation():
    assert _validate_python_version(None) == "{0}.{1}".format(*sys.version_info)
    assert _validate_python_version("3.12") == "3.12"
    assert _validate_python_version("3.12.0") == "3.12.0"

    with pytest.raises(InvalidError, match="Unsupported Python version"):
        _validate_python_version("3.7")

    with pytest.raises(InvalidError, match="Python version must be specified as a string"):
        _validate_python_version(3.10)  # type: ignore

    with pytest.raises(InvalidError, match="Invalid Python version"):
        _validate_python_version("3.10.2.9")

    with pytest.raises(InvalidError, match="Invalid Python version"):
        _validate_python_version("3.10.x")

    with pytest.raises(InvalidError, match="Python version must be specified as 'major.minor'"):
        _validate_python_version("3.10.5", allow_micro_granularity=False)


def test_dockerhub_python_version(builder_version):
    assert _dockerhub_python_version(builder_version, "3.9.1") == "3.9.1"

    expected_39_full = {"2023.12": "3.9.15", "2024.04": "3.9.19"}[builder_version]
    assert _dockerhub_python_version(builder_version, "3.9") == expected_39_full

    v = _dockerhub_python_version(builder_version, None).split(".")
    assert len(v) == 3
    assert (int(v[0]), int(v[1])) == sys.version_info[:2]


def test_image_base(builder_version, servicer, client, test_dir):
    app = App()
    constructors = [
        (Image.debian_slim, ()),
        (Image.from_registry, ("ubuntu",)),
        (Image.from_dockerfile, (test_dir / "supports" / "test-dockerfile",)),
        (Image.micromamba, ()),
    ]
    for meth, args in constructors:
        image = meth(*args)  # type: ignore
        app.function(image=image)(dummy)
        with app.run(client=client):
            commands = get_all_dockerfile_commands(image.object_id, servicer)
            assert "COPY /modal_requirements.txt /modal_requirements.txt" in commands
            if builder_version == "2023.12":
                assert "pip install -r /modal_requirements.txt" in commands
            else:
                assert "pip install --no-cache --no-deps -r /modal_requirements.txt" in commands
                assert "rm /modal_requirements.txt" in commands


@pytest.mark.parametrize("python_version", [None, "3.10", "3.11.4"])
def test_python_version(builder_version, servicer, client, python_version):
    local_python = "{0}.{1}".format(*sys.version_info)
    expected_python = local_python if python_version is None else python_version

    app = App()
    image = Image.debian_slim() if python_version is None else Image.debian_slim(python_version)
    app.function(image=image)(dummy)
    expected_dockerhub_python = _dockerhub_python_version(builder_version, expected_python)
    expected_dockerhub_debian = _dockerhub_debian_codename(builder_version)
    assert expected_dockerhub_python.startswith(expected_python)
    with app.run(client):
        commands = get_all_dockerfile_commands(image.object_id, servicer)
        assert re.match(rf"FROM python:{expected_dockerhub_python}-slim-{expected_dockerhub_debian}", commands)

    image = Image.micromamba() if python_version is None else Image.micromamba(python_version)
    app.function(image=image)(dummy)
    if python_version is None and builder_version == "2023.12":
        expected_python = "3.9"
    with app.run(client):
        commands = get_all_dockerfile_commands(image.object_id, servicer)
        assert re.search(rf"install.* python={expected_python}", commands)


def test_image_python_packages(builder_version, servicer, client):
    app = App()
    image = (
        Image.debian_slim()
        .pip_install("sklearn[xyz]")
        .pip_install("numpy", "scipy", extra_index_url="https://xyz", find_links="https://abc?q=123", pre=True)
        .pip_install("flash-attn", extra_options="--no-build-isolation --no-cache-dir")
        .pip_install("pandas", pre=True)
    )
    app.function(image=image)(dummy)
    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        assert any("pip install 'sklearn[xyz]'" in cmd for cmd in layers[3].dockerfile_commands)
        assert any(
            "pip install numpy scipy --find-links 'https://abc?q=123' --extra-index-url https://xyz --pre" in cmd
            for cmd in layers[2].dockerfile_commands
        )
        assert any(
            "pip install flash-attn --no-build-isolation --no-cache-dir" in cmd for cmd in layers[1].dockerfile_commands
        )
        assert any("pip install pandas" + 2 * " " + "--pre" in cmd for cmd in layers[0].dockerfile_commands)

    with pytest.warns(DeprecationError):
        app = App(image=Image.debian_slim().pip_install("--no-build-isolation", "flash-attn"))
        app.function()(dummy)
        with app.run(client=client):
            pass


def test_image_kwargs_validation(builder_version, servicer, client):
    app = App()
    image = Image.debian_slim().run_commands(
        "echo hi", secrets=[Secret.from_dict({"xyz": "123"}), Secret.from_name("foo")]
    )
    app.function(image=image)(dummy)
    with pytest.raises(InvalidError):
        Image.debian_slim().run_commands(
            "echo hi",
            secrets=[
                Secret.from_dict({"xyz": "123"}),
                Secret.from_name("foo"),
                Mount.from_local_dir("/", remote_path="/"),  # type: ignore
            ],  # Mount is not a valid Secret
        )

    Image.debian_slim().copy_local_dir("/", remote_path="/dummy")
    Image.debian_slim().copy_mount(Mount.from_name("foo"), remote_path="/dummy")
    with pytest.raises(InvalidError):
        # Secret is not a valid Mount
        Image.debian_slim().copy_mount(Secret.from_dict({"xyz": "123"}), remote_path="/dummy")  # type: ignore


def test_wrong_type(builder_version, servicer, client):
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


def test_image_requirements_txt(builder_version, servicer, client):
    requirements_txt = os.path.join(os.path.dirname(__file__), "supports/test-requirements.txt")

    app = App()
    image = Image.debian_slim().pip_install_from_requirements(requirements_txt)
    app.function(image=image)(dummy)
    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)

        assert any("COPY /.requirements.txt /.requirements.txt" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("pip install -r /.requirements.txt" in cmd for cmd in layers[0].dockerfile_commands)
        assert any(b"banana" in f.data for f in layers[0].context_files)


def test_empty_install(builder_version, servicer, client):
    # Install functions with no packages should be ignored.
    app = App(
        image=Image.debian_slim()
        .pip_install()
        .pip_install([], [], [], [])
        .apt_install([])
        .run_commands()
        .micromamba_install()
    )
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert len(layers) == 1


def test_debian_slim_apt_install(builder_version, servicer, client):
    app = App(image=Image.debian_slim().pip_install("numpy").apt_install("git", "ssh").pip_install("scikit-learn"))
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)

        assert any("pip install scikit-learn" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("apt-get install -y git ssh" in cmd for cmd in layers[1].dockerfile_commands)
        assert any("pip install numpy" in cmd for cmd in layers[2].dockerfile_commands)


def test_image_pip_install_pyproject(builder_version, servicer, client):
    pyproject_toml = os.path.join(os.path.dirname(__file__), "supports/test-pyproject.toml")

    app = App()
    image = Image.debian_slim().pip_install_from_pyproject(pyproject_toml)
    app.function(image=image)(dummy)
    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)

        print(layers[0].dockerfile_commands)
        assert any("pip install 'banana >=1.2.0' 'potato >=0.1.0'" in cmd for cmd in layers[0].dockerfile_commands)


def test_image_pip_install_pyproject_with_optionals(builder_version, servicer, client):
    pyproject_toml = os.path.join(os.path.dirname(__file__), "supports/test-pyproject.toml")

    app = App()
    image = Image.debian_slim().pip_install_from_pyproject(pyproject_toml, optional_dependencies=["dev", "test"])
    app.function(image=image)(dummy)
    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)

        print(layers[0].dockerfile_commands)
        assert any(
            "pip install 'banana >=1.2.0' 'linting-tool >=0.0.0' 'potato >=0.1.0' 'pytest >=1.2.0'" in cmd
            for cmd in layers[0].dockerfile_commands
        )
        assert not (any("'mkdocs >=1.4.2'" in cmd for cmd in layers[0].dockerfile_commands))


def test_image_pip_install_private_repos(builder_version, servicer, client):
    with pytest.raises(InvalidError):
        Image.debian_slim().pip_install_private_repos(
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
            Image.debian_slim().pip_install_private_repos(
                invalid_ref,
                git_user="erikbern",
                secrets=[Secret.from_name("test-gh-read")],
            )

    image = Image.debian_slim().pip_install_private_repos(
        "github.com/corp/private-one@1.0.0",
        "gitlab.com/corp2/private-two@0.0.2",
        git_user="erikbern",
        secrets=[
            Secret.from_dict({"GITHUB_TOKEN": "not-a-secret"}),
            Secret.from_dict({"GITLAB_TOKEN": "not-a-secret"}),
        ],
    )
    app = App()
    app.function(image=image)(dummy)

    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        assert len(layers[0].secret_ids) == 2
        assert any(
            'pip install "git+https://erikbern:$GITHUB_TOKEN@github.com/corp/private-one@1.0.0"' in cmd
            for cmd in layers[0].dockerfile_commands
        )
        assert any(
            'pip install "git+https://erikbern:$GITLAB_TOKEN@gitlab.com/corp2/private-two@0.0.2"' in cmd
            for cmd in layers[0].dockerfile_commands
        )


def test_dockerfile_image(builder_version, servicer, client):
    path = os.path.join(os.path.dirname(__file__), "supports/test-dockerfile")

    app = App(image=Image.from_dockerfile(path))
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)

        assert any("RUN pip install numpy" in cmd for cmd in layers[1].dockerfile_commands)


def test_conda_install(builder_version, servicer, client):
    with pytest.warns(DeprecationError, match="Image.micromamba"):
        image = Image.conda().pip_install("numpy").conda_install("pymc3", "theano").pip_install("scikit-learn")
    app = App(image=image)
    app.function(image=image)(dummy)

    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)

        assert any("pip install scikit-learn" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("conda install pymc3 theano --yes" in cmd for cmd in layers[1].dockerfile_commands)
        assert any("pip install numpy" in cmd for cmd in layers[2].dockerfile_commands)


def test_conda_update_from_environment(builder_version, servicer, client):
    path = os.path.join(os.path.dirname(__file__), "supports/test-conda-environment.yml")

    with pytest.warns(DeprecationError, match="Image.micromamba"):
        app = App(image=Image.conda().conda_update_from_environment(path))

    app.function()(dummy)
    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)

        assert any("RUN conda env update" in cmd for cmd in layers[0].dockerfile_commands)
        assert any(b"foo=1.0" in f.data for f in layers[0].context_files)
        assert any(b"bar=2.1" in f.data for f in layers[0].context_files)


def test_micromamba_install(builder_version, servicer, client):
    spec_file = os.path.join(os.path.dirname(__file__), "supports/test-conda-environment.yml")
    image = (
        Image.micromamba()
        .pip_install("numpy")
        .micromamba_install("pymc3", "theano", channels=["conda-forge"])
        .pip_install("scikit-learn")
        .micromamba_install(spec_file=spec_file)
    )
    app = App(image=image)
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)

        assert any(
            "COPY /test-conda-environment.yml /test-conda-environment.yml" in cmd
            for cmd in layers[0].dockerfile_commands
        )
        assert any("micromamba install -f /test-conda-environment.yml" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("pip install scikit-learn" in cmd for cmd in layers[1].dockerfile_commands)
        assert any(
            "micromamba install pymc3 theano -c conda-forge --yes" in cmd for cmd in layers[2].dockerfile_commands
        )
        assert any("pip install numpy" in cmd for cmd in layers[3].dockerfile_commands)
        assert any(b"foo=1.0" in f.data for f in layers[0].context_files)
        assert any(b"bar=2.1" in f.data for f in layers[0].context_files)


def test_run_commands(builder_version, servicer, client):
    base = Image.debian_slim()

    command = "echo 'Hello Modal'"
    app = App(image=base.run_commands(command))
    app.function()(dummy)
    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert layers[0].dockerfile_commands[1] == f"RUN {command}"

    commands = ["echo 'Hello world'", "touch agi.yaml"]
    for image in [base.run_commands(commands), base.run_commands(*commands)]:
        app = App(image=image)
        app.function()(dummy)
        with app.run(client=client):
            layers = get_image_layers(app.image.object_id, servicer)
            for i, cmd in enumerate(commands, 1):
                assert layers[0].dockerfile_commands[i] == f"RUN {cmd}"


def test_dockerhub_install(builder_version, servicer, client):
    app = App(image=Image.from_registry("gisops/valhalla:latest", setup_dockerfile_commands=["RUN apt-get update"]))
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)

        assert any("FROM gisops/valhalla:latest" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("RUN apt-get update" in cmd for cmd in layers[0].dockerfile_commands)


def test_ecr_install(builder_version, servicer, client):
    image_tag = "000000000000.dkr.ecr.us-east-1.amazonaws.com/my-private-registry:latest"
    app = App(
        image=Image.from_aws_ecr(
            image_tag,
            setup_dockerfile_commands=["RUN apt-get update"],
            secret=Secret.from_dict({"AWS_ACCESS_KEY_ID": "", "AWS_SECRET_ACCESS_KEY": ""}),
        )
    )

    app.function()(dummy)
    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)

        assert any(f"FROM {image_tag}" in cmd for cmd in layers[0].dockerfile_commands)
        assert any("RUN apt-get update" in cmd for cmd in layers[0].dockerfile_commands)


def run_f():
    print("foo!")


def test_image_run_function(builder_version, servicer, client):
    app = App()
    app.image = (
        Image.debian_slim().pip_install("pandas").run_function(run_f, secrets=[Secret.from_dict({"xyz": "123"})])
    )

    app.function()(dummy)
    with app.run(client=client):
        image_id = app.image.object_id
        layers = get_image_layers(image_id, servicer)
        assert "foo!" in layers[0].build_function.definition
        assert "Secret.from_dict([xyz])" in layers[0].build_function.definition
        # globals is none when no globals are referenced
        assert layers[0].build_function.globals == b""

    function_id = servicer.image_build_function_ids[image_id]
    assert function_id
    assert servicer.app_functions[function_id].function_name == "run_f"
    assert len(servicer.app_functions[function_id].secret_ids) == 1

    with pytest.raises(InvalidError, match="does not support lambda functions"):
        Image.debian_slim().run_function(lambda x: x)

    with pytest.raises(InvalidError, match="must be a function"):
        Image.debian_slim().run_function([])  # type: ignore  # Testing runtime error for bad type


def test_image_run_function_interactivity(builder_version, servicer, client):
    app = App()
    app.image = Image.debian_slim().pip_install("pandas").run_function(run_f)
    app.function()(dummy)

    from modal.runner import run_app

    with run_app(app, client=client):
        image_id = app.image.object_id
        layers = get_image_layers(image_id, servicer)
        assert "foo!" in layers[0].build_function.definition

    function_id = servicer.image_build_function_ids[image_id]
    assert function_id
    assert servicer.app_functions[function_id].function_name == "run_f"
    assert not servicer.app_functions[function_id].pty_info.enabled


VARIABLE_1 = 1
VARIABLE_2 = 3


def run_f_globals():
    print("foo!", VARIABLE_1)


def test_image_run_function_globals(builder_version, servicer, client):
    global VARIABLE_1, VARIABLE_2

    app = App()
    app.image = Image.debian_slim().run_function(run_f_globals)
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        old_globals = layers[0].build_function.globals
        assert b"VARIABLE_1" in old_globals
        assert b"VARIABLE_2" not in old_globals

    VARIABLE_1 = 3
    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert layers[0].build_function.globals != old_globals

    VARIABLE_1 = 1
    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert layers[0].build_function.globals == old_globals


VARIABLE_3 = threading.Lock()
VARIABLE_4 = "bar"


def run_f_unserializable_globals():
    print("foo!", VARIABLE_3, VARIABLE_4)


def test_image_run_unserializable_function(builder_version, servicer, client):
    app = App()
    app.image = Image.debian_slim().run_function(run_f_unserializable_globals)
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        old_globals = layers[0].build_function.globals
        assert b"VARIABLE_4" in old_globals


def run_f_with_args(arg, *, kwarg):
    print("building!", arg, kwarg)


def test_image_run_function_with_args(builder_version, servicer, client):
    app = App()
    app.image = Image.debian_slim().run_function(run_f_with_args, args=("foo",), kwargs={"kwarg": "bar"})
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        input = layers[0].build_function.input
        assert input.args == serialize((("foo",), {"kwarg": "bar"}))


def test_poetry(builder_version, servicer, client):
    path = os.path.join(os.path.dirname(__file__), "supports/pyproject.toml")

    # No lockfile provided and there's no lockfile found
    # TODO we deferred the exception until _load runs, not sure how to test that here
    # with pytest.raises(NotFoundError):
    #     Image.debian_slim().poetry_install_from_file(path)

    # Explicitly ignore lockfile - this should work
    Image.debian_slim().poetry_install_from_file(path, ignore_lockfile=True)

    # Provide lockfile explicitly - this should also work
    lockfile_path = os.path.join(os.path.dirname(__file__), "supports/special_poetry.lock")
    image = Image.debian_slim().poetry_install_from_file(path, lockfile_path)

    # Build iamge
    app = App()
    app.function(image=image)(dummy)
    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        context_files = {f.filename for layer in layers for f in layer.context_files}
        assert context_files == {"/.poetry.lock", "/.pyproject.toml", "/modal_requirements.txt"}


@pytest.fixture
def tmp_path_with_content(tmp_path):
    (tmp_path / "data.txt").write_text("hello")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sub").write_text("world")
    return tmp_path


def test_image_copy_local_dir(builder_version, servicer, client, tmp_path_with_content):
    app = App()
    app.image = Image.debian_slim().copy_local_dir(tmp_path_with_content, remote_path="/dummy")
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert "COPY . /dummy" in layers[0].dockerfile_commands
        mount_id = layers[0].context_mount_id
        assert set(servicer.mount_contents[mount_id].keys()) == {"/data.txt", "/data/sub"}


def test_image_docker_command_copy(builder_version, servicer, client, tmp_path_with_content):
    app = App()
    data_mount = Mount.from_local_dir(tmp_path_with_content, remote_path="/")
    app.image = Image.debian_slim().dockerfile_commands(["COPY . /dummy"], context_mount=data_mount)
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert "COPY . /dummy" in layers[0].dockerfile_commands
        files = {f.mount_filename: f.content for f in Mount._get_files(data_mount.entries)}
        assert files == {"/data.txt": b"hello", "/data/sub": b"world"}


def test_image_dockerfile_copy(builder_version, servicer, client, tmp_path_with_content):
    dockerfile = NamedTemporaryFile("w", delete=False)
    dockerfile.write("COPY . /dummy\n")
    dockerfile.close()

    app = App()
    data_mount = Mount.from_local_dir(tmp_path_with_content, remote_path="/")
    app.image = Image.debian_slim().from_dockerfile(dockerfile.name, context_mount=data_mount)
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert "COPY . /dummy" in layers[1].dockerfile_commands
        files = {f.mount_filename: f.content for f in Mount._get_files(data_mount.entries)}
        assert files == {"/data.txt": b"hello", "/data/sub": b"world"}


def test_image_docker_command_entrypoint(builder_version, servicer, client, tmp_path_with_content):
    app = App()
    app.image = Image.debian_slim().entrypoint([])
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert "ENTRYPOINT []" in layers[0].dockerfile_commands


def test_image_docker_command_entrypoint_nonempty(builder_version, servicer, client, tmp_path_with_content):
    app = App()
    app.image = (
        Image.debian_slim()
        .dockerfile_commands(
            [
                "FROM public.ecr.aws/docker/library/alpine:3.19.1",
                'RUN echo $\'#!/usr/bin/env sh\necho "hi"\nexec "$@"\' > /temp.sh',
                "RUN chmod +x /temp.sh",
            ]
        )
        .entrypoint(["/temp.sh"])
    )

    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert 'ENTRYPOINT ["/temp.sh"]' in layers[0].dockerfile_commands


def test_image_docker_command_shell(builder_version, servicer, client, tmp_path_with_content):
    app = App()
    app.image = Image.debian_slim().shell(["/bin/sh", "-c"])
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert 'SHELL ["/bin/sh", "-c"]' in layers[0].dockerfile_commands


def test_image_env(builder_version, servicer, client):
    app = App(image=Image.debian_slim().env({"HELLO": "world!"}))
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert any("ENV HELLO=" in cmd and "world!" in cmd for cmd in layers[0].dockerfile_commands)


def test_image_gpu(builder_version, servicer, client):
    app = App(image=Image.debian_slim().run_commands("echo 0"))
    app.function()(dummy)
    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert layers[0].gpu_config.type == api_pb2.GPU_TYPE_UNSPECIFIED

    with pytest.raises(DeprecationError):
        Image.debian_slim().run_commands("echo 0", gpu=True)

    app = App(image=Image.debian_slim().run_commands("echo 1", gpu="any"))
    app.function()(dummy)
    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert layers[0].gpu_config.type == api_pb2.GPU_TYPE_ANY

    app = App(image=Image.debian_slim().run_commands("echo 2", gpu=gpu.A10G()))
    app.function()(dummy)
    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert layers[0].gpu_config.type == api_pb2.GPU_TYPE_A10G


def test_image_force_build(builder_version, servicer, client):
    app = App()
    app.image = Image.debian_slim().run_commands("echo 1").pip_install("foo", force_build=True).run_commands("echo 2")
    app.function()(dummy)
    with app.run(client=client):
        assert servicer.force_built_images == ["im-3", "im-4"]

    app.image = (
        Image.from_gcp_artifact_registry("foo", force_build=True)
        .run_commands("python_packagesecho 1")
        .pip_install("foo", force_build=True)
        .run_commands("echo 2")
    )
    app.function()(dummy)
    with app.run(client=client):
        assert servicer.force_built_images == ["im-3", "im-4", "im-5", "im-6", "im-7", "im-8"]


def test_workdir(builder_version, servicer, client):
    app = App(image=Image.debian_slim().workdir("/foo/bar"))
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)

        assert any("WORKDIR /foo/bar" in cmd for cmd in layers[0].dockerfile_commands)


cls_app = App()

VARIABLE_5 = 1
VARIABLE_6 = 1


@cls_app.cls(
    image=Image.debian_slim().pip_install("pandas"),
    secrets=[Secret.from_dict({"xyz": "123"})],
)
class Foo:
    @build()
    def build_func(self):
        global VARIABLE_5

        print("foo!", VARIABLE_5)

    @method()
    def f(self):
        global VARIABLE_6

        print("bar!", VARIABLE_6)


class FooInstance:
    not_used_by_build_method: str = "normal"
    used_by_build_method: str = "normal"

    @build()
    def build_func(self):
        global VARIABLE_5

        print("global variable", VARIABLE_5)
        print("static class var", FooInstance.used_by_build_method)
        FooInstance.used_by_build_method = "normal"


def test_image_cls_var_rebuild(client, servicer):
    rebuild_app = App()
    image_ids = []
    rebuild_app.cls(image=Image.debian_slim())(FooInstance)
    with rebuild_app.run(client=client):
        image_ids = list(servicer.images)
    FooInstance.used_by_build_method = "rebuild"
    rebuild_app.cls(image=Image.debian_slim())(FooInstance)
    with rebuild_app.run(client=client):
        image_ids_rebuild = list(servicer.images)
    # Ensure that a new image was created
    assert image_ids[-1] != image_ids_rebuild[-1]
    FooInstance.used_by_build_method = "normal"
    rebuild_app.cls(image=Image.debian_slim())(FooInstance)
    with rebuild_app.run(client=client):
        image_ids = list(servicer.images)
    # Ensure that no new image was created
    assert len(image_ids) == len(image_ids_rebuild)


def test_image_cls_var_no_rebuild(client, servicer):
    rebuild_app = App()
    image_id = -1
    rebuild_app.cls(image=Image.debian_slim())(FooInstance)
    with rebuild_app.run(client=client):
        image_id = list(servicer.images)[-1]
    rebuild_app.cls(image=Image.debian_slim())(FooInstance)
    with rebuild_app.run(client=client):
        image_id2 = list(servicer.images)[-1]
    FooInstance.not_used_by_build_method = "no rebuild"
    rebuild_app.cls(image=Image.debian_slim())(FooInstance)
    with rebuild_app.run(client=client):
        image_id3 = list(servicer.images)[-1]
    assert image_id == image_id2
    assert image_id2 == image_id3


def test_image_build_snapshot(client, servicer):
    with cls_app.run(client=client):
        image_id = list(servicer.images)[-1]
        layers = get_image_layers(image_id, servicer)

        assert "foo!" in layers[0].build_function.definition
        assert "Secret.from_dict([xyz])" in layers[0].build_function.definition
        assert any("pip install pandas" in cmd for cmd in layers[1].dockerfile_commands)

        globals = layers[0].build_function.globals
        assert b"VARIABLE_5" in globals

        # Globals and def for the main function should not affect build step.
        assert "bar!" not in layers[0].build_function.definition
        assert b"VARIABLE_6" not in globals

    function_id = servicer.image_build_function_ids[image_id]
    assert function_id
    assert servicer.app_functions[function_id].function_name == "Foo.build_func"
    assert len(servicer.app_functions[function_id].secret_ids) == 1


def test_inside_ctx_unhydrated(client):
    image_1 = Image.debian_slim()
    image_2 = Image.debian_slim()

    with mock.patch.dict(os.environ, {"MODAL_IMAGE_ID": "im-123"}):
        # This should initially swallow the exception
        with image_1.imports():
            raise ImportError("foo")

        # This one too
        with image_2.imports():
            raise ImportError("bar")

        # non-ImportErrors should trigger a warning
        with pytest.warns(match="ImportError"):
            with image_2.imports():
                raise Exception("foo")

        # Old one raises
        with pytest.raises(DeprecationError, match="imports()"):
            image_1.run_inside()

        # Hydration of the image should raise the exception
        with pytest.raises(ImportError, match="foo"):
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
        with pytest.raises(ImportError, match="baz"):
            with image_1.imports():
                raise ImportError("baz")

        # We're not inside this image so this should be swallowed
        with image_2.imports():
            raise ImportError("bar")


@pytest.mark.parametrize("python_version", ["3.11", "3.12", "3.12.1", "3.12.1-gnu"])
def test_get_modal_requirements_path(builder_version, python_version):
    path = _get_modal_requirements_path(builder_version, python_version)
    if builder_version == "2023.12" and python_version.startswith("3.12"):
        assert path.endswith("2023.12.312.txt")
    else:
        assert path.endswith(f"{builder_version}.txt")


def test_image_builder_version(servicer, test_dir, modal_config):
    app = App(image=Image.debian_slim())
    app.function()(dummy)

    # TODO use a single with statement and tuple of managers when we drop Py3.8
    test_requirements = str(test_dir / "supports" / "test-requirements.txt")
    with mock.patch("modal.image._get_modal_requirements_path", lambda *_, **__: test_requirements):
        with mock.patch("modal.image._dockerhub_python_version", lambda *_, **__: "3.11.0"):
            with mock.patch("modal.image._dockerhub_debian_codename", lambda *_, **__: "bullseye"):
                with mock.patch("test.conftest.ImageBuilderVersion", Literal["2000.01"]):
                    with mock.patch("modal.image.ImageBuilderVersion", Literal["2000.01"]):
                        with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, ("ak-123", "as-xyz")) as client:
                            with modal_config():
                                with app.run(client=client):
                                    assert servicer.image_builder_versions
                                    for version in servicer.image_builder_versions.values():
                                        assert version == "2000.01"


def test_image_builder_supported_versions(servicer):
    app = App(image=Image.debian_slim())
    app.function()(dummy)

    # TODO use a single with statement and tuple of managers when we drop Py3.8
    with pytest.raises(VersionError, match=r"This version of the modal client supports.+{'2000.01'}"):
        with mock.patch("modal.image.ImageBuilderVersion", Literal["2000.01"]):
            with mock.patch("test.conftest.ImageBuilderVersion", Literal["2023.11"]):
                with Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, ("ak-123", "as-xyz")) as client:
                    with app.run(client=client):
                        pass


@pytest.fixture
def force_2023_12(modal_config):
    with mock.patch("test.conftest.ImageBuilderVersion", Literal["2023.12"]):
        with modal_config():
            yield


@skip_windows("Different hash values for context file paths")
def test_image_stability_on_2023_12(force_2023_12, servicer, client, test_dir):
    def get_hash(img: Image) -> str:
        app = App(image=img)
        app.function()(dummy)
        with app.run(client=client):
            layers = get_image_layers(app.image.object_id, servicer)
            commands = [layer.dockerfile_commands for layer in layers]
            context_files = [[(f.filename, f.data) for f in layer.context_files] for layer in layers]
        return sha256(repr(list(zip(commands, context_files))).encode()).hexdigest()

    if sys.version_info[:2] == (3, 11):
        # Matches my development environment — default is to match Python version from local system
        img = Image.debian_slim()
        assert get_hash(img) == "183b86356d9eb3bd3d78adf70f16b35b63ba9bf4e1816b0cacc549541718e555"

    img = Image.debian_slim(python_version="3.12")
    assert get_hash(img) == "53b6205e1dc2a0ca7ebed862e4f3a5887367587be13e81f65a4ac8f8a1e9be91"

    if sys.version_info[:2] < (3, 12):
        # Client dependencies on 3.12 are different
        img = Image.from_registry("ubuntu:22.04")
        assert get_hash(img) == "b5f1cc544a412d1b23a5ebf9a8859ea9a86975ecbc7325b83defc0ce3fe956d3"

        with pytest.warns(DeprecationError):
            img = Image.conda()
        assert get_hash(img) == "f69d6af66fb5f1a2372a61836e6166ce79ebe2cd628d12addea8e8e80cc98dc1"

        img = Image.micromamba()
        assert get_hash(img) == "fa883741544ea191ecd197c8f83a1ffe9912575faa8c107c66b3dda761b2e401"

        img = Image.from_dockerfile(test_dir / "supports" / "test-dockerfile")
        assert get_hash(img) == "0aec2f66f28ee7511c1b36604214ae7b40d9bc1fa3e6b8883001e933a966ff78"

    with pytest.warns(DeprecationError):
        img = Image.conda(python_version="3.12")
    assert get_hash(img) == "c4b3f7350116d323dded29c9c9b78b62593f0fc943ccf83a09b27185bfdc2a07"

    img = Image.micromamba(python_version="3.12")
    assert get_hash(img) == "468befe16f703a3ae1a794dfe54c1a3445ca0ffda233f55f1d66c45ad608e8aa"

    base = Image.debian_slim(python_version="3.12")

    img = base.run_commands("echo 'Hello Modal'", "rm /usr/local/bin/kubectl")
    assert get_hash(img) == "4e1ac62eb33b44dd16940c9d2719eb79f945cee61cbf4641ca99b19cd9e0976d"

    img = base.pip_install("torch~=2.2", "transformers==4.23.0", pre=True, index_url="agi.se")
    assert get_hash(img) == "2a4fa8e3b32c70a41b3a3efd5416540b1953430543f6c27c984e7f969c2ca874"

    with pytest.warns(DeprecationError):
        img = base.conda_install("torch=2.2", "transformers<4.23.0", channels=["conda-forge", "my-channel"])
    assert get_hash(img) == "dd6f27f636293996a64a98c250161d8092cb23d02629d9070493f00aad8d7266"

    img = base.pip_install_from_requirements(test_dir / "supports" / "test-requirements.txt")
    assert get_hash(img) == "69d41e699d4ecef399e51e8460f8857aa0ec57f71f00eca81c8886ec062e5c2b"

    with pytest.warns(DeprecationError):
        img = base.conda_update_from_environment(test_dir / "supports" / "test-conda-environment.yml")
    assert get_hash(img) == "00940e0ee2998bfe0a337f51a5fdf5f4b29bf9d42dda3635641d44bfeb42537e"

    img = base.poetry_install_from_file(
        test_dir / "supports" / "test-pyproject.toml",
        poetry_lockfile=test_dir / "supports" / "special_poetry.lock",
    )
    assert get_hash(img) == "a25dd4cc2e8d88f92bfdaf2e82b9d74144d1928926bf6be2ca1cdfbbf562189e"


@pytest.fixture
def force_2024_04(modal_config):
    with mock.patch("test.conftest.ImageBuilderVersion", Literal["2024.04"]):
        with modal_config():
            yield


@skip_windows("Different hash values for context file paths")
def test_image_stability_on_2024_04(force_2024_04, servicer, client, test_dir):
    def get_hash(img: Image) -> str:
        app = App(image=img)
        app.function()(dummy)
        with app.run(client=client):
            layers = get_image_layers(app.image.object_id, servicer)
            commands = [layer.dockerfile_commands for layer in layers]
            context_files = [[(f.filename, f.data) for f in layer.context_files] for layer in layers]
        return sha256(repr(list(zip(commands, context_files))).encode()).hexdigest()

    if sys.version_info[:2] == (3, 11):
        # Matches my development environment — default is to match Python version from local system
        img = Image.debian_slim()
        assert get_hash(img) == "b8f887744fa285250c72fccbecbca8b946726ec27b6acb804cd66cb2fe02cc63"

    img = Image.debian_slim(python_version="3.12")
    assert get_hash(img) == "6d01817be6e04444fe2ec8fa13615ec9e2aae4e415c5db2464ecd9f42ed2ed91"

    img = Image.from_registry("ubuntu:22.04")
    assert get_hash(img) == "285272f4049c812a72e1deecd4c98ac41be516670738214d2e7c4eb98a8f1ce8"

    img = Image.from_dockerfile(test_dir / "supports" / "test-dockerfile")
    assert get_hash(img) == "a683ed2a2fd9ca960d818aebd7459932494da63aa27d5d84443c578a5ba3fe05"

    with pytest.warns(DeprecationError):
        img = Image.conda()
    if sys.version_info[:2] == (3, 11):
        assert get_hash(img) == "404e1d80bd639321d1115ae00b8d1b6f3d257be7d400bad46d3c39da09c193c8"
    elif sys.version_info[:2] == (3, 10):
        # Assert that we follow the local Python, which is a new behavior in 2024.04
        assert get_hash(img) == "9299b7935461e021ea6a3749adac8ea4de333fd443e74192739fbdd60da3b0b5"

    with pytest.warns(DeprecationError):
        img = Image.conda(python_version="3.12")
    assert get_hash(img) == "113d959483ff099ba161438f02a370322bc53a68d5c5989245f4002b08e3be9d"

    img = Image.micromamba()
    if sys.version_info[:2] == (3, 11):
        assert get_hash(img) == "8c0a30c7d14eb709953161cae39aa7d39afe3bb5014b7c6cf5dd93de56dcb32b"
    elif sys.version_info[:2] == (3, 10):
        # Assert that we follow the local Python, which is a new behavior in 2024.04
        assert get_hash(img) == "4a6e9d94e3b9a15158dd97eeaf0275c8f5a80733f5acfdc8ad1a88094468dd5e"

    img = Image.micromamba(python_version="3.12")
    assert get_hash(img) == "966e1d1f3f652cfc2cd9dd7054b14a9883163d139311429857ecff7a9190b319"

    base = Image.debian_slim(python_version="3.12")

    img = base.run_commands("echo 'Hello Modal'", "rm /usr/local/bin/kubectl")
    assert get_hash(img) == "acd4db6d206ea605f1bad4727acd654fb32c28e1f2fe7e9fe9a602ed54723828"

    img = base.pip_install("torch~=2.2", "transformers==4.23.0", pre=True, index_url="agi.se")
    assert get_hash(img) == "4e8cea916369fc545a5139312a9633691f7624ec2b6d4075014a7602b23584c0"

    with pytest.warns(DeprecationError):
        img = base.conda_install("torch=2.2", "transformers<4.23.0", channels=["conda-forge", "my-channel"])
    assert get_hash(img) == "bd9fb2b86a39886d618b37950d1a7c4d8826048f191fa00c5f87c71be88bf8f7"

    img = base.pip_install_from_requirements(test_dir / "supports" / "test-requirements.txt")
    assert get_hash(img) == "78392aca4ea135ab53b9d183eedbb2a7e32f9b3c0cfb42b03a7bd7c4f013f3c8"

    with pytest.warns(DeprecationError):
        img = base.conda_update_from_environment(test_dir / "supports" / "test-conda-environment.yml")
    assert get_hash(img) == "4bb5fd232956050e66256d7e00c088e1c7cd4d7217bc0d15bcc4fbd08aa2f3b6"

    img = base.micromamba_install(
        "torch=2.2",
        "transformers<4.23.0",
        spec_file=test_dir / "supports" / "test-conda-environment.yml",
        channels=["conda-forge", "my-channel"],
    )
    assert get_hash(img) == "d9d4c9fe24769ce587877b9752a64486e8f7d8520731110bd2fa666de82f43fd"

    img = base.poetry_install_from_file(
        test_dir / "supports" / "test-pyproject.toml",
        poetry_lockfile=test_dir / "supports" / "special_poetry.lock",
    )
    assert get_hash(img) == "bfce5811c04c1243f12cbb9cca1522cb901f52410986925bcfa3b3c2d7adc7a0"


parallel_app = App()


@parallel_app.function(image=Image.debian_slim().run_commands("sleep 1", "echo hi"))
def f1():
    pass


@parallel_app.function(image=Image.debian_slim().run_commands("sleep 1", "echo bye"))
def f2():
    pass


def test_image_parallel_build(builder_version, servicer, client):
    num_concurrent = 0

    async def MockImageJoinStreaming(self, stream):
        nonlocal num_concurrent
        num_concurrent += 1
        while num_concurrent < 2:
            await asyncio.sleep(0.01)

        await stream.send_message(
            api_pb2.ImageJoinStreamingResponse(
                result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS)
            )
        )

    with servicer.intercept() as ctx:
        ctx.set_responder("ImageJoinStreaming", MockImageJoinStreaming)
        with parallel_app.run(client=client):
            pass


@pytest.mark.asyncio
async def test_logs(servicer, client):
    app = App()
    image = Image.debian_slim().pip_install("foobarbaz")
    app.function(image=image)(dummy)
    async with app.run.aio(client=client):
        pass

    logs = [data async for data in image._logs.aio()]
    assert logs == ["build starting\n", "build finished\n"]
