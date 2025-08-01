# Copyright Modal Labs 2022
import asyncio
import os
import pytest
import re
import shutil
import sys
import threading
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Callable, Literal, Sequence, Union, get_args
from unittest import mock

import modal
from modal import App, Dict, Image, Secret, environments
from modal._serialization import serialize
from modal._utils.async_utils import synchronizer
from modal.client import Client
from modal.exception import ExecutionError, InvalidError, ModuleNotMountable, NotFoundError, VersionError
from modal.experimental import raw_dockerfile_image, raw_registry_image
from modal.file_pattern_matcher import FilePatternMatcher
from modal.image import (
    SUPPORTED_PYTHON_SERIES,
    ImageBuilderVersion,
    _base_image_config,
    _dockerhub_python_version,
    _get_modal_requirements_path,
    _validate_python_version,
)
from modal.mount import PYTHON_STANDALONE_VERSIONS
from modal.runner import deploy_app
from modal_proto import api_pb2

from .supports.skip import skip_windows

# Avoid parametrizing tests over ImageBuilderVersion not supported by current Python
PYTHON_MAJOR_MINOR = "{}.{}".format(*sys.version_info)
SUPPORTED_IMAGE_BUILDER_VERSIONS = [
    v for v in get_args(ImageBuilderVersion) if PYTHON_MAJOR_MINOR in SUPPORTED_PYTHON_SERIES[v]
]


def dummy() -> None:
    return None


def test_supported_python_series():
    for builder_version in get_args(ImageBuilderVersion):
        assert SUPPORTED_PYTHON_SERIES[builder_version] <= list(PYTHON_STANDALONE_VERSIONS)


def get_image_layers(image_id: str, servicer) -> list[api_pb2.Image]:
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


@pytest.fixture(params=SUPPORTED_IMAGE_BUILDER_VERSIONS)
def builder_version(request, server_url_env, modal_config):
    builder_version = request.param
    with modal_config():
        with mock.patch("test.conftest.ImageBuilderVersion", Literal[builder_version]):  # type: ignore
            yield builder_version


@pytest.fixture(autouse=True)
def clear_environment_cache():
    # Clear the environment cache so we can mock the server returning different image builder versions
    # Alternatively could rewrite some of those tests to use different environments?
    environments.ENVIRONMENT_CACHE.clear()


def test_python_version_validation(builder_version):
    assert _validate_python_version(None, builder_version) == "{}.{}".format(*sys.version_info)
    assert _validate_python_version("3.12", builder_version) == "3.12"
    assert _validate_python_version("3.12.0", builder_version) == "3.12.0"

    with pytest.raises(InvalidError, match="Unsupported Python version"):
        _validate_python_version("3.7", builder_version)

    with pytest.raises(InvalidError, match="Python version must be specified as a string"):
        _validate_python_version(3.10, builder_version)  # type: ignore

    with pytest.raises(InvalidError, match="Invalid Python version"):
        _validate_python_version("3.10.2.9", builder_version)

    with pytest.raises(InvalidError, match="Invalid Python version"):
        _validate_python_version("3.10.x", builder_version)

    with pytest.raises(InvalidError, match="Python version must be specified as 'major.minor'"):
        _validate_python_version("3.10.5", builder_version, allow_micro_granularity=False)


def test_dockerhub_python_version(builder_version):
    assert _dockerhub_python_version(builder_version, "3.9.1") == "3.9.1"

    expected_39_full = {
        "2023.12": "3.9.15",
        "2024.04": "3.9.19",
        "2024.10": "3.9.20",
        "2025.06": "3.9.22",
        "PREVIEW": "3.9.22",
    }[builder_version]
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
            if builder_version <= "2024.10":
                assert "COPY /modal_requirements.txt /modal_requirements.txt" in commands
                if builder_version == "2023.12":
                    assert "pip install -r /modal_requirements.txt" in commands
                else:
                    assert "rm /modal_requirements.txt" in commands
                    if builder_version == "2024.04":
                        assert "pip install --no-cache --no-deps -r /modal_requirements.txt" in commands
                    else:
                        assert (
                            "uv pip install --system --compile-bytecode --no-cache --no-deps -r /modal_requirements.txt"
                        ) in commands
            else:
                assert "modal_requirements.txt" not in commands


@pytest.mark.parametrize("python_version", [None, "3.10", "3.11.4"])
def test_python_version(builder_version, servicer, client, python_version):
    local_python = "{}.{}".format(*sys.version_info)
    expected_python = local_python if python_version is None else python_version

    app = App()
    image = Image.debian_slim() if python_version is None else Image.debian_slim(python_version)
    app.function(image=image)(dummy)
    expected_dockerhub_python = _dockerhub_python_version(builder_version, expected_python)
    expected_dockerhub_debian = _base_image_config("debian", builder_version)
    assert expected_dockerhub_python.startswith(expected_python)
    with app.run(client=client):
        commands = get_all_dockerfile_commands(image.object_id, servicer)
        assert re.match(rf"FROM python:{expected_dockerhub_python}-slim-{expected_dockerhub_debian}", commands)

    image = Image.micromamba() if python_version is None else Image.micromamba(python_version)
    app.function(image=image)(dummy)
    if python_version is None and builder_version == "2023.12":
        expected_python = "3.9"
    with app.run(client=client):
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

    with pytest.raises(InvalidError, match="try the `extra_options` parameter instead"):
        app = App(image=Image.debian_slim().pip_install("--no-build-isolation", "flash-attn"))
        app.function()(dummy)
        with app.run(client=client):
            pass


def test_image_uv_python_packages(builder_version, servicer, client, test_dir):
    requirements = test_dir / "supports" / "test-requirements.txt"
    app = App()
    image = (
        Image.debian_slim()
        .uv_pip_install("sklearn[xyz]")
        .uv_pip_install("numpy", "scipy", extra_index_url="https://xyz", find_links="https://abc?q=123", pre=True)
        .uv_pip_install("flash-attn", extra_options="--no-build-isolation")
        .uv_pip_install("pandas", pre=True)
        .uv_pip_install(requirements=[requirements, requirements])
    )
    app.function(image=image)(dummy)
    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        assert any(
            "/.uv/uv pip install --python $(command -v python) --compile-bytecode 'sklearn[xyz]'" in cmd
            for cmd in layers[4].dockerfile_commands
        )
        assert any(
            "/.uv/uv pip install --python $(command -v python) --compile-bytecode "
            "--find-links 'https://abc?q=123' --extra-index-url https://xyz --prerelease allow numpy scipy" in cmd
            for cmd in layers[3].dockerfile_commands
        )
        assert any(
            "/.uv/uv pip install --python $(command -v python) --compile-bytecode --no-build-isolation flash-attn"
            in cmd
            for cmd in layers[2].dockerfile_commands
        )
        assert any(
            "/.uv/uv pip install --python $(command -v python) --compile-bytecode --prerelease allow pandas" in cmd
            for cmd in layers[1].dockerfile_commands
        )
        assert any(
            "/.uv/uv pip install --python $(command -v python) --compile-bytecode --requirements "
            "/.uv/0/test-requirements.txt" in cmd
            for cmd in layers[0].dockerfile_commands
        )
        assert any(
            "COPY /.0_test-requirements.txt /.uv/0/test-requirements.txt" in cmd
            for cmd in layers[0].dockerfile_commands
        )
        assert any(
            "COPY /.1_test-requirements.txt /.uv/1/test-requirements.txt" in cmd
            for cmd in layers[0].dockerfile_commands
        )


def test_image_uv_python_requirements_error():
    with pytest.raises(InvalidError, match="requirements must be None or a list of strings"):
        Image.debian_slim().uv_pip_install(requirements="requirements.txt")  # type: ignore


def test_run_commands_secrets_type_validation(builder_version, servicer, client):
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
                Dict.from_name("mydict"),  # type: ignore
            ],  # Dict is not a valid Secret
        )


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


def assert_metadata(image, expected_metadata):
    # TODO: change this to use public property workdir when/if we introduce one
    _image = synchronizer._translate_in(image)
    assert _image._metadata == expected_metadata


def test_image_from_id(builder_version, servicer, client):
    app = App()
    image = Image.debian_slim().pip_install("numpy")
    app.function(image=image)(dummy)
    dummy_metadata = api_pb2.ImageMetadata(
        workdir="/proj",
        python_packages={"fastapi": "0.100.0"},
        python_version_info="Python 3.11.8 (main, Feb 25 2024, 03:55:37) [Clang 17.0.6 ]",
    )
    with servicer.intercept() as ctx:
        with app.run(client=client):
            pass

        ctx.add_response(
            "ImageFromId",
            api_pb2.ImageFromIdResponse(
                image_id=image.object_id,
                metadata=dummy_metadata,
            ),
        )

        with app.run(client=client):
            image_from_id = Image.from_id(image.object_id, client)
            hydrate_image(image_from_id, client)
            assert image_from_id.object_id == image.object_id
            assert_metadata(image_from_id, dummy_metadata)


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


def test_from_registry_add_python(builder_version, servicer, client):
    app = App(image=Image.from_registry("ubuntu", add_python="3.9"))
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        commands = layers[0].dockerfile_commands
        assert layers[0].context_mount_id == "mo-py39"
        assert any("COPY /python/. /usr/local" in cmd for cmd in commands)
        assert any("ln -s /usr/local/bin/python3" in cmd for cmd in commands)

    if builder_version >= "2024.10":
        app = App(image=Image.from_registry("ubuntu", add_python="3.13"))
        app.function()(dummy)

        with app.run(client=client):
            layers = get_image_layers(app.image.object_id, servicer)
            commands = layers[0].dockerfile_commands
            assert layers[0].context_mount_id == "mo-py313"
            assert any("COPY /python/. /usr/local" in cmd for cmd in commands)
            assert not any("ln -s /usr/local/bin/python3" in cmd for cmd in commands)


def test_image_pip_install_pyproject(builder_version, servicer, client):
    pyproject_toml = os.path.join(os.path.dirname(__file__), "supports/test-pyproject.toml")

    app = App()
    image = Image.debian_slim().pip_install_from_pyproject(pyproject_toml)
    app.function(image=image)(dummy)
    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        assert any("pip install 'banana >=1.2.0' 'potato >=0.1.0'" in cmd for cmd in layers[0].dockerfile_commands)


def test_image_pip_install_pyproject_with_optionals(builder_version, servicer, client):
    pyproject_toml = os.path.join(os.path.dirname(__file__), "supports/test-pyproject.toml")

    app = App()
    image = Image.debian_slim().pip_install_from_pyproject(pyproject_toml, optional_dependencies=["dev", "test"])
    app.function(image=image)(dummy)
    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)

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

    app = App()
    image = Image.from_dockerfile(path)
    app.function(image=image)(dummy)

    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)

        assert any("RUN pip install numpy" in cmd for cmd in layers[1].dockerfile_commands)


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


def test_image_run_function_with_region_selection(servicer, client):
    app = App()
    app.image = Image.debian_slim().run_function(run_f, region="us-east")
    app.function()(dummy)

    with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 2
    func_def = next(iter(servicer.app_functions.values()))

    assert func_def.scheduler_placement == api_pb2.SchedulerPlacement(
        regions=["us-east"],
    )


def test_image_run_function_with_cloud_selection(servicer, client):
    app = App()
    app.image = Image.debian_slim().run_function(run_f, cloud="oci")
    app.function()(dummy)

    with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 2
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.cloud_provider == api_pb2.CLOUD_PROVIDER_UNSPECIFIED  # No longer set
    assert func_def.cloud_provider_str == "oci"


def test_poetry_files(builder_version, servicer, client):
    path = os.path.join(os.path.dirname(__file__), "supports/pyproject.toml")

    # No lockfile provided and there's no lockfile found
    image = Image.debian_slim().poetry_install_from_file(path)
    app = App()
    app.function(image=image)(dummy)
    with pytest.raises(NotFoundError):
        with app.run(client=client):
            pass

    # Explicitly ignore lockfile - this should work
    image = Image.debian_slim().poetry_install_from_file(path, ignore_lockfile=True)
    app = App()
    app.function(image=image)(dummy)
    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        context_files = {f.filename for layer in layers for f in layer.context_files}
        assert "/.pyproject.toml" in context_files
        assert "/.poetry.lock" not in context_files

    # Provide lockfile explicitly - this should also work
    lockfile_path = os.path.join(os.path.dirname(__file__), "supports/special_poetry.lock")
    image = Image.debian_slim().poetry_install_from_file(path, lockfile_path)
    app = App()
    app.function(image=image)(dummy)
    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        context_files = {f.filename for layer in layers for f in layer.context_files}
        if builder_version <= "2024.10":
            assert context_files == {"/.poetry.lock", "/.pyproject.toml", "/modal_requirements.txt"}
        else:
            assert context_files == {"/.poetry.lock", "/.pyproject.toml"}


@pytest.mark.parametrize(
    "groups, extras, frozen",
    [
        (["group1"], None, False),
        (None, ["extra1"], True),
        (["group1", "group2"], ["extra1"], True),
        (["group1"], ["extra1", "extra2"], True),
    ],
)
def test_uv_sync(builder_version, servicer, client, groups, extras, frozen):
    uv_project_path = os.path.join(os.path.dirname(__file__), "supports", "uv_lock_project")

    image = Image.debian_slim().uv_sync(uv_project_path, groups=groups, extras=extras, frozen=frozen)

    app = App()
    app.function(image=image)(dummy)

    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        context_files = {f.filename for layer in layers for f in layer.context_files}
        assert "COPY /.pyproject.toml /.uv/pyproject.toml" in layers[0].dockerfile_commands
        assert "COPY /.uv.lock /.uv/uv.lock" in layers[0].dockerfile_commands
        if frozen:
            assert any("--frozen" in cmd for cmd in layers[0].dockerfile_commands)
        if groups is not None:
            group_cmd = " ".join(f"--group={g}" for g in groups)
            assert any(group_cmd in cmd for cmd in layers[0].dockerfile_commands)
        if extras is not None:
            extra_cmd = " ".join(f"--extra={e}" for e in extras)
            assert any(extra_cmd in cmd for cmd in layers[0].dockerfile_commands)

        if builder_version <= "2024.10":
            assert context_files == {"/.uv.lock", "/.pyproject.toml", "/modal_requirements.txt"}
        else:
            assert context_files == {"/.uv.lock", "/.pyproject.toml"}


def test_uv_sync_error_invalid_kwargs(servicer, client):
    uv_project_path = os.path.join(os.path.dirname(__file__), "supports", "uv_lock_project")

    with pytest.raises(InvalidError, match="groups must be None or a list of strings"):
        Image.debian_slim().uv_sync(uv_project_path, groups="xyz")  # type: ignore

    with pytest.raises(InvalidError, match="extras must be None or a list of strings"):
        Image.debian_slim().uv_sync(uv_project_path, extras="xyz")  # type: ignore


def test_uv_sync_just_pyproject(builder_version, servicer, client):
    uv_project_path = os.path.join(os.path.dirname(__file__), "supports", "uv_sync_just_pyproject")

    image = Image.debian_slim().uv_sync(uv_project_path)

    app = App()
    app.function(image=image)(dummy)

    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        assert "COPY /.pyproject.toml /.uv/pyproject.toml" in layers[0].dockerfile_commands
        assert "COPY /.uv.lock /.uv/uv.lock" not in layers[0].dockerfile_commands
        assert not any("--frozen" in cmd for cmd in layers[0].dockerfile_commands)

        context_files = {f.filename for layer in layers for f in layer.context_files}
        if builder_version <= "2024.10":
            assert context_files == {"/.pyproject.toml", "/modal_requirements.txt"}
        else:
            assert context_files == {"/.pyproject.toml"}


def test_uv_sync_no_modal(builder_version, client):
    uv_project_path = os.path.join(os.path.dirname(__file__), "supports", "uv_lock_no_modal")

    image = Image.debian_slim().uv_sync(uv_project_path)

    app = App()
    app.function(image=image)(dummy)

    if builder_version <= "2024.10":
        with pytest.raises(InvalidError, match="requires modal to be specified"):
            with app.run(client=client):
                pass


@pytest.mark.parametrize("kwargs", [{"groups": ["group1", "group2"]}, {"extras": ["extra1"]}])
def test_uv_sync_modal_in_group_or_extra(builder_version, client, servicer, kwargs):
    uv_project_path = os.path.join(os.path.dirname(__file__), "supports", "uv_lock_no_modal")

    image = Image.debian_slim().uv_sync(uv_project_path, **kwargs)

    app = App()
    app.function(image=image)(dummy)

    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        if "groups" in kwargs:
            groups = kwargs["groups"]
            groups_cli = " ".join(f"--group={group}" for group in groups)
            assert any(groups_cli in cmd for cmd in layers[0].dockerfile_commands)
        if "extras" in kwargs:
            extras = kwargs["extras"]
            extras_cli = " ".join(f"--extra={extra}" for extra in extras)
            assert any(extras_cli in cmd for cmd in layers[0].dockerfile_commands)


def test_uv_lock_workspaces_error(builder_version, client):
    uv_project_path = os.path.join(os.path.dirname(__file__), "supports", "uv_lock_workspace")

    image = Image.debian_slim().uv_sync(uv_project_path)

    app = App()
    app.function(image=image)(dummy)

    with pytest.raises(InvalidError, match="uv workspaces are not supported"):
        with app.run(client=client):
            pass


def test_uv_sync_error(client, tmp_path):
    image = Image.debian_slim().uv_sync(tmp_path)

    app = App()
    app.function(image=image)(dummy)

    expected_pyproject_file = os.path.join(tmp_path, "pyproject.toml")
    msg = re.escape(f"Expected {expected_pyproject_file} to exist")
    with pytest.raises(InvalidError, match=msg):
        with app.run(client=client):
            pass


@pytest.mark.parametrize("poetry_version", [None, "latest", "2.1.2"])
def test_poetry_commands(builder_version, servicer, client, poetry_version):
    path = os.path.join(os.path.dirname(__file__), "supports/pyproject.toml")
    app = App()
    image = Image.debian_slim().poetry_install_from_file(
        path,
        ignore_lockfile=True,
        with_=["foo", "bar"],
        without=["buz"],
        poetry_version=poetry_version,
    )
    app.function(image=image)(dummy)
    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        dockerfile_commands = " ".join(layers[0].dockerfile_commands)
        if poetry_version is None:
            assert not any("pip install" in cmd for cmd in dockerfile_commands)
        else:
            if builder_version <= "2024.10":
                poetry_spec = "~=1.7" if poetry_version == "latest" else f"=={poetry_version}"
            else:
                poetry_spec = "" if poetry_version == "latest" else f"=={poetry_version}"
            assert f"python -m pip install poetry{poetry_spec}" in dockerfile_commands
        assert "poetry config virtualenvs.create false" in dockerfile_commands
        assert "poetry install --no-root --with foo,bar --without buz" in dockerfile_commands


def test_image_add_local_file_error(tmp_path, client):
    app = App()

    unknown_file = tmp_path / "does-not-exist.txt"
    img = Image.debian_slim().add_local_file(unknown_file, "/file.txt")
    app.function(image=img)(dummy)

    msg = re.escape(f"local file {unknown_file} does not exist")
    with pytest.raises(FileNotFoundError, match=msg):
        with app.run(client=client):
            pass


def test_image_add_local_dir_err(tmp_path, client):
    app = App()

    unknown_dir = tmp_path / "does-not-exist"
    img = Image.debian_slim().add_local_dir(unknown_dir, "/file.txt")
    app.function(image=img)(dummy)

    msg = re.escape(f"local dir {unknown_dir} does not exist")
    with pytest.raises(FileNotFoundError, match=msg):
        with app.run(client=client):
            pass


@pytest.mark.parametrize(["copy"], [(True,), (False,)])
@pytest.mark.parametrize(
    ["remote_path", "expected_dest"],
    [
        ("/place/nice.txt", "/place/nice.txt"),
        # Not supported yet, but soon:
        ("/place/", "/place/data.txt"),  # use original basename if destination has a trailing slash
        # ("output.txt", "/proj/output.txt")  # workdir relative target
        # (None, "/proj/data.txt")  # default target - basename in current directory
    ],
)
def test_image_add_local_file(servicer, client, tmp_path_with_content, copy, remote_path, expected_dest):
    app = App()

    if remote_path is None:
        remote_path_kwargs = {}
    else:
        remote_path_kwargs = {"remote_path": remote_path}

    img = (
        Image.from_registry("unknown_image")
        .workdir("/proj")
        .add_local_file(tmp_path_with_content / "data.txt", **remote_path_kwargs, copy=copy)
    )
    app.function(image=img)(dummy)

    with app.run(client=client):
        if copy:
            # check that dockerfile commands include COPY . .
            layers = get_image_layers(img.object_id, servicer)
            assert layers[0].dockerfile_commands == ["FROM base", "COPY . /"]
            mount_id = layers[0].context_mount_id
            # and then get the relevant context mount to check
        if not copy:
            assert len(img._mount_layers) == 1
            mount_id = img._mount_layers[0].object_id

        assert set(servicer.mount_contents[mount_id].keys()) == {expected_dest}


@pytest.mark.parametrize(["copy"], [(True,), (False,)])
@pytest.mark.parametrize(
    ["remote_path", "expected_dest"],
    [
        ("/place/", "/place/sub"),  # copy full dir
        ("/place", "/place/sub"),  # removing trailing slash on source makes no difference, unlike shell cp
        # TODO: add support for relative paths:
        # Not supported yet, but soon:
        # ("place", "/proj/place/sub")  # workdir relative target
        # (None, "/proj/sub")  # default target - copy into current directory
    ],
)
def test_image_add_local_dir(servicer, client, tmp_path_with_content, copy, remote_path, expected_dest):
    app = App()

    if remote_path is None:
        remote_path_kwargs = {}
    else:
        remote_path_kwargs = {"remote_path": remote_path}

    img = (
        Image.from_registry("unknown_image")
        .workdir("/proj")
        .add_local_dir(tmp_path_with_content / "data", **remote_path_kwargs, copy=copy)
    )
    app.function(image=img)(dummy)

    with app.run(client=client):
        if copy:
            # check that dockerfile commands include COPY . .
            layers = get_image_layers(img.object_id, servicer)
            assert layers[0].dockerfile_commands == ["FROM base", "COPY . /"]
            mount_id = layers[0].context_mount_id
            # and then get the relevant context mount to check
        if not copy:
            assert len(img._mount_layers) == 1
            mount_id = img._mount_layers[0].object_id

        assert set(servicer.mount_contents[mount_id].keys()) == {expected_dest}


@pytest.mark.usefixtures("tmp_cwd")
def test_image_docker_command_no_copy_commands(builder_version, servicer, client):
    # make sure that no mount is created if there are no copy commands
    Path("data.txt").write_text("hello")
    app = App()
    image = Image.debian_slim().dockerfile_commands(["RUN something"])
    app.function(image=image, serialized=True)(dummy)

    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        assert "RUN something" in layers[0].dockerfile_commands
        context_mount_id = layers[0].context_mount_id
        assert not context_mount_id  # there should be no mount
        assert not servicer.mounts_excluding_published_client()


@pytest.fixture()
def dummy_context_dir(tmp_cwd):
    # create dummy data in current working directory
    rel_top_dir = Path("top")
    rel_top_dir.mkdir()
    (rel_top_dir / "data").mkdir()
    (rel_top_dir / "data" / "sub").write_text("world")
    (rel_top_dir / "module").mkdir()
    (rel_top_dir / "module" / "__init__.py").write_text("foo")
    (rel_top_dir / "module" / "sub.py").write_text("bar")
    (rel_top_dir / "module" / "sub").mkdir()
    (rel_top_dir / "module" / "sub" / "__init__.py").write_text("baz")
    yield
    shutil.rmtree(rel_top_dir)


ALL_DUMMY_FILES = {"/top/data/sub", "/top/module/__init__.py", "/top/module/sub.py", "/top/module/sub/__init__.py"}


@pytest.mark.usefixtures("tmp_cwd", "dummy_context_dir")
@pytest.mark.parametrize(
    ("docker_commands", "expected_mount_files"),
    [
        (["COPY . /"], ALL_DUMMY_FILES),  # common
        (["COPY top /"], ALL_DUMMY_FILES),  # single dir
        (["COPY top/ /"], ALL_DUMMY_FILES),  # trailing slash
        # changing the destination directory shouldn't matter for how the *context mount* looks:
        (["COPY top /dummy"], ALL_DUMMY_FILES),
        (["COPY top/data /"], {"/top/data/sub"}),  # from subdir
        (["COPY top/data/sub /"], {"/top/data/sub"}),  # individual file
        (["COPY ./** /"], ALL_DUMMY_FILES),  # glob
        (
            ["COPY top/data/sub /", "COPY top/module/sub.py /"],
            {"/top/data/sub", "/top/module/sub.py"},
        ),  # multiple copy statements
    ],
)
@pytest.mark.parametrize("use_dockerfile", [False, True])
def test_image_docker_command_copy(
    builder_version, servicer, client, docker_commands, expected_mount_files, use_dockerfile
):
    app = App()
    maybe_dockerfile = None
    if use_dockerfile:
        # auto deleting files appears to break windows ci
        maybe_dockerfile = NamedTemporaryFile("w", delete=False)
        maybe_dockerfile.write("\n".join(docker_commands))
        maybe_dockerfile.close()
        image = Image.from_dockerfile(maybe_dockerfile.name)
        copy_layer_index = 1  # layer 0 is the base image
    else:
        image = Image.debian_slim().dockerfile_commands(*docker_commands)
        copy_layer_index = 0

    app.function(image=image)(dummy)

    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        mount_id = layers[copy_layer_index].context_mount_id
        files = set(servicer.mount_contents[mount_id].keys())
        assert files == expected_mount_files

    if maybe_dockerfile:
        Path(maybe_dockerfile.name).unlink()


@pytest.mark.parametrize("use_dockerfile", (True, False))
@pytest.mark.usefixtures("tmp_cwd")
def test_image_dockerfile_copy_auto_dockerignore(builder_version, servicer, client, use_dockerfile):
    rel_top_dir = Path("top")
    rel_top_dir.mkdir()
    (rel_top_dir / "data.txt").write_text("world")
    (rel_top_dir / "file.py").write_text("world")
    dockerfile = rel_top_dir / "Dockerfile"
    dockerfile.write_text(f"COPY {rel_top_dir} /dummy\n")

    # automatically find the .dockerignore file
    dockerignore = Path(".") / ".dockerignore"
    dockerignore.write_text("**/*.txt")

    app = App()
    if use_dockerfile:
        image = Image.debian_slim().from_dockerfile(dockerfile)
        layer = 1
    else:
        image = Image.debian_slim().dockerfile_commands([f"COPY {rel_top_dir} /dummy"])
        layer = 0
    app.function(image=image)(dummy)

    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        assert f"COPY {rel_top_dir} /dummy" in layers[layer].dockerfile_commands
        mount_id = layers[layer].context_mount_id
        files = set(Path(fn) for fn in servicer.mount_contents[mount_id].keys())
        assert files == {Path("/") / rel_top_dir / "Dockerfile", Path("/") / rel_top_dir / "file.py"}


@pytest.mark.parametrize("use_dockerfile", (True, False))
@pytest.mark.usefixtures("tmp_cwd")
def test_image_dockerfile_copy_ignore_from_file(builder_version, servicer, client, use_dockerfile):
    rel_top_dir = Path("top")
    rel_top_dir.mkdir()
    (rel_top_dir / "data.txt").write_text("world")
    (rel_top_dir / "file.py").write_text("world")
    dockerfile = rel_top_dir / "Dockerfile"
    dockerfile.write_text(f"COPY {rel_top_dir} /dummy\n")

    # explicitly provide the .dockerignore file
    dockerignore = rel_top_dir / "something_special.dockerignore"
    dockerignore.write_text("**/*.py")

    app = App()
    if use_dockerfile:
        image = Image.debian_slim().from_dockerfile(dockerfile, ignore=FilePatternMatcher.from_file(dockerignore))
        layer = 1
    else:
        image = Image.debian_slim().dockerfile_commands(
            [f"COPY {rel_top_dir} /dummy"], ignore=FilePatternMatcher.from_file(dockerignore)
        )
        layer = 0
    app.function(image=image)(dummy)

    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        assert f"COPY {rel_top_dir} /dummy" in layers[layer].dockerfile_commands
        mount_id = layers[layer].context_mount_id
        files = set(Path(fn) for fn in servicer.mount_contents[mount_id].keys())
        assert files == {
            Path("/") / rel_top_dir / "Dockerfile",
            Path("/") / rel_top_dir / "data.txt",
            Path("/") / rel_top_dir / "something_special.dockerignore",
        }


@pytest.mark.parametrize("use_dockerfile", (True, False))
@pytest.mark.usefixtures("tmp_cwd")
def test_image_dockerfile_ignore_context_dir(builder_version, servicer, client, use_dockerfile):
    rel_top_dir = Path("top")
    rel_top_dir.mkdir()
    (rel_top_dir / "data.txt").write_text("world")
    (rel_top_dir / "file.py").write_text("world")

    sub_dir = rel_top_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "notes.txt").write_text("whatever")

    docker_cmd = "COPY . /dummy"
    dockerfile = rel_top_dir / "Dockerfile"
    dockerfile.write_text(docker_cmd + "\n")

    dockerignore = rel_top_dir / ".dockerignore"
    dockerignore.write_text("*.txt")

    app = App()
    if use_dockerfile:
        image = Image.debian_slim().from_dockerfile(dockerfile, context_dir=rel_top_dir)
        layer = 1
    else:
        image = Image.debian_slim().dockerfile_commands([docker_cmd], context_dir=rel_top_dir)
        layer = 0
    app.function(image=image)(dummy)

    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        assert docker_cmd in layers[layer].dockerfile_commands
        mount_id = layers[layer].context_mount_id
        files = set(Path(fn) for fn in servicer.mount_contents[mount_id].keys())
        assert files == {
            Path("/") / "Dockerfile",
            Path("/") / ".dockerignore",
            Path("/") / "file.py",
            Path("/") / "sub" / "notes.txt",
        }


@pytest.mark.parametrize("use_callable", (True, False))
@pytest.mark.parametrize("use_dockerfile", (True, False))
@pytest.mark.usefixtures("tmp_cwd")
def test_image_dockerfile_copy_ignore(builder_version, servicer, client, use_callable, use_dockerfile):
    # TODO: combine with test_image_docker_command_copy?
    def cb(x: Path):
        return x.suffix == ".txt"

    ignore: Union[Sequence[str], Callable[[Path], bool]] = cb if use_callable else ["**/*.txt"]

    rel_top_dir = Path("top")
    rel_top_dir.mkdir()
    (rel_top_dir / "data.txt").write_text("world")
    (rel_top_dir / "file.py").write_text("world")
    dockerfile = rel_top_dir / "Dockerfile"
    dockerfile.write_text(f"COPY {rel_top_dir} /dummy\n")

    app = App()
    if use_dockerfile:
        image = Image.debian_slim().from_dockerfile(dockerfile, ignore=ignore)
        layer = 1
    else:
        image = Image.debian_slim().dockerfile_commands([f"COPY {rel_top_dir} /dummy"], ignore=ignore)
        layer = 0
    app.function(image=image)(dummy)

    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        assert f"COPY {rel_top_dir} /dummy" in layers[layer].dockerfile_commands
        mount_id = layers[layer].context_mount_id
        files = set(Path(fn) for fn in servicer.mount_contents[mount_id].keys())
        assert files == {Path("/") / rel_top_dir / "Dockerfile", Path("/") / rel_top_dir / "file.py"}


@pytest.mark.usefixtures("tmp_cwd")
def test_dockerfile_context_dir(builder_version, servicer, client):
    # Write a file to a different temporary directory
    with TemporaryDirectory() as context_dir:
        (Path(context_dir) / "data.txt").write_text("hello")
        (Path(context_dir) / "file.py").write_text("world")

        image = Image.debian_slim().dockerfile_commands(["COPY . /"], context_dir=context_dir)
        app = App(include_source=False)
        app.function(image=image)(dummy)
        with app.run(client=client):
            layers = get_image_layers(image.object_id, servicer)
            mount_id = layers[0].context_mount_id
            files = set(Path(fn) for fn in servicer.mount_contents[mount_id].keys())
            assert files == {Path("/data.txt"), Path("/file.py")}


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


def test_image_docker_command_entrypoint_malformed():
    with pytest.raises(InvalidError, match="must be a list of strings"):
        Image.debian_slim().entrypoint("this is not a list of strings")  # type: ignore
    with pytest.raises(InvalidError, match="must be a list of strings"):
        Image.debian_slim().entrypoint(4711)  # type: ignore


def test_image_docker_command_shell(builder_version, servicer, client, tmp_path_with_content):
    app = App()
    app.image = Image.debian_slim().shell(["/bin/sh", "-c"])
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert 'SHELL ["/bin/sh", "-c"]' in layers[0].dockerfile_commands


def test_image_docker_command_shell_malformed():
    with pytest.raises(InvalidError, match="must be a list of strings"):
        Image.debian_slim().shell("this is not a list of strings")  # type: ignore
    with pytest.raises(InvalidError, match="must be a list of strings"):
        Image.debian_slim().shell(4711)  # type: ignore


def test_image_env(builder_version, servicer, client):
    app = App(image=Image.debian_slim().env({"HELLO": "world!"}))
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert any("ENV HELLO=" in cmd and "world!" in cmd for cmd in layers[0].dockerfile_commands)

    # unhappy path, reject invalid input
    with pytest.raises(InvalidError, match="Image ENV variables must be strings."):
        app = App(image=Image.debian_slim().env({"HELLO": 123}))  # type: ignore
        app.function()(dummy)
        with app.run(client=client):
            pass


def test_image_cmd_empty(builder_version, servicer, client, tmp_path_with_content):
    app = App()
    app.image = Image.debian_slim().cmd([])
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert "CMD []" in layers[0].dockerfile_commands


def test_image_cmd_nonempty(builder_version, servicer, client, tmp_path_with_content):
    app = App()
    app.image = Image.debian_slim().cmd(["echo", "hello"])
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert 'CMD ["echo", "hello"]' in layers[0].dockerfile_commands


def test_image_cmd_malformed():
    with pytest.raises(InvalidError, match="must be a list of strings"):
        Image.debian_slim().cmd("this is not a list of strings")  # type: ignore
    with pytest.raises(InvalidError, match="must be a list of strings"):
        Image.debian_slim().cmd(4711)  # type: ignore


def test_image_debian_slim_default_cmd(builder_version, servicer, client, test_dir):
    app = App(image=Image.debian_slim())
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        if builder_version > "2024.10":
            assert 'CMD ["sleep", "172800"]' in layers[0].dockerfile_commands
        else:
            assert "CMD" not in layers[0].dockerfile_commands


def test_image_micromamba_default_cmd(builder_version, servicer, client, test_dir):
    app = App(image=Image.micromamba())
    app.function()(dummy)

    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        if builder_version > "2024.10":
            assert 'CMD ["sleep", "172800"]' in layers[0].dockerfile_commands
        else:
            assert "CMD" not in layers[0].dockerfile_commands


def test_image_gpu(builder_version, servicer, client):
    app = App(image=Image.debian_slim().run_commands("echo 0"))
    app.function()(dummy)
    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert not layers[0].gpu_config.gpu_type
        assert not layers[0].gpu_config.count

    app = App(image=Image.debian_slim().run_commands("echo 1", gpu="any"))
    app.function()(dummy)
    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert layers[0].gpu_config.gpu_type == "ANY"
        assert layers[0].gpu_config.count == 1

    app = App(image=Image.debian_slim().run_commands("echo 2", gpu="a10g"))
    app.function()(dummy)
    with app.run(client=client):
        layers = get_image_layers(app.image.object_id, servicer)
        assert layers[0].gpu_config.gpu_type == "A10G"
        assert layers[0].gpu_config.count == 1


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


def test_hydration_metadata(servicer, client):
    img = Image.debian_slim()
    app = App(image=img)
    app.function()(dummy)
    dummy_metadata = api_pb2.ImageMetadata(
        workdir="/proj",
        python_packages={"fastapi": "0.100.0"},
        python_version_info="Python 3.11.8 (main, Feb 25 2024, 03:55:37) [Clang 17.0.6 ]",
    )
    with servicer.intercept() as ctx:
        ctx.add_response(
            "ImageJoinStreaming",
            api_pb2.ImageJoinStreamingResponse(
                result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS),
                metadata=dummy_metadata,
            ),
        )

        with app.run(client=client):
            assert_metadata(img, dummy_metadata)


cls_app = App()

VARIABLE_5 = 1
VARIABLE_6 = 1


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


def test_image_builder_version(servicer, credentials, test_dir, modal_config):
    app = App(image=Image.debian_slim())
    app.function()(dummy)

    def mock_base_image_config(group, version):
        config = {
            "debian": "bookworm",
            "python": "3.11.0",
            "package_tools": "pip wheel uv",
        }
        return config[group]

    test_requirements = str(test_dir / "supports" / "test-requirements.txt")
    with (
        mock.patch("modal.image._get_modal_requirements_path", lambda *_, **__: test_requirements),
        mock.patch("modal.image._dockerhub_python_version", lambda *_, **__: "3.11.0"),
        mock.patch("modal.image._base_image_config", mock_base_image_config),
        mock.patch("test.conftest.ImageBuilderVersion", Literal["2000.01"]),
        mock.patch("modal.image.ImageBuilderVersion", Literal["2000.01"]),
        Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, credentials) as client,
        modal_config(),
        app.run(client=client),
    ):
        assert servicer.image_builder_versions
        for version in servicer.image_builder_versions.values():
            assert version == "2000.01"


def test_image_builder_supported_versions(servicer, credentials):
    app = App(image=Image.debian_slim())
    app.function()(dummy)

    with (
        pytest.raises(VersionError, match=r"This version of the modal client supports.+{'2000.01'}"),
        mock.patch("modal.image.ImageBuilderVersion", Literal["2000.01"]),
        mock.patch("test.conftest.ImageBuilderVersion", Literal["2023.11"]),
        Client(servicer.client_addr, api_pb2.CLIENT_TYPE_CLIENT, credentials) as client,
        app.run(client=client),
    ):
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
            layers = get_image_layers(img.object_id, servicer)
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

        img = Image.micromamba()
        assert get_hash(img) == "fa883741544ea191ecd197c8f83a1ffe9912575faa8c107c66b3dda761b2e401"

        img = Image.from_dockerfile(test_dir / "supports" / "test-dockerfile")
        assert get_hash(img) == "0aec2f66f28ee7511c1b36604214ae7b40d9bc1fa3e6b8883001e933a966ff78"

    img = Image.micromamba(python_version="3.12")
    assert get_hash(img) == "468befe16f703a3ae1a794dfe54c1a3445ca0ffda233f55f1d66c45ad608e8aa"

    base = Image.debian_slim(python_version="3.12")

    img = base.run_commands("echo 'Hello Modal'", "rm /usr/local/bin/kubectl")
    assert get_hash(img) == "4e1ac62eb33b44dd16940c9d2719eb79f945cee61cbf4641ca99b19cd9e0976d"

    img = base.pip_install("torch~=2.2", "transformers==4.23.0", pre=True, index_url="agi.se")
    assert get_hash(img) == "2a4fa8e3b32c70a41b3a3efd5416540b1953430543f6c27c984e7f969c2ca874"

    img = base.pip_install_from_requirements(test_dir / "supports" / "test-requirements.txt")
    assert get_hash(img) == "69d41e699d4ecef399e51e8460f8857aa0ec57f71f00eca81c8886ec062e5c2b"

    img = base.poetry_install_from_file(
        test_dir / "supports" / "test-pyproject.toml",
        poetry_lockfile=test_dir / "supports" / "special_poetry.lock",
    )
    assert get_hash(img) == "a25dd4cc2e8d88f92bfdaf2e82b9d74144d1928926bf6be2ca1cdfbbf562189e"

    img = base.uv_pip_install("torch~=2.2", "transformers==4.23.0", pre=True, index_url="agi.se")
    assert get_hash(img) == "4255e224cc4b05589efdfe227a06659ca2dc64b053f43e57808d04e34da94738"

    img = base.uv_pip_install(requirements=[test_dir / "supports" / "test-requirements.txt"])
    assert get_hash(img) == "fb1ea2563fad89d3e6d00abf1c91a3835b6ba897693a61cdfe29dc1e5ea01c33"

    img = base.uv_sync(test_dir / "supports" / "uv_lock_project")
    assert get_hash(img) == "6c9f3debe511508a99ec70212ff79dcfc01ec95be9400c63edfb36c9035be9de"


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

    img = base.pip_install_from_requirements(test_dir / "supports" / "test-requirements.txt")
    assert get_hash(img) == "78392aca4ea135ab53b9d183eedbb2a7e32f9b3c0cfb42b03a7bd7c4f013f3c8"

    img = base.micromamba_install(
        "torch=2.2",
        "transformers<4.23.0",
        spec_file=test_dir / "supports" / "test-conda-environment.yml",
        channels=["conda-forge", "my-channel"],
    )
    assert get_hash(img) == "f8701ce500d6aa1fecefd9c2869aef4a13c77ab03925333c011d7eca60bbf08a"

    img = base.poetry_install_from_file(
        test_dir / "supports" / "test-pyproject.toml",
        poetry_lockfile=test_dir / "supports" / "special_poetry.lock",
    )
    assert get_hash(img) == "bfce5811c04c1243f12cbb9cca1522cb901f52410986925bcfa3b3c2d7adc7a0"

    img = base.uv_pip_install("torch~=2.2", "transformers==4.23.0", pre=True, index_url="agi.se")
    assert get_hash(img) == "f96d8a1dac1cb5a65f5b83a7f63cc60dce671ee08749f7c12ec9dc13743da0e9"

    img = base.uv_pip_install(requirements=[test_dir / "supports" / "test-requirements.txt"])
    assert get_hash(img) == "1b45416b4a1563a0522047e1faa124c8132fc068720299445c3d7b6204b696f9"

    img = base.uv_sync(test_dir / "supports" / "uv_lock_project")
    assert get_hash(img) == "925054c1aed3a194de979389eeacb5a842695316f7a9f889e314de5c51a62760"


@pytest.fixture
def force_2024_10(modal_config):
    with mock.patch("test.conftest.ImageBuilderVersion", Literal["2024.10"]):
        with modal_config():
            yield


@skip_windows("Different hash values for context file paths")
def test_image_stability_on_2024_10(force_2024_10, servicer, client, test_dir):
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
        assert get_hash(img) == "f03d3a2bd1a859349320b216311902982aebad30f135b1bef68e3c6cc8a6bfbe"

    img = Image.debian_slim(python_version="3.12")
    assert get_hash(img) == "385413df75dffe57f41d4c8eef45ce6cec6de54dda348f3b93c3b7d36fdf0973"

    img = Image.from_registry("ubuntu:22.04")
    assert get_hash(img) == "aa4b17f73658c5ae09ca8dfce9419cd50c6179ecf015208151cc2c7109ed8e40"

    img = Image.from_dockerfile(test_dir / "supports" / "test-dockerfile")
    assert get_hash(img) == "8997493d8ff7b8d25fc5c1943626d262afacc64f14ad91edbdc4536600528e3d"

    img = Image.micromamba()
    if sys.version_info[:2] == (3, 11):
        assert get_hash(img) == "4a0241417a2e67d995cce36c0ee4907ec249b008e05658eb98ce0a655e7e9861"
    elif sys.version_info[:2] == (3, 10):
        # Assert that we follow the local Python, which is a new behavior in 2024.04
        assert get_hash(img) == "e9d42609633d0e24822bcb77d0ca4de5fc706df1e38a8eebe1b322fb1964dafe"

    img = Image.micromamba(python_version="3.12")
    assert get_hash(img) == "bcccccc3dda15c813f73be58514aaadfe270a0e9a0ecb1817dea630bdb31e357"

    base = Image.debian_slim(python_version="3.12")

    img = base.run_commands("echo 'Hello Modal'", "rm /usr/local/bin/kubectl")
    assert get_hash(img) == "bee3ae0a37ea24925e616d5a90bc7564848b15fac85a3d8d12c2be88038f3011"

    img = base.pip_install("torch~=2.2", "transformers==4.23.0", pre=True, index_url="agi.se")
    assert get_hash(img) == "b3f271b826547d5c2a62a58765e7456ad957c982c2dd2fd0a1d6d472b4a2e928"

    img = base.pip_install_from_requirements(test_dir / "supports" / "test-requirements.txt")
    assert get_hash(img) == "94960315ff71d9521890a0472ea389c4ef3be76b9b439a6bcb28d7e17a4ee7ea"

    img = base.micromamba_install(
        "torch=2.2",
        "transformers<4.23.0",
        spec_file=test_dir / "supports" / "test-conda-environment.yml",
        channels=["conda-forge", "my-channel"],
    )
    assert get_hash(img) == "072e70b2f05327f606c261ad48a68cf8db5e592e7019f6ee7dbaccf28f2ef537"

    img = base.poetry_install_from_file(
        test_dir / "supports" / "test-pyproject.toml",
        poetry_lockfile=test_dir / "supports" / "special_poetry.lock",
    )
    assert get_hash(img) == "78d579f243c21dcaa59e5daf97f732e2453b004bc2122de692617d4d725c6184"

    img = base.uv_pip_install("torch~=2.2", "transformers==4.23.0", pre=True, index_url="agi.se")
    assert get_hash(img) == "e4927f14fcd09e1d3b8a73a17a25392e1988ee4232e5130ea8fddce80b49f5e1"

    img = base.uv_pip_install(requirements=[test_dir / "supports" / "test-requirements.txt"])
    assert get_hash(img) == "166413c4c2a08decd03092a328aca9b34f85b5e77eb50a584aebd8f754523412"

    img = base.uv_sync(test_dir / "supports" / "uv_lock_project")
    assert get_hash(img) == "2b6cd5b524ac796cafdabe8b95bf626a765f28c909d23f6051fc4329d6edbc0b"


@pytest.fixture
def force_2025_06(modal_config):
    with mock.patch("test.conftest.ImageBuilderVersion", Literal["2025.06"]):
        with modal_config():
            yield


@skip_windows("Different hash values for context file paths")
def test_image_stability_on_2025_06(force_2025_06, servicer, client, test_dir):
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
        assert get_hash(img) == "691f48fbf5211d4d7670962d47da3029c056868df50bb66e01f577badd9dcdfb"

    img = Image.debian_slim(python_version="3.12")
    assert get_hash(img) == "8f5fe3378cd1c4311f9d092776ace022b57745a924843b7b3b394326f6d742ae"

    img = Image.from_registry("ubuntu:22.04")
    assert get_hash(img) == "d13118a015a55282d62b7c93bb20611ea79f8320735c72df57966b5c298b1c3e"

    img = Image.from_dockerfile(test_dir / "supports" / "test-dockerfile")
    assert get_hash(img) == "2cc97f48ac2699c80d7c21363a99e23b4c60e2c3a876d53742b60cb3957fc650"

    img = Image.micromamba()
    if sys.version_info[:2] == (3, 11):
        assert get_hash(img) == "7257e8b6593c75a80301bb899f9c88da157a28ba3f5265b92b144960bd1819e1"
    elif sys.version_info[:2] == (3, 10):
        # Assert that we follow the local Python, which is a new behavior in 2024.04
        assert get_hash(img) == "6522a2426f2f5929eeac50a43141fa1b5fe510b6b66238ad8c0e8c41c9fae4fd"

    img = Image.micromamba(python_version="3.12")
    assert get_hash(img) == "c368b09c8dea84c5a26f97fc0028a52c46ca3a97caf070ceade36d44471547ad"

    base = Image.debian_slim(python_version="3.12")

    img = base.run_commands("echo 'Hello Modal'", "rm /usr/local/bin/kubectl")
    assert get_hash(img) == "9403a725e8b7693b2aec476683e4a4fd75e82e9ca37323948e5d2ffbe7551b3b"

    img = base.pip_install("torch~=2.2", "transformers==4.23.0", pre=True, index_url="agi.se")
    assert get_hash(img) == "ad6696a7ad6e67a48c5cc613f5682bc20e36803e2e22bb9c9e65f93e8af02e14"

    img = base.pip_install_from_requirements(test_dir / "supports" / "test-requirements.txt")
    assert get_hash(img) == "c99a54ef791846d761a93c06f65d2f2a25e088f0790aecaa12e8e702422935a5"

    img = base.micromamba_install(
        "torch=2.2",
        "transformers<4.23.0",
        spec_file=test_dir / "supports" / "test-conda-environment.yml",
        channels=["conda-forge", "my-channel"],
    )
    assert get_hash(img) == "30b1a4375ccd85314366b51cbec09691cbd7222979cefe1d3d6dc01413c54b9a"

    img = base.poetry_install_from_file(
        test_dir / "supports" / "test-pyproject.toml",
        poetry_lockfile=test_dir / "supports" / "special_poetry.lock",
    )
    assert get_hash(img) == "0b1dabc063d6525a0cf21abe5c72f3ac4bffd6d49067ba6e0e7172537963c02b"

    img = base.uv_pip_install("torch~=2.2", "transformers==4.23.0", pre=True, index_url="agi.se")
    assert get_hash(img) == "718e540692f813ee6f71623636ef813648ffff2515e4dfe312abd602ea5bf122"

    img = base.uv_pip_install(requirements=[test_dir / "supports" / "test-requirements.txt"])
    assert get_hash(img) == "1ceaf730b03f1b068e5c66f0fb701ce932eaa555cbc91fa1823c1ed7cc8bc687"

    img = base.uv_sync(test_dir / "supports" / "uv_lock_project")
    assert get_hash(img) == "60304d4a13826e0a450248cb37bc0192020b83dd20bcb2dbea058e0149d54532"


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
                result=api_pb2.GenericResult(
                    status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS,
                ),
                metadata=api_pb2.ImageMetadata(image_builder_version="foobar"),
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


def hydrate_image(img, client):
    # there should be a more straight forward way to do this?
    app = App()
    app.function(serialized=True, image=img)(lambda: None)
    with app.run(client=client):
        pass


def test_add_local_lazy_vs_copy(client, servicer, set_env_client, supports_on_path):
    deb = Image.debian_slim()
    image_with_mount = deb.add_local_python_source("pkg_a")

    hydrate_image(image_with_mount, client)
    assert image_with_mount.object_id == deb.object_id
    assert len(image_with_mount._mount_layers) == 1

    image_additional_mount = image_with_mount.add_local_python_source("pkg_b")
    hydrate_image(image_additional_mount, client)
    assert len(image_additional_mount._mount_layers) == 2  # another mount added to lazy layer
    assert len(image_with_mount._mount_layers) == 1  # original image should not be affected

    # running commands
    image_non_mount = image_with_mount.run_commands("echo 'hello'")
    with pytest.raises(InvalidError, match="copy=True"):
        # error about using non-copy add commands before other build steps
        hydrate_image(image_non_mount, client)

    image_with_copy = deb.add_local_python_source("pkg_a", copy=True)
    hydrate_image(image_with_copy, client)
    assert len(image_with_copy._mount_layers) == 0

    # do the same exact image using copy=True
    image_with_copy_and_commands = deb.add_local_python_source("pkg_a", copy=True).run_commands("echo 'hello'")
    hydrate_image(image_with_copy_and_commands, client)
    assert len(image_with_copy_and_commands._mount_layers) == 0

    layers = get_image_layers(image_with_copy_and_commands.object_id, servicer)

    echo_layer = layers[0]
    assert echo_layer.dockerfile_commands == ["FROM base", "RUN echo 'hello'"]

    copy_layer = layers[1]
    assert copy_layer.dockerfile_commands == ["FROM base", "COPY . /"]
    copied_files = servicer.mount_contents[copy_layer.context_mount_id].keys()
    assert len(copied_files) == 8
    assert all(fn.startswith("/root/pkg_a/") for fn in copied_files)


def test_add_locals_are_attached_to_functions(servicer, client, supports_on_path):
    deb_slim = Image.debian_slim()
    img = deb_slim.add_local_python_source("pkg_a")
    app = App("my-app")
    control_fun: modal.Function = app.function(serialized=True, image=deb_slim, name="control")(
        dummy
    )  # no mounts on image
    fun: modal.Function = app.function(serialized=True, image=img, name="fun")(dummy)  # mounts on image
    deploy_app(app, client=client)

    control_func_mounts = set(servicer.app_functions[control_fun.object_id].mount_ids)
    fun_def = servicer.app_functions[fun.object_id]
    added_mounts = set(fun_def.mount_ids) - control_func_mounts
    assert len(added_mounts) == 1
    assert added_mounts == {img._mount_layers[0].object_id}


def test_add_locals_are_attached_to_classes(servicer, client, supports_on_path, set_env_client):
    deb_slim = Image.debian_slim()
    img = deb_slim.add_local_python_source("pkg_a")
    app = App("my-app")
    control_fun: modal.Function = app.function(serialized=True, image=deb_slim, name="control")(
        dummy
    )  # no mounts on image

    class A:
        some_arg: str = modal.parameter()

    ACls = app.cls(serialized=True, image=img)(A)  # mounts on image
    deploy_app(app, client=client)

    control_func_mounts = set(servicer.app_functions[control_fun.object_id].mount_ids)
    fun_def = servicer.function_by_name("A.*")  # class service function
    added_mounts = set(fun_def.mount_ids) - control_func_mounts
    assert len(added_mounts) == 1
    assert added_mounts == {img._mount_layers[0].object_id}

    obj = ACls(some_arg="foo")  # type: ignore
    # hacky way to force hydration of the *parameter bound* function (instance service function):
    obj.update_autoscaler(min_containers=0)  #  type: ignore

    obj_fun_def = servicer.function_by_name("A.*", ((), {"some_arg": "foo"}))  # instance service function
    added_mounts = set(obj_fun_def.mount_ids) - control_func_mounts
    assert len(added_mounts) == 1
    assert added_mounts == {img._mount_layers[0].object_id}


def test_from_local_python_packages_missing_module(servicer, client, test_dir, server_url_env):
    app = App()
    app.function(image=Image.debian_slim().add_local_python_source("nonexistent_package"))(dummy)

    with pytest.raises(ModuleNotMountable):
        with app.run(client=client):
            pass


def test_from_local_python_packages_wrong_type():
    with pytest.raises(InvalidError, match="specified as strings"):
        Image.debian_slim().add_local_python_source(os, sys)  # type: ignore


@skip_windows("servicer sandbox implementation not working on windows")
def test_add_locals_are_attached_to_sandboxes(servicer, client, supports_on_path):
    deb_slim = Image.debian_slim()
    img = deb_slim.add_local_python_source("pkg_a")
    app = App("my-app")
    with app.run(client=client):
        modal.Sandbox.create(image=img, app=app, client=client)
        sandbox_def = servicer.sandbox_defs[0]

    assert sandbox_def.image_id == deb_slim.object_id
    assert sandbox_def.mount_ids == [img._mount_layers[0].object_id]
    copied_files = servicer.mount_contents[sandbox_def.mount_ids[0]]
    assert len(copied_files) == 8
    assert all(fn.startswith("/root/pkg_a/") for fn in copied_files)


def empty_fun():
    pass


def test_add_locals_build_function(servicer, client, supports_on_path):
    deb_slim = Image.debian_slim()
    img = deb_slim.add_local_python_source("pkg_a")
    img_with_build_function = img.run_function(empty_fun)
    with pytest.raises(InvalidError):
        # build functions could still potentially rewrite mount contents,
        # so we still require them to use copy=True
        # TODO(elias): what if someone wants do use an equivalent of `run_function(..., mounts=[...]) ?
        hydrate_image(img_with_build_function, client)

    img_with_copy = deb_slim.add_local_python_source("pkg_a", copy=True)
    hydrate_image(img_with_copy, client)  # this is fine


# TODO: test modal shell w/ lazy mounts
# this works since the image is passed on as is to a sandbox which will load it and
# transfer any virtual mount layers from the image as mounts to the sandbox


def test_image_only_joins_unfinished_steps(servicer, client):
    app = App()
    deb_slim = Image.debian_slim()
    image = deb_slim.pip_install("foobarbaz")
    app.function(image=image)(dummy)
    with servicer.intercept() as ctx:
        # default - image not built, should stream
        with app.run(client=client):
            pass
        image_gets = ctx.get_requests("ImageGetOrCreate")
        assert len(image_gets) == 2
        image_joins = ctx.get_requests("ImageJoinStreaming")
        assert len(image_joins) == 2

    with servicer.intercept() as ctx:
        # lets mock that deb_slim has been built already

        async def custom_responder(servicer, stream):
            image_get_or_create_request = await stream.recv_message()
            is_base_image = any("FROM python:" in cmd for cmd in image_get_or_create_request.image.dockerfile_commands)
            if is_base_image:
                # base image done
                await stream.send_message(
                    api_pb2.ImageGetOrCreateResponse(
                        image_id="im-123",
                        result=api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_SUCCESS),
                        metadata=api_pb2.ImageMetadata(),
                    )
                )
            else:
                await stream.send_message(
                    api_pb2.ImageGetOrCreateResponse(
                        image_id="im-124",
                        metadata=api_pb2.ImageMetadata(),
                    )
                )

        ctx.set_responder("ImageGetOrCreate", custom_responder)
        with app.run(client=client):
            pass
        image_gets = ctx.get_requests("ImageGetOrCreate")
        assert len(image_gets) == 2
        image_joins = ctx.get_requests("ImageJoinStreaming")
        assert len(image_joins) == 1  # should now skip building of second build step
        assert image_joins[0].image_id == "im-124"


@pytest.mark.parametrize("copy", [True, False])
def test_image_local_dir_ignore_patterns(servicer, client, tmp_path_with_content, copy):
    ignore = ["**/*.txt", "**/module", "!**/*.txt", "!**/*.py"]
    expected = {
        "/module/sub.py",
        "/module/sub/sub.py",
        "/data/sub",
        "/module/__init__.py",
        "/data.txt",
        "/module/sub/__init__.py",
    }
    app = App()

    img = (
        Image.from_registry("unknown_image")
        .workdir("/proj")
        .add_local_dir(tmp_path_with_content, "/place/", ignore=ignore, copy=copy)
    )
    app.function(image=img)(dummy)
    with app.run(client=client):
        if copy:
            assert len(img._mount_layers) == 0
            layers = get_image_layers(img.object_id, servicer)
            mount_id = layers[0].context_mount_id
        else:
            assert len(img._mount_layers) == 1
            mount_id = img._mount_layers[0].object_id
        assert servicer.mount_contents[mount_id].keys() == {f"/place{f}" for f in expected}


@pytest.mark.parametrize("copy", [True, False])
def test_image_local_dir_ignore_relative(servicer, client, tmp_path_with_content, copy):
    assert (tmp_path_with_content / "data.txt").exists()
    app = App()

    img = Image.from_registry("unknown_image").add_local_dir(
        tmp_path_with_content, "/place", ignore=["*.txt"], copy=copy
    )

    app.function(image=img)(dummy)
    with app.run(client=client):
        if copy:
            assert len(img._mount_layers) == 0
            layers = get_image_layers(img.object_id, servicer)
            mount_id = layers[0].context_mount_id
        else:
            assert len(img._mount_layers) == 1
            mount_id = img._mount_layers[0].object_id
        assert "/place/data.txt" not in servicer.mount_contents[mount_id].keys()


@pytest.mark.parametrize("copy", [True, False])
def test_image_add_local_dir_ignore_callable(servicer, client, tmp_path_with_content, copy):
    def ignore(x: Path):
        return str(x) != "data.txt"

    expected = {"/place/data.txt"}
    app = App()

    img = Image.from_registry("unknown_image").workdir("/proj")
    img = img.add_local_dir(tmp_path_with_content, "/place/", ignore=ignore, copy=copy)
    app.function(image=img)(dummy)
    with app.run(client=client):
        if copy:
            assert len(img._mount_layers) == 0
            layers = get_image_layers(img.object_id, servicer)
            mount_id = layers[0].context_mount_id
            assert set(servicer.mount_contents[mount_id].keys()) == expected
        else:
            assert len(img._mount_layers) == 1
            mount_id = img._mount_layers[0].object_id
            assert set(servicer.mount_contents[mount_id].keys()) == expected


@pytest.mark.parametrize("copy", [True, False])
def test_image_add_local_dir_ignore_nothing(servicer, client, tmp_path_with_content, copy):
    file_paths = set()
    for root, _, files in os.walk(tmp_path_with_content):
        for file in files:
            file_paths.add(os.path.join(root, file))

    expected = {f"/place/{Path(fn).relative_to(tmp_path_with_content).as_posix()}" for fn in file_paths}
    app = App()

    img = Image.from_registry("unknown_image").workdir("/proj")
    img = img.add_local_dir(tmp_path_with_content, "/place/", copy=copy)
    app.function(image=img)(dummy)
    with app.run(client=client):
        if copy:
            assert len(img._mount_layers) == 0
            layers = get_image_layers(img.object_id, servicer)
            mount_id = layers[0].context_mount_id
            assert servicer.mount_contents[mount_id].keys() == expected
        else:
            assert len(img._mount_layers) == 1
            mount_id = img._mount_layers[0].object_id
            assert servicer.mount_contents[mount_id].keys() == expected


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.usefixtures("tmp_cwd")
def test_image_add_local_dir_ignore_from_file(servicer, client, tmp_path_with_content, copy):
    rel_top_dir = Path("top")
    rel_top_dir.mkdir()
    ignore_file = rel_top_dir / "something_special.ignore_file"
    ignore_file.write_text("**/*.txt")

    expected = {"/place/data.txt"}
    app = App()

    img = Image.from_registry("unknown_image").workdir("/proj")

    img = img.add_local_dir(
        tmp_path_with_content, "/place/", ignore=~FilePatternMatcher.from_file(ignore_file), copy=copy
    )
    app.function(image=img)(dummy)
    with app.run(client=client):
        if copy:
            assert len(img._mount_layers) == 0
            layers = get_image_layers(img.object_id, servicer)
            mount_id = layers[0].context_mount_id
            assert servicer.mount_contents[mount_id].keys() == expected
        else:
            assert len(img._mount_layers) == 1
            mount_id = img._mount_layers[0].object_id
            assert servicer.mount_contents[mount_id].keys() == expected


@pytest.mark.usefixtures("tmp_cwd")
def test_raw_dockerfile_image(servicer, client):
    dockerfile_commands = [
        "FROM python:3.6",
        "COPY . .",
        "ENV AGE=old",
        "RUN pip install seaborn==0.6.0",
    ]

    extra_commands = ["echo 'hello'", "echo 'goodbye'"]

    Path("Dockerfile").write_text("\n".join(dockerfile_commands))
    image = raw_dockerfile_image("Dockerfile").run_commands(extra_commands)

    app = App()
    app.function(image=image)(dummy)
    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        assert len(layers) == 2

        assert layers[1].dockerfile_commands == dockerfile_commands
        assert not layers[1].context_files

        assert layers[0].dockerfile_commands == ["FROM base"] + [f"RUN {c}" for c in extra_commands]
        assert not layers[0].context_files


@pytest.mark.usefixtures("tmp_cwd")
def test_raw_registry_image(servicer, client):
    tag = "python:3.6"
    extra_commands = ["echo 'hello'", "echo 'goodbye'"]

    image = raw_registry_image(tag).run_commands(extra_commands)

    app = App()
    app.function(image=image)(dummy)
    with app.run(client=client):
        layers = get_image_layers(image.object_id, servicer)
        assert len(layers) == 2

        assert layers[1].dockerfile_commands == [f"FROM {tag}"]
        assert not layers[1].context_files

        assert layers[0].dockerfile_commands == ["FROM base"] + [f"RUN {c}" for c in extra_commands]
        assert not layers[0].context_files

    registry_secret = Secret.from_name("registry-secret")
    with pytest.raises(InvalidError, match="credential_type"):
        raw_registry_image(tag, registry_secret=registry_secret)

    with pytest.raises(InvalidError, match="whatever"):
        raw_registry_image(tag, registry_secret=registry_secret, credential_type="whatever")  # type: ignore


def test_hydration_refusal():
    with pytest.raises(ExecutionError, match="Images cannot currently be hydrated on demand"):
        Image.debian_slim().hydrate()
