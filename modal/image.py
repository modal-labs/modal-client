import os
import shlex
import sys
import warnings
from pathlib import Path
from typing import Collection, Dict, List, Optional, Union

from modal_proto import api_pb2
from modal_utils.async_utils import retry, synchronize_apis

from .config import config, logger
from .exception import InvalidError, NotFoundError, RemoteError
from .object import Object, ref
from .secret import _Secret


def _make_bytes(s):
    assert type(s) in (str, bytes)
    return s.encode("utf-8") if type(s) is str else s


class _Image(Object, type_prefix="im"):
    """Base class for container images to run functions in.

    Do not construct this class directly; instead use
    `modal.image.DebianSlim`, `modal.image.DockerHubImage` or `modal.image.Conda`.
    """

    def __init__(
        self,
        base_images={},
        context_files={},
        dockerfile_commands=[],
        secrets=[],
        version=None,
    ):
        self._base_images = base_images
        self._context_files = context_files
        self._dockerfile_commands = dockerfile_commands
        self._version = version
        self._secrets = secrets
        super().__init__()

    async def load(self, client, app_id, existing_image_id):
        # Recursively build base images
        base_image_ids = []
        for image in self._base_images.values():
            base_image_ids.append(await image)
        base_images_pb2s = [
            api_pb2.BaseImage(docker_tag=docker_tag, image_id=image_id)
            for docker_tag, image_id in zip(self._base_images.keys(), base_image_ids)
        ]

        secret_ids = []
        for secret in self._secrets:
            try:
                secret_id = await secret
            except NotFoundError as ex:
                raise NotFoundError(str(ex) + "\n" + "You can add secrets to your account at https://modal.com/secrets")
            secret_ids.append(secret_id)

        context_file_pb2s = []
        for filename, path in self._context_files.items():
            with open(path, "rb") as f:
                context_file_pb2s.append(api_pb2.ImageContextFile(filename=filename, data=f.read()))

        dockerfile_commands = [_make_bytes(s) for s in self._dockerfile_commands]
        image_definition = api_pb2.Image(
            base_images=base_images_pb2s,
            dockerfile_commands=dockerfile_commands,
            context_files=context_file_pb2s,
            version=self._version,
            secret_ids=secret_ids,
        )

        req = api_pb2.ImageGetOrCreateRequest(
            app_id=app_id,
            image=image_definition,
            existing_image_id=existing_image_id,  # TODO: ignored
        )
        resp = await client.stub.ImageGetOrCreate(req)
        image_id = resp.image_id

        logger.debug("Waiting for image %s" % image_id)
        while True:
            request = api_pb2.ImageJoinRequest(image_id=image_id, timeout=60)
            response = await retry(client.stub.ImageJoin)(request)
            if not response.result.status:
                continue
            elif response.result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE:
                raise RemoteError(response.result.exception)
            elif response.result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
                break
            else:
                raise RemoteError("Unknown status %s!" % response.result.status)

        return image_id

    def _is_inside(self):
        """Returns whether this container is active or not.

        This is not meant to be called directly: see app.is_inside(image)
        """
        env_image_id = config.get("image_id")
        logger.debug(f"Is image inside? env {env_image_id} image {self.object_id}")
        return self.object_id == env_image_id

    def pip_install(
        self,
        packages: List[str] = [],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
        find_links: Optional[str] = None,
    ):
        """Install a list of packages using pip."""

        find_links_arg = f"-f {find_links}" if find_links else ""
        package_args = " ".join(shlex.quote(pkg) for pkg in packages)

        dockerfile_commands = [
            "FROM base",
            f"RUN pip install {package_args} {find_links_arg}",
        ]

        return _Image(
            base_images={"base": self},
            dockerfile_commands=dockerfile_commands,
        )

    def pip_install_from_requirements(
        self,
        requirements_txt: str,  # Path to a requirements.txt file.
        find_links: Optional[str] = None,
    ):
        """Install a list of packages using pip."""

        find_links_arg = f"-f {find_links}" if find_links else ""
        context_files = {"/.requirements.txt": requirements_txt}

        dockerfile_commands = [
            "FROM base",
            "COPY /.requirements.txt /.requirements.txt",
            f"RUN pip install -r /.requirements.txt {find_links_arg}",
        ]

        return _Image(
            base_images={"base": self},
            dockerfile_commands=dockerfile_commands,
            context_files=context_files,
        )

    def poetry_install_from_file(
        self,
        poetry_pyproject_toml: str,
    ):
        """Install poetry deps from a pyproject.toml file. Uses poetry.lock
        if it exists."""

        dockerfile_commands = ["FROM base", "RUN pip install poetry"]

        context_files = {"/.pyproject.toml": poetry_pyproject_toml}

        poetry_lockfile: Path = Path(poetry_pyproject_toml).parent / "poetry.lock"
        if poetry_lockfile.exists():
            context_files["/.poetry.lock"] = poetry_lockfile.as_posix()
            dockerfile_commands += ["COPY /.poetry.lock /tmp/poetry/poetry.lock"]
        else:
            logger.warn("poetry.lock not found.")

        dockerfile_commands += [
            "COPY /.pyproject.toml /tmp/poetry/pyproject.toml",
            "RUN cd /tmp/poetry && \ ",  # noqa
            "  poetry config virtualenvs.create false && \ ",  # noqa
            "  poetry install --no-root",
        ]

        return _Image(
            base_images={"base": self},
            dockerfile_commands=dockerfile_commands,
            context_files=context_files,
        )

    def dockerfile_commands(
        self,
        dockerfile_commands: Union[str, List[str]],
        context_files: Dict[str, str] = {},
        secrets: Collection[_Secret] = [],
    ):
        """Extend an image with arbitrary dockerfile commands"""

        _dockerfile_commands = ["FROM base"]

        if isinstance(dockerfile_commands, str):
            _dockerfile_commands += dockerfile_commands.split("\n")
        else:
            _dockerfile_commands += dockerfile_commands

        return _Image(
            base_images={"base": self},
            dockerfile_commands=_dockerfile_commands,
            context_files=context_files,
            secrets=secrets,
        )

    def run_commands(
        self,
        commands: List[str],
        secrets: Collection[_Secret] = [],
    ):
        """Extend an image with a list of run commands"""
        dockerfile_commands = ["FROM base"] + [f"RUN {cmd}" for cmd in commands]

        return _Image(
            base_images={"base": self},
            dockerfile_commands=dockerfile_commands,
            secrets=secrets,
        )


def _dockerhub_python_version(python_version=None):
    if python_version is None:
        python_version = config["image_python_version"]
    if python_version is None:
        python_version = "%d.%d" % sys.version_info[:2]

    # We use the same major/minor version, but the highest micro version
    # See https://hub.docker.com/_/python
    latest_micro_version = {
        "3.10": "1",
        "3.9": "9",
        "3.8": "12",
        "3.7": "12",
    }
    major_minor_version = ".".join(python_version.split(".")[:2])
    python_version = major_minor_version + "." + latest_micro_version[major_minor_version]
    return python_version


class _DebianSlim(_Image):
    """A default base image, built on the official python:x.y.z-slim-bullseye Docker hub images

    Can also be called as a function to build a new image with additional bash
    commands or python packages.
    """

    def __init__(
        self,
        app=None,
        extra_commands: List[str] = [],  # A list of shell commands executed while building the image
        python_packages: List[str] = [],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
        python_version: Optional[str] = None,  # Set a specific Python version
        pip_find_links: Optional[str] = None,
        requirements_txt: Optional[str] = None,  # File contents of a requirements.txt
        context_files: Dict[
            str, str
        ] = {},  # A dict containing path to files that will be present during the build to use with COPY
        secrets: List[
            Object
        ] = [],  # List of Modal secrets that will be available as environment variables during the build process
        version: Optional[str] = None,  # Custom string to break the image hashing and force the image to be rebuilt
    ):
        if app is not None:
            raise InvalidError("The latest API does no longer require the `app` argument, so please update your code!")

        python_version = _dockerhub_python_version(python_version)
        base_image = ref(f"debian-slim-{python_version}", namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)

        dockerfile_commands = ["FROM base as target"]
        base_images = {"base": base_image}
        dockerfile_commands += [f"RUN {cmd}" for cmd in extra_commands]

        if requirements_txt is not None:
            warnings.warn(
                "Arguments to DebianSlim are deprecated. Use image.pip_install_from_requirements() instead",
                DeprecationWarning,
            )
            context_files = context_files.copy()
            context_files["/.requirements.txt"] = requirements_txt

            dockerfile_commands += [
                "COPY /.requirements.txt /.requirements.txt",
                "RUN pip install -r /.requirements.txt",
            ]

        if python_packages:
            warnings.warn("Arguments to DebianSlim are deprecated. Use image.pip_install() instead", DeprecationWarning)

            find_links_arg = f"-f {pip_find_links}" if pip_find_links else ""
            package_args = " ".join(shlex.quote(pkg) for pkg in python_packages)

            dockerfile_commands += [
                f"RUN pip install {package_args} {find_links_arg}",
            ]

        if extra_commands:
            warnings.warn(
                "Arguments to DebianSlim are deprecated. Use image.run_commands() instead", DeprecationWarning
            )

        super().__init__(
            dockerfile_commands=dockerfile_commands,
            context_files=context_files,
            base_images=base_images,
            version=version,
            secrets=secrets,
        )

    def apt_install(
        self,
        packages: List[str] = [],  # A list of Debian packages, eg. ["ssh", "libpq-dev"]
    ):
        """Install a list of Debian using apt."""

        package_args = " ".join(shlex.quote(pkg) for pkg in packages)

        dockerfile_commands = [
            "FROM base",
            "RUN apt-get update",
            f"RUN apt-get install -y {package_args}",
        ]

        return _Image(
            base_images={"base": self},
            dockerfile_commands=dockerfile_commands,
        )


def get_client_requirements_path():
    # Locate Modal client requirements.txt
    import modal

    modal_path = modal.__path__[0]
    return os.path.join(modal_path, "requirements.txt")


def _DockerhubImage(app=None, tag=None):
    """
    Build a modal image from a pre-existing image on DockerHub.

    This assumes the following about the image:
    - Python 3.7 or above is present, and is available as `python`
    - `pip` is installed correctly
    - The image is built for the `linux/amd64` platform
    """
    if app is not None:
        raise InvalidError("The latest API does no longer require the `app` argument, so please update your code!")

    requirements_path = get_client_requirements_path()

    dockerfile_commands = [
        f"FROM {tag}",
        "COPY /modal_requirements.txt /modal_requirements.txt",
        "RUN pip install --upgrade pip",
        "RUN pip install -r /modal_requirements.txt",
    ]

    return _Image(
        dockerfile_commands=dockerfile_commands,
        context_files={"/modal_requirements.txt": requirements_path},
    )


class _Conda(_Image):
    """A Conda base image, built on the official miniconda3 Docker hub image."""

    def __init__(self):
        super().__init__()

    # Override load to just resolve a ref.
    async def load(self, client, app_id, existing_image_id):
        return await ref("conda", namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)

    def conda_install(
        self,
        packages: List[str] = [],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
    ):
        """Install a list of packages using conda."""

        package_args = " ".join(shlex.quote(pkg) for pkg in packages)

        dockerfile_commands = [
            "FROM base",
            f"RUN conda install {package_args} --yes",
        ]

        return _Image(
            base_images={"base": self},
            dockerfile_commands=dockerfile_commands,
        )


Conda, AioConda = synchronize_apis(_Conda)
DebianSlim, AioDebianSlim = synchronize_apis(_DebianSlim)
DockerhubImage, AioDockerhubImage = synchronize_apis(_DockerhubImage)
Image, AioImage = synchronize_apis(_Image)
