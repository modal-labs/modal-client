import os
import shlex
import sys
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Union

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.grpc_utils import retry_transient_errors

from .config import config, logger
from .exception import InvalidError, NotFoundError, RemoteError
from .object import Object, ref
from .secret import _Secret


class _Image(Object, type_prefix="im"):
    """Base class for container images to run functions in.

    Do not construct this class directly; instead use
    `modal.image.DebianSlim`, `modal.image.DockerhubImage` or `modal.image.Conda`.
    """

    def __init__(
        self,
        base_images={},
        context_files={},
        dockerfile_commands: Union[List[str], Callable[[], List[str]]] = [],
        secrets=[],
        version=None,
        ref=None,
    ) -> None:
        if ref and (base_images or dockerfile_commands or context_files):
            raise InvalidError("No other arguments can be provided when initializing an image from a ref.")
        if not ref and not dockerfile_commands:
            raise InvalidError("No commands were provided for the image — have you tried using modal.DebianSlim()?")

        self._ref = ref
        self._base_images = base_images
        self._context_files = context_files
        self._dockerfile_commands = dockerfile_commands
        self._version = version
        self._secrets = secrets
        super().__init__()

    async def _load(self, client, app_id, existing_image_id):
        if self._ref:
            return await self._ref

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

        dockerfile_commands: List[str]
        if callable(self._dockerfile_commands):
            # It's a closure (see DockerfileImage)
            dockerfile_commands = self._dockerfile_commands()
        else:
            dockerfile_commands = self._dockerfile_commands
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
            response = await retry_transient_errors(client.stub.ImageJoin, request)
            if not response.result.status:
                continue
            elif response.result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE:
                raise RemoteError(response.result.exception)
            elif response.result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
                break
            else:
                raise RemoteError("Unknown status %s!" % response.result.status)

        return image_id

    def _is_inside(self) -> bool:
        """Returns whether this container is active or not.

        This is not meant to be called directly: see app.is_inside(image)
        """
        env_image_id = config.get("image_id")
        logger.debug(f"Is image inside? env {env_image_id} image {self.object_id}")
        return self.object_id == env_image_id

    def extend(self, **kwargs) -> "_Image":
        """Extend an image (named "base") with additional options or commands.

        This is a low-level command. Generally, you should prefer using functions
        like `Image.pip_install` or `DebianSlim.apt_install` if possible.

        **Example**

        ```python notest
        image = modal.DebianSlim().extend(
            dockerfile_commands=[
                "FROM base",
                "WORKDIR /pkg",
                'RUN echo "hello world" > hello.txt',
            ],
            secrets=[secret1, secret2],
        )
        ```
        """

        obj = _Image.__new__(type(self))
        _Image.__init__(obj, base_images={"base": self}, **kwargs)
        return obj

    def pip_install(
        self,
        packages: List[str] = [],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
        find_links: Optional[str] = None,
    ) -> "_Image":
        """Install a list of Python packages using pip."""
        if not isinstance(packages, list) or any(not isinstance(package, str) for package in packages):
            raise InvalidError("pip_install: packages must be a list of Python packages (as strings)")

        find_links_arg = f"-f {find_links}" if find_links else ""
        package_args = " ".join(shlex.quote(pkg) for pkg in packages)

        dockerfile_commands = [
            "FROM base",
            f"RUN pip install {package_args} {find_links_arg}",
        ]

        return self.extend(dockerfile_commands=dockerfile_commands)

    def pip_install_from_requirements(
        self,
        requirements_txt: str,  # Path to a requirements.txt file.
        find_links: Optional[str] = None,
    ) -> "_Image":
        """Install a list of Python packages from a `requirements.txt` file."""

        requirements_txt = os.path.expanduser(requirements_txt)

        find_links_arg = f"-f {find_links}" if find_links else ""
        context_files = {"/.requirements.txt": requirements_txt}

        dockerfile_commands = [
            "FROM base",
            "COPY /.requirements.txt /.requirements.txt",
            f"RUN pip install -r /.requirements.txt {find_links_arg}",
        ]

        return self.extend(
            dockerfile_commands=dockerfile_commands,
            context_files=context_files,
        )

    def poetry_install_from_file(
        self,
        poetry_pyproject_toml: str,
    ):
        """Install poetry dependencies specified by a pyproject.toml file.

        If a poetry.lock file exists in the same directory, then this will be
        used to specify exact dependency versions instead.
        """

        poetry_pyproject_toml = os.path.expanduser(poetry_pyproject_toml)

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
            "RUN cd /tmp/poetry && \\ ",
            "  poetry config virtualenvs.create false && \\ ",
            "  poetry install --no-root",
        ]

        return self.extend(
            dockerfile_commands=dockerfile_commands,
            context_files=context_files,
        )

    def dockerfile_commands(
        self,
        dockerfile_commands: Union[str, List[str]],
        context_files: Dict[str, str] = {},
        secrets: Collection[_Secret] = [],
    ):
        """Extend an image with arbitrary Dockerfile-like commands."""

        _dockerfile_commands = ["FROM base"]

        if isinstance(dockerfile_commands, str):
            _dockerfile_commands += dockerfile_commands.split("\n")
        else:
            _dockerfile_commands += dockerfile_commands

        return self.extend(
            dockerfile_commands=_dockerfile_commands,
            context_files=context_files,
            secrets=secrets,
        )

    def run_commands(
        self,
        commands: List[str],
        secrets: Collection[_Secret] = [],
    ):
        """Extend an image with a list of shell commands to run."""
        dockerfile_commands = ["FROM base"] + [f"RUN {cmd}" for cmd in commands]

        return self.extend(
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
    """Default image, based on the official `python:X.Y.Z-slim-bullseye` Docker images.

    This image also be called as a function and customized, which allows you to
    extend the image with additional shell commands or Python packages.
    """

    def __init__(self, python_version: Optional[str] = None):
        """Construct a default Modal image based on Debian-Slim."""
        python_version = _dockerhub_python_version(python_version)
        base_image = ref(f"debian-slim-{python_version}", namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL)

        super().__init__(ref=base_image)

    def apt_install(
        self,
        packages: List[str] = [],  # A list of packages, e.g. ["ssh", "libpq-dev"]
    ):
        """Install a list of Debian packages using `apt`."""

        package_args = " ".join(shlex.quote(pkg) for pkg in packages)

        dockerfile_commands = [
            "FROM base",
            "RUN apt-get update",
            f"RUN apt-get install -y {package_args}",
        ]

        return self.extend(dockerfile_commands=dockerfile_commands)


def get_client_requirements_path():
    # Locate Modal client requirements.txt
    import modal

    modal_path = modal.__path__[0]
    return os.path.join(modal_path, "requirements.txt")


def _DockerhubImage(app=None, tag=None):
    """
    Build a Modal image from a pre-existing image on Docker Hub.

    This assumes the following about the image:

    - Python 3.7 or above is present, and is available as `python`.
    - `pip` is installed correctly.
    - The image is built for the `linux/amd64` platform.
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


def _DockerfileImage(path: Union[str, Path]):
    """Build a Modal image from a local Dockerfile.

    Note that the the following must be true about the image you provide:

    - Python 3.7 or above needs to be present and available as `python`.
    - `pip` needs to be installed and available as `pip`.
    """

    path = os.path.expanduser(path)

    def base_dockerfile_commands():
        # Make it a closure so that it's only invoked locally
        with open(path) as f:
            return f.read().split("\n")

    base_image = _Image(dockerfile_commands=base_dockerfile_commands)

    requirements_path = get_client_requirements_path()

    dockerfile_commands = [
        "FROM base",
        "COPY /modal_requirements.txt /modal_requirements.txt",
        "RUN pip install --upgrade pip",
        "RUN pip install -r /modal_requirements.txt",
    ]

    return base_image.extend(
        dockerfile_commands=dockerfile_commands,
        context_files={"/modal_requirements.txt": requirements_path},
    )


class _Conda(_Image):
    """A Conda base image, built on the official miniconda3 Docker Hub image."""

    def __init__(self):
        """Construct the default base Conda image."""
        super().__init__(ref=ref("conda", namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL))

    def conda_install(
        self,
        packages: List[str] = [],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
    ):
        """Install a list of additional packages using conda."""

        package_args = " ".join(shlex.quote(pkg) for pkg in packages)

        dockerfile_commands = [
            "FROM base",
            f"RUN conda install {package_args} --yes",
        ]

        return self.extend(dockerfile_commands=dockerfile_commands)


Conda, AioConda = synchronize_apis(_Conda)
DebianSlim, AioDebianSlim = synchronize_apis(_DebianSlim)
DockerhubImage, AioDockerhubImage = synchronize_apis(_DockerhubImage)
DockerfileImage, AioDockerfileImage = synchronize_apis(_DockerfileImage)
Image, AioImage = synchronize_apis(_Image)
