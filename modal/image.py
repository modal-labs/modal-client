# Copyright Modal Labs 2022
from __future__ import annotations

import inspect
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Callable, Collection, Optional, Union

import toml

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.grpc_utils import retry_transient_errors

from .config import config, logger
from .exception import InvalidError, NotFoundError, RemoteError
from .mount import _Mount
from .object import Handle, Provider
from .secret import _Secret


def _validate_python_version(version: str) -> None:
    components = version.split(".")
    supported_versions = {"3.10", "3.9", "3.8", "3.7"}
    if len(components) == 2 and version in supported_versions:
        return
    elif len(components) == 3:
        raise InvalidError(
            f"major.minor.patch version specification not valid. Supported major.minor versions are {supported_versions}."
        )
    raise InvalidError(f"Unsupported version {version}. Supported versions are {supported_versions}.")


def _dockerhub_python_version(python_version=None):
    if python_version is None:
        python_version = config["image_python_version"]
    if python_version is None:
        python_version = "%d.%d" % sys.version_info[:2]

    # We use the same major/minor version, but the highest micro version
    # See https://hub.docker.com/_/python
    latest_micro_version = {
        "3.11": "0",
        "3.10": "8",
        "3.9": "15",
        "3.8": "15",
        "3.7": "15",
    }
    major_minor_version = ".".join(python_version.split(".")[:2])
    python_version = major_minor_version + "." + latest_micro_version[major_minor_version]
    return python_version


def _get_client_requirements_path():
    # Locate Modal client requirements.txt
    import modal

    modal_path = modal.__path__[0]
    return os.path.join(modal_path, "requirements.txt")


def _flatten_str_args(function_name: str, arg_name: str, args: tuple[Union[str, list[str]], ...]) -> list[str]:
    """Takes a tuple of strings, or string lists, and flattens it.

    Raises an error if any of the elements are not strings or string lists.
    """
    # TODO(erikbern): maybe we can just build somthing intelligent that checks
    # based on type annotations in real time?
    # Or use something like this? https://github.com/FelixTheC/strongtyping

    def is_str_list(x):
        return isinstance(x, list) and all(isinstance(y, str) for y in x)

    ret: list[str] = []
    for x in args:
        if isinstance(x, str):
            ret.append(x)
        elif is_str_list(x):
            ret.extend(x)
        else:
            raise InvalidError(f"{function_name}: {arg_name} must only contain strings")
    return ret


class _ImageHandle(Handle, type_prefix="im"):
    def _is_inside(self) -> bool:
        """Returns whether this container is active or not.

        This is not meant to be called directly: see app.is_inside(image)
        """
        env_image_id = config.get("image_id")
        logger.debug(f"Image._is_inside(): env_image_id={env_image_id} self.object_id={self.object_id}")
        return self.object_id == env_image_id


class _Image(Provider[_ImageHandle]):
    """Base class for container images to run functions in.

    Do not construct this class directly; instead use one of its static factory methods,
    like `modal.Image.from_dockerhub` or `modal.Image.conda`.
    """

    def __init__(
        self,
        base_images={},
        context_files={},
        dockerfile_commands: Union[list[str], Callable[[], list[str]]] = [],
        secrets=[],
        ref=None,
        gpu=False,
        build_function=None,
        context_mount: _Mount = None,
    ):
        if ref and (base_images or dockerfile_commands or context_files):
            raise InvalidError("No other arguments can be provided when initializing an image from a ref.")
        if not ref and not dockerfile_commands and not build_function:
            raise InvalidError(
                "No commands were provided for the image — have you tried using modal.Image.debian_slim()?"
            )

        if build_function and dockerfile_commands:
            raise InvalidError("Cannot provide both a build function and Dockerfile commands!")

        if build_function and len(base_images) != 1:
            raise InvalidError("Cannot run a build function with multiple base images!")

        self._ref = ref
        self._base_images = base_images
        self._context_files = context_files
        self._dockerfile_commands = dockerfile_commands
        self._secrets = secrets
        self._gpu = gpu
        self._build_function = build_function
        self._context_mount = context_mount
        super().__init__()

    def __repr__(self):
        return f"Image({self._dockerfile_commands})"

    async def _load(self, client, stub, app_id, loader, message_callback, existing_image_id):
        if self._ref:
            image_id = await loader(self._ref)
            return _ImageHandle._from_id(image_id, client)

        # Recursively build base images
        base_image_ids: list[str] = []
        for image in self._base_images.values():
            base_image_ids.append(await loader(image))
        base_images_pb2s = [
            api_pb2.BaseImage(docker_tag=docker_tag, image_id=image_id)
            for docker_tag, image_id in zip(self._base_images.keys(), base_image_ids)
        ]

        secret_ids = []
        for secret in self._secrets:
            try:
                secret_id = await loader(secret)
            except NotFoundError as ex:
                raise NotFoundError(str(ex) + "\n" + "You can add secrets to your account at https://modal.com/secrets")
            secret_ids.append(secret_id)

        context_file_pb2s = []
        for filename, path in self._context_files.items():
            with open(path, "rb") as f:
                context_file_pb2s.append(api_pb2.ImageContextFile(filename=filename, data=f.read()))

        if self._build_function:
            (fn, kwargs) = self._build_function
            # Plaintext source and arg definition for the function, so it's part of the image
            # hash. We can't use the cloudpickle hash because it's not very stable.
            build_function_def = f"{inspect.getsource(fn)}\n{repr(kwargs)}"

            base_images = list(self._base_images.values())
            assert len(base_images) == 1
            kwargs = {"timeout": 86400, **kwargs, "image": base_images[0], "_is_build_step": True}
            build_function_handle = stub.function(**kwargs)(fn)
            build_function_id = await loader(build_function_handle._function)
        else:
            build_function_def = None
            build_function_id = None

        dockerfile_commands: list[str]
        if callable(self._dockerfile_commands):
            # It's a closure (see DockerfileImage)
            dockerfile_commands = self._dockerfile_commands()
        else:
            dockerfile_commands = self._dockerfile_commands

        if self._context_mount:
            context_mount_id = await loader(self._context_mount)
        else:
            context_mount_id = None

        image_definition = api_pb2.Image(
            base_images=base_images_pb2s,
            dockerfile_commands=dockerfile_commands,
            context_files=context_file_pb2s,
            secret_ids=secret_ids,
            gpu=self._gpu,
            build_function_def=build_function_def,
            context_mount_id=context_mount_id,
        )

        req = api_pb2.ImageGetOrCreateRequest(
            app_id=app_id,
            image=image_definition,
            existing_image_id=existing_image_id,  # TODO: ignored
            build_function_id=build_function_id,
        )
        resp = await client.stub.ImageGetOrCreate(req)
        image_id = resp.image_id

        logger.debug("Waiting for image %s" % image_id)
        while True:
            request = api_pb2.ImageJoinRequest(image_id=image_id, timeout=55)
            response = await retry_transient_errors(client.stub.ImageJoin, request)
            if not response.result.status:
                continue
            elif response.result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE:
                raise RemoteError(response.result.exception)
            elif response.result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED:
                raise RemoteError("Image build terminated due to external shut-down. Please try again.")
            elif response.result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
                raise RemoteError("Image build timed out. Please try again with a larger `timeout` parameter.")
            elif response.result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
                break
            else:
                raise RemoteError("Unknown status %s!" % response.result.status)

        return _ImageHandle(client, image_id)

    def extend(self, **kwargs) -> "_Image":
        """Extend an image (named "base") with additional options or commands.

        This is a low-level command. Generally, you should prefer using functions
        like `Image.pip_install` or `Image.apt_install` if possible.

        **Example**

        ```python notest
        image = modal.Image.debian_slim().extend(
            dockerfile_commands=[
                "FROM base",
                "WORKDIR /pkg",
                'RUN echo "hello world" > hello.txt',
            ],
            secrets=[secret1, secret2],
        )
        ```
        """

        return _Image(base_images={"base": self}, **kwargs)

    def copy(self, mount: _Mount, remote_path: Union[str, Path] = "."):
        return self.extend(
            dockerfile_commands=["FROM base", f"COPY . {remote_path}"],  # copy everything from the supplied mount
            context_mount=mount,
        )

    def pip_install(
        self,
        *packages: Union[str, list[str]],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
        find_links: Optional[str] = None,
    ) -> "_Image":
        """Install a list of Python packages using pip."""
        pkgs = _flatten_str_args("pip_install", "packages", packages)
        if not pkgs:
            return self

        find_links_arg = f"-f {find_links}" if find_links else ""
        package_args = " ".join(shlex.quote(pkg) for pkg in pkgs)

        dockerfile_commands = [
            "FROM base",
            f"RUN python -m pip install {package_args} {find_links_arg}",
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
            f"RUN python -m pip install -r /.requirements.txt {find_links_arg}",
        ]

        return self.extend(
            dockerfile_commands=dockerfile_commands,
            context_files=context_files,
        )

    def pip_install_from_pyproject(
        self,
        pyproject_toml: str,
    ):
        """Install dependencies specified by a pyproject.toml file."""
        from modal import is_local

        # Don't re-run inside container.
        if not is_local():
            return []

        pyproject_toml = os.path.expanduser(pyproject_toml)

        config = toml.load(pyproject_toml)

        return self.pip_install(*config["project"]["dependencies"])

    def poetry_install_from_file(
        self,
        poetry_pyproject_toml: str,
        poetry_lockfile: Optional[
            str
        ] = None,  # Path to the lockfile. If not provided, uses poetry.lock in the same directory.
        ignore_lockfile=False,  # If set to True, it will not use poetry.lock
        old_installer=False,  # If set to True, use old installer. See https://github.com/python-poetry/poetry/issues/3336
    ):
        """Install poetry dependencies specified by a pyproject.toml file.

        The path to the lockfile is inferred, if not provided. However, the
        file has to exist, unless `ignore_lockfile` is set to `True`.
        """

        poetry_pyproject_toml = os.path.expanduser(poetry_pyproject_toml)

        dockerfile_commands = [
            "FROM base",
            "RUN python -m pip install poetry",
        ]

        if old_installer:
            dockerfile_commands += ["RUN poetry config experimental.new-installer false"]

        context_files = {"/.pyproject.toml": poetry_pyproject_toml}

        if not ignore_lockfile:
            if poetry_lockfile is None:
                p = Path(poetry_pyproject_toml).parent / "poetry.lock"
                if not p.exists():
                    raise NotFoundError(f"poetry.lock not found at {p}")
                poetry_lockfile = p.as_posix()
            context_files["/.poetry.lock"] = poetry_lockfile
            dockerfile_commands += ["COPY /.poetry.lock /tmp/poetry/poetry.lock"]

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
        dockerfile_commands: Union[str, list[str]],
        context_files: dict[str, str] = {},
        secrets: Collection[_Secret] = [],
        gpu: bool = False,
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
            gpu=gpu,
        )

    def run_commands(
        self,
        *commands: Union[str, list[str]],
        secrets: Collection[_Secret] = [],
        gpu: bool = False,
    ):
        """Extend an image with a list of shell commands to run."""
        cmds = _flatten_str_args("run_commands", "commands", commands)
        if not cmds:
            return self

        dockerfile_commands = ["FROM base"] + [f"RUN {cmd}" for cmd in cmds]

        return self.extend(
            dockerfile_commands=dockerfile_commands,
            secrets=secrets,
            gpu=gpu,
        )

    @staticmethod
    def conda(python_version: str = "3.9") -> "_Image":
        """A Conda base image, using miniconda3 and derived from the official Docker Hub image."""
        _validate_python_version(python_version)
        requirements_path = _get_client_requirements_path()
        # Doesn't use the official continuumio/miniconda3 image as a base. That image has maintenance
        # issues (https://github.com/ContinuumIO/docker-images/issues) and building our own is more flexible.
        conda_install_script = "https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh"
        dockerfile_commands = [
            "FROM debian:bullseye",  # the -slim images lack files required by Conda.
            # Temporarily add utility packages for conda installation.
            "RUN apt-get --quiet update && apt-get --quiet --yes install curl bzip2 \\",
            f"&& curl --silent --show-error --location {conda_install_script} --output /tmp/miniconda.sh \\",
            # Install miniconda to a filesystem location on the $PATH of Modal container tasks.
            # -b = install in batch mode w/o manual intervention.
            # -f = allow install prefix to already exist.
            # -p = the install prefix location.
            "&& bash /tmp/miniconda.sh -bfp /usr/local \\ ",
            "&& rm -rf /tmp/miniconda.sh",
            # Biggest and most stable community-led Conda channel.
            "RUN conda config --add channels conda-forge \\ ",
            # "Strict channel priority can dramatically speed up conda operations and also reduce package incompatibility problems."
            "&& conda config --set channel_priority strict \\ ",
            # softlinking can put conda in a broken state, surfacing error on uninstall like:
            # `No such device or address: '/usr/local/lib/libz.so' -> '/usr/local/lib/libz.so.c~'`
            "&& conda config --set allow_softlinks false \\ ",
            # Install requested Python version from conda-forge channel; base debian image has only 3.7.
            f"&& conda install --yes --channel conda-forge python={python_version} \\ ",
            "&& conda update conda \\ ",
            # Remove now unneeded packages and files.
            "&& apt-get --quiet --yes remove curl bzip2 \\ ",
            "&& apt-get --quiet --yes autoremove \\ ",
            "&& apt-get autoclean \\ ",
            "&& rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \\ ",
            "&& conda clean --all --yes",
            # Setup .bashrc for conda.
            "RUN conda init bash --verbose",
            "COPY /modal_requirements.txt /modal_requirements.txt",
            # .bashrc is explicitly sourced because RUN is a non-login shell and doesn't run bash.
            "RUN . /root/.bashrc && conda activate base \\ ",
            "&& python -m pip install --upgrade pip \\ ",
            "&& python -m pip install -r /modal_requirements.txt",
        ]

        return _Image(
            dockerfile_commands=dockerfile_commands,
            context_files={"/modal_requirements.txt": requirements_path},
        )

    def conda_install(
        self,
        *packages: Union[str, list[str]],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
        channels: list[str] = [],  # A list of Conda channels, eg. ["conda-forge", "nvidia"]
    ) -> "_Image":
        """Install a list of additional packages using conda."""
        pkgs = _flatten_str_args("conda_install", "packages", packages)
        if not pkgs:
            return self

        package_args = " ".join(shlex.quote(pkg) for pkg in pkgs)
        channel_args = "".join(f" -c {channel}" for channel in channels)

        dockerfile_commands = [
            "FROM base",
            f"RUN conda install {package_args}{channel_args} --yes \\ ",
            "&& conda clean --yes --index-cache --tarballs --tempfiles --logfiles",
        ]

        return self.extend(dockerfile_commands=dockerfile_commands)

    def conda_update_from_environment(
        self,
        environment_yml: str,
    ) -> "_Image":
        """Update conda environment using dependencies from a given environment.yml file."""

        environment_yml = os.path.expanduser(environment_yml)

        context_files = {"/environment.yml": environment_yml}

        dockerfile_commands = [
            "FROM base",
            "COPY /environment.yml /environment.yml",
            "RUN conda env update --name base -f /environment.yml \\ ",
            "&& conda clean --yes --index-cache --tarballs --tempfiles --logfiles",
        ]

        return self.extend(dockerfile_commands=dockerfile_commands, context_files=context_files)

    @staticmethod
    def from_dockerhub(tag: str, setup_commands: list[str] = [], **kwargs) -> "_Image":
        """
        Build a Modal image from a pre-existing image on Docker Hub.

        This assumes the following about the image:

        - Python 3.7 or above is present, and is available as `python`.
        - `pip` is installed correctly.
        - The image is built for the `linux/amd64` platform.

        You can use the `setup_commands` argument to run any
        commands in the image before Modal is installed.
        This might be useful if Python or pip is not installed.
        For instance:
        ```python
        modal.Image.from_dockerhub(
          "gisops/valhalla:latest",
          setup_commands=["apt-get update", "apt-get install -y python3-pip"]
        )
        ```
        """
        requirements_path = _get_client_requirements_path()

        dockerfile_commands = [
            f"FROM {tag}",
            *(f"RUN {cmd}" for cmd in setup_commands),
            "COPY /modal_requirements.txt /modal_requirements.txt",
            "RUN python -m pip install --upgrade pip",
            "RUN python -m pip install -r /modal_requirements.txt",
        ]

        return _Image(
            dockerfile_commands=dockerfile_commands,
            context_files={"/modal_requirements.txt": requirements_path},
            **kwargs,
        )

    @staticmethod
    def from_dockerfile(path: Union[str, Path]) -> "_Image":
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

        requirements_path = _get_client_requirements_path()

        dockerfile_commands = [
            "FROM base",
            "COPY /modal_requirements.txt /modal_requirements.txt",
            "RUN python -m pip install --upgrade pip",
            "RUN python -m pip install -r /modal_requirements.txt",
        ]

        return base_image.extend(
            dockerfile_commands=dockerfile_commands,
            context_files={"/modal_requirements.txt": requirements_path},
        )

    @staticmethod
    def debian_slim(python_version: Optional[str] = None) -> "_Image":
        """Default image, based on the official `python:X.Y.Z-slim-bullseye` Docker images."""
        python_version = _dockerhub_python_version(python_version)

        requirements_path = _get_client_requirements_path()
        dockerfile_commands = [
            f"FROM python:{python_version}-slim-bullseye",
            "COPY /modal_requirements.txt /modal_requirements.txt",
            "RUN apt-get update",
            "RUN apt-get install -y gcc gfortran build-essential",
            "RUN pip install --upgrade pip",
            "RUN pip install -r /modal_requirements.txt",
            # Set debian front-end to non-interactive to avoid users getting stuck with input
            # prompts.
            "RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections",
        ]

        return _Image(
            dockerfile_commands=dockerfile_commands,
            context_files={"/modal_requirements.txt": requirements_path},
        )

    def apt_install(
        self,
        *packages: Union[str, list[str]],  # A list of packages, e.g. ["ssh", "libpq-dev"]
    ) -> "_Image":
        """Install a list of Debian packages using `apt`."""
        pkgs = _flatten_str_args("apt_install", "packages", packages)
        if not pkgs:
            return self

        package_args = " ".join(shlex.quote(pkg) for pkg in pkgs)

        dockerfile_commands = [
            "FROM base",
            "RUN apt-get update",
            f"RUN apt-get install -y {package_args}",
        ]

        return self.extend(dockerfile_commands=dockerfile_commands)

    def run_function(
        self,
        raw_function: Callable[[], Any],
        **kwargs,
    ) -> "_Image":
        """Run user-defined function `raw_function` as an image build step. The function runs just like an ordinary Modal
        function, and any kwargs accepted by `@stub.function` (such as `Mount`s, `SharedVolume`s, and resource requests) can
        be supplied to it. After it finishes execution, a snapshot of the resulting container file system is saved as an image.

        **Note**

        Only the source code of `raw_function` and the contents of `**kwargs` are used to determine whether the image has changed
        and needs to be rebuilt. If this function references other functions or variables, the image will not be rebuilt if you
        make changes to them. You can force a rebuild by changing the function's source code itself.

        **Example**

        ```python notest

        def my_build_function():
            open("model.pt", "w").write("parameters!")

        image = (
            modal.Image
                .debian_slim()
                .pip_install("torch")
                .run_function(my_build_function, secrets=[...], mounts=[...])
        )
        ```
        """
        return self.extend(build_function=(raw_function, kwargs))


synchronize_apis(_ImageHandle)
Image, AioImage = synchronize_apis(_Image)
