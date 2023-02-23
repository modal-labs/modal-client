# Copyright Modal Labs 2022
from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Union

import toml

from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_apis
from modal_utils.grpc_utils import retry_transient_errors

from . import is_local
from ._function_utils import FunctionInfo
from ._resolver import Resolver
from .config import config, logger
from .exception import InvalidError, NotFoundError, RemoteError
from .gpu import GPU_T, parse_gpu_config
from .mount import _get_client_mount, _Mount
from .object import Handle, Provider
from .secret import _Secret
from .shared_volume import _SharedVolume


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


class _ImageRegistryConfig:
    """mdmd:hidden"""

    def __init__(
        self, registry_type: api_pb2.RegistryType.V = api_pb2.RegistryType.DOCKERHUB, secret: Optional[_Secret] = None
    ):
        self.registry_type = registry_type
        self.secret = secret

    async def resolve(self, resolver: Resolver) -> api_pb2.ImageRegistryConfig:
        if not self.secret:
            return api_pb2.ImageRegistryConfig(registry_type=self.registry_type)

        return api_pb2.ImageRegistryConfig(
            registry_type=self.registry_type, secret_id=(await resolver.load(self.secret)).object_id
        )


class _Image(Provider[_ImageHandle]):
    """Base class for container images to run functions in.

    Do not construct this class directly; instead use one of its static factory methods,
    such as `modal.Image.debian_slim`, `modal.Image.from_dockerhub`, or `modal.Image.conda`.
    """

    @staticmethod
    def _from_args(
        base_images={},
        context_files={},
        dockerfile_commands: Union[list[str], Callable[[], list[str]]] = [],
        secrets: Collection[_Secret] = [],
        ref=None,
        gpu_config: api_pb2.GPUConfig = api_pb2.GPUConfig(),
        build_function=None,
        context_mount: Optional[_Mount] = None,
        image_registry_config: _ImageRegistryConfig = _ImageRegistryConfig(),
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

        for secret in secrets:
            if not isinstance(secret, _Secret):
                raise InvalidError(f"Secret {secret!r} must be a modal.Secret object")

        if context_mount is not None and not isinstance(context_mount, _Mount):
            raise InvalidError(f"Context mount {context_mount!r} must be a modal.Mount object")

        async def _load(resolver: Resolver, existing_object_id: str):
            if ref:
                image_id = (await resolver.load(ref)).object_id
                return _ImageHandle._from_id(image_id, resolver.client, None)

            # Recursively build base images
            base_image_ids: list[str] = []
            for image in base_images.values():
                base_image_ids.append((await resolver.load(image)).object_id)
            base_images_pb2s = [
                api_pb2.BaseImage(
                    docker_tag=docker_tag,
                    image_id=image_id,
                )
                for docker_tag, image_id in zip(base_images.keys(), base_image_ids)
            ]

            secret_ids = []
            for secret in secrets:
                secret_id = (await resolver.load(secret)).object_id
                secret_ids.append(secret_id)

            context_file_pb2s = []
            for filename, path in context_files.items():
                with open(path, "rb") as f:
                    context_file_pb2s.append(api_pb2.ImageContextFile(filename=filename, data=f.read()))

            if build_function:
                build_function_def = build_function.get_build_def()
                build_function_id = (await resolver.load(build_function)).object_id
            else:
                build_function_def = None
                build_function_id = None

            dockerfile_commands_list: list[str]
            if callable(dockerfile_commands):
                # It's a closure (see DockerfileImage)
                dockerfile_commands_list = dockerfile_commands()
            else:
                dockerfile_commands_list = dockerfile_commands

            if context_mount:
                context_mount_id = (await resolver.load(context_mount)).object_id
            else:
                context_mount_id = None

            image_definition = api_pb2.Image(
                base_images=base_images_pb2s,
                dockerfile_commands=dockerfile_commands_list,
                context_files=context_file_pb2s,
                secret_ids=secret_ids,
                gpu=bool(gpu_config.type),  # Note: as of 2023-01-27, server still uses this
                build_function_def=build_function_def,
                context_mount_id=context_mount_id,
                gpu_config=gpu_config,  # Note: as of 2023-01-27, server ignores this
                image_registry_config=await image_registry_config.resolve(
                    resolver
                ),  # Resolves private registry secret.
            )

            req = api_pb2.ImageGetOrCreateRequest(
                app_id=resolver.app_id,
                image=image_definition,
                existing_image_id=existing_object_id,  # TODO: ignored
                build_function_id=build_function_id,
            )
            resp = await resolver.client.stub.ImageGetOrCreate(req)
            image_id = resp.image_id

            logger.debug("Waiting for image %s" % image_id)
            while True:
                request = api_pb2.ImageJoinRequest(image_id=image_id, timeout=55)
                response = await retry_transient_errors(resolver.client.stub.ImageJoin, request)
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

            return _ImageHandle._from_id(image_id, resolver.client, None)

        rep = f"Image({dockerfile_commands})"
        return _Image._from_loader(_load, rep)

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

        return _Image._from_args(base_images={"base": self}, **kwargs)

    def copy(self, mount: _Mount, remote_path: Union[str, Path] = "."):
        """Copy the entire contents of a `modal.Mount` into an image.
        Useful when files only available locally are required during the image
        build process.

        **Example**

        ```python
        static_images_dir = "./static"
        # place all static images in root of mount
        mount = modal.Mount.from_local_dir(static_images_dir, remote_path="/")
        # place mount's contents into /static directory of image.
        image = modal.Image.debian_slim().copy(mount, remote_path="/static")
        ```
        """
        return self.extend(
            dockerfile_commands=["FROM base", f"COPY . {remote_path}"],  # copy everything from the supplied mount
            context_mount=mount,
        )

    def pip_install(
        self,
        *packages: Union[str, list[str]],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
        find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
        index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
        extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
        pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
    ) -> "_Image":
        """Install a list of Python packages using pip.

        **Example**

        ```python
        image = modal.Image.debian_slim().pip_install("click", "httpx~=0.23.3")
        ```
        """
        pkgs = _flatten_str_args("pip_install", "packages", packages)
        if not pkgs:
            return self

        flags = [
            ("--find-links", find_links),  # TODO(erikbern): allow multiple?
            ("--index-url", index_url),
            ("--extra-index-url", extra_index_url),  # TODO(erikbern): allow multiple?
        ]
        extra_args = " ".join(flag + " " + shlex.quote(value) for flag, value in flags if value is not None)
        if pre:
            extra_args += " --pre"
        package_args = " ".join(shlex.quote(pkg) for pkg in pkgs)

        dockerfile_commands = [
            "FROM base",
            f"RUN python -m pip install {package_args} {extra_args}",
            # TODO(erikbern): if extra_args is empty, we add a superfluous space at the end.
            # However removing it at this point would cause image hashes to change.
            # Maybe let's remove it later when/if client requirements change.
        ]

        return self.extend(dockerfile_commands=dockerfile_commands)

    def pip_install_private_repos(
        self,
        *repositories: str,
        git_user: str,
        secrets: Collection[_Secret] = [],
    ) -> "_Image":
        """
        Install a list of Python packages from private git repositories using pip.

        This method currently supports Github and Gitlab only.

        - **Github:** Provide a `modal.Secret` that contains a `GITHUB_TOKEN` key-value pair
        - **Gitlab:** Provide a `modal.Secret` that contains a `GITLAB_TOKEN` key-value pair

        These API tokens should have permissions to read the list of private repositories provided as arguments.

        We recommend using Github's ['fine-grained' access tokens](https://github.blog/2022-10-18-introducing-fine-grained-personal-access-tokens-for-github/).
        These tokens are repo-scoped, and avoid granting read permission across all of a user's private repos.

        **Example**

        ```python
        image = (
            modal.Image
            .debian_slim()
            .pip_install_private_repos(
                "github.com/ecorp/private-one@1.0.0",
                "github.com/ecorp/private-two@main"
                "github.com/ecorp/private-three@d4776502"
                # install from 'inner' directory on default branch.
                "github.com/ecorp/private-four#subdirectory=inner",
                git_user="erikbern",
                secrets=[modal.Secret.from_name("github-read-private")],
            )
        )
        ```
        """
        if not secrets:
            raise InvalidError(
                "No secrets provided to function. Installing private packages requires tokens to be passed via modal.Secret objects."
            )

        invalid_repos = []
        install_urls = []
        for repo_ref in repositories:
            if not isinstance(repo_ref, str):
                invalid_repos.append(repo_ref)
            parts = repo_ref.split("/")
            if parts[0] == "github.com":
                install_urls.append(f"git+https://{git_user}:$GITHUB_TOKEN@{repo_ref}")
            elif parts[0] == "gitlab.com":
                install_urls.append(f"git+https://{git_user}:$GITLAB_TOKEN@{repo_ref}")
            else:
                invalid_repos.append(repo_ref)

        if invalid_repos:
            raise InvalidError(
                f"{len(invalid_repos)} out of {len(repositories)} given repository refs are invalid. "
                f"Invalid refs: {invalid_repos}. "
            )

        secret_names = ",".join([s.app_name if hasattr(s, "app_name") else str(s) for s in secrets])  # type: ignore
        dockerfile_commands = ["FROM base"]
        if any(r.startswith("github") for r in repositories):
            dockerfile_commands.append(
                f"RUN bash -c \"[[ -v GITHUB_TOKEN ]] || (echo 'GITHUB_TOKEN env var not set by provided modal.Secret(s): {secret_names}' && exit 1)\"",
            )
        if any(r.startswith("gitlab") for r in repositories):
            dockerfile_commands.append(
                f"RUN bash -c \"[[ -v GITLAB_TOKEN ]] || (echo 'GITLAB_TOKEN env var not set by provided modal.Secret(s): {secret_names}' && exit 1)\"",
            )

        dockerfile_commands.extend(["RUN apt-get update && apt-get install -y git"])
        dockerfile_commands.extend([f"RUN python3 -m pip install {url}" for url in install_urls])
        return self.extend(
            dockerfile_commands=dockerfile_commands,
            secrets=secrets,
        )

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
        optional_dependencies: List[str] = [],
    ):
        """Install dependencies specified by a `pyproject.toml` file.

        When `optional_dependencies`, a list of the keys of the
        optional-dependencies section(s) of the `pyproject.toml` file
        (e.g. test, doc, experiment, etc), is provided,
        all of those packages in each section are installed as well."""
        from modal import is_local

        # Don't re-run inside container.
        if not is_local():
            return []

        pyproject_toml = os.path.expanduser(pyproject_toml)

        config = toml.load(pyproject_toml)

        dependencies = []
        dependencies.extend(config["project"]["dependencies"])
        if optional_dependencies:
            optionals = config["project"]["optional-dependencies"]
            for dep_group_name in optional_dependencies:
                if dep_group_name in optionals:
                    dependencies.extend(optionals[dep_group_name])

        return self.pip_install(*dependencies)

    def poetry_install_from_file(
        self,
        poetry_pyproject_toml: str,
        poetry_lockfile: Optional[
            str
        ] = None,  # Path to the lockfile. If not provided, uses poetry.lock in the same directory.
        ignore_lockfile: bool = False,  # If set to True, it will not use poetry.lock
        old_installer: bool = False,  # If set to True, use old installer. See https://github.com/python-poetry/poetry/issues/3336
    ):
        """Install poetry *dependencies* specified by a pyproject.toml file.

        The path to the lockfile is inferred, if not provided. However, the
        file has to exist, unless `ignore_lockfile` is set to `True`.

        Note that the root project of the poetry project is not installed,
        only the dependencies. For including local packages see `modal.create_package_mounts`
        """
        if not is_local():
            # existence checks can fail in global scope of the containers
            return

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
                    raise NotFoundError(f"poetry.lock not found at inferred location, {p}")
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
        gpu: GPU_T = None,
        context_mount: Optional[
            _Mount
        ] = None,  # modal.Mount with local files to supply as build context for COPY commands
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
            gpu_config=parse_gpu_config(gpu, warn_on_true=False),
            context_mount=context_mount,
        )

    def run_commands(
        self,
        *commands: Union[str, list[str]],
        secrets: Collection[_Secret] = [],
        gpu: GPU_T = None,
    ):
        """Extend an image with a list of shell commands to run."""
        cmds = _flatten_str_args("run_commands", "commands", commands)
        if not cmds:
            return self

        dockerfile_commands = ["FROM base"] + [f"RUN {cmd}" for cmd in cmds]

        return self.extend(
            dockerfile_commands=dockerfile_commands,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu, warn_on_true=False),
        )

    @staticmethod
    def conda(python_version: str = "3.9") -> "_Image":
        """A Conda base image, using miniconda3 and derived from the official Docker Hub image."""
        _validate_python_version(python_version)
        requirements_path = _get_client_requirements_path()
        # Doesn't use the official continuumio/miniconda3 image as a base. That image has maintenance
        # issues (https://github.com/ContinuumIO/docker-images/issues) and building our own is more flexible.
        conda_install_script = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
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

        return _Image._from_args(
            dockerfile_commands=dockerfile_commands,
            context_files={"/modal_requirements.txt": requirements_path},
        ).dockerfile_commands(
            [
                "ENV CONDA_EXE=/usr/local/bin/conda",
                "ENV CONDA_PREFIX=/usr/local",
                "ENV CONDA_PROMPT_MODIFIER=(base)",
                "ENV CONDA_SHLVL=1",
                "ENV CONDA_PYTHON_EXE=/usr/local/bin/python",
                "ENV CONDA_DEFAULT_ENV=base",
            ]
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
    def _registry_setup_commands(tag: str, setup_commands: list[str]) -> list[str]:
        """mdmd:hidden"""
        return [
            f"FROM {tag}",
            *(f"RUN {cmd}" for cmd in setup_commands),
            "COPY /modal_requirements.txt /modal_requirements.txt",
            "RUN python -m pip install --upgrade pip",
            "RUN python -m pip install -r /modal_requirements.txt",
        ]

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

        **Example**

        ```python
        modal.Image.from_dockerhub(
          "gisops/valhalla:latest",
          setup_commands=["apt-get update", "apt-get install -y python3-pip"]
        )
        ```
        """
        requirements_path = _get_client_requirements_path()

        dockerfile_commands = _Image._registry_setup_commands(tag, setup_commands)

        return _Image._from_args(
            dockerfile_commands=dockerfile_commands,
            context_files={"/modal_requirements.txt": requirements_path},
            **kwargs,
        )

    @staticmethod
    def from_aws_ecr(tag: str, secret: Optional[_Secret] = None, setup_commands: list[str] = [], **kwargs) -> "_Image":
        """
        Build a Modal image from a pre-existing image on a private AWS Elastic
        Container Registry (ECR). You will need to pass a `modal.Secret` containing
        an AWS key (`AWS_ACCESS_KEY_ID`) and secret (`AWS_SECRET_ACCESS_KEY`)
        with permissions to access the target ECR registry.

        Refer to ["Private repository policies"](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-policies.html)
        for details about IAM configuration.

        The same assumptions hold from `from_dockerhub`:

        - Python 3.7 or above is present, and is available as `python`.
        - `pip` is installed correctly.
        - The image is built for the `linux/amd64` platform.

        You can use the `setup_commands` argument to run any
        commands in the image before Modal is installed.
        This might be useful if Python or pip is not installed.

        **Example**

        ```python
        modal.Image.from_aws_ecr(
          "000000000000.dkr.ecr.us-east-1.amazonaws.com/my-private-registry:my-version",
          secret=modal.Secret.from_name("aws"),
          setup_commands=["apt-get update", "apt-get install -y python3-pip"]
        )
        ```
        """
        requirements_path = _get_client_requirements_path()

        dockerfile_commands = _Image._registry_setup_commands(tag, setup_commands)

        return _Image._from_args(
            dockerfile_commands=dockerfile_commands,
            context_files={"/modal_requirements.txt": requirements_path},
            image_registry_config=_ImageRegistryConfig(api_pb2.RegistryType.ECR, secret),
            **kwargs,
        )

    @staticmethod
    def from_dockerfile(
        path: Union[str, Path],
        context_mount: Optional[
            _Mount
        ] = None,  # modal.Mount with local files to supply as build context for COPY commands
    ) -> "_Image":
        """Build a Modal image from a local Dockerfile.

        Note that the following must be true about the image you provide:

        - Python 3.7 or above needs to be present and available as `python`.
        - `pip` needs to be installed and available as `pip`.
        """

        path = os.path.expanduser(path)

        def base_dockerfile_commands():
            # Make it a closure so that it's only invoked locally
            with open(path) as f:
                return f.read().split("\n")

        base_image = _Image._from_args(dockerfile_commands=base_dockerfile_commands, context_mount=context_mount)

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

        return _Image._from_args(
            dockerfile_commands=dockerfile_commands,
            context_files={"/modal_requirements.txt": requirements_path},
        )

    def apt_install(
        self,
        *packages: Union[str, list[str]],  # A list of packages, e.g. ["ssh", "libpq-dev"]
    ) -> "_Image":
        """Install a list of Debian packages using `apt`.

        **Example**

        ```python
        image = modal.Image.debian_slim().apt_install("git")
        ```
        """
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
        raw_f: Callable[[], Any],
        *,
        secret: Optional[_Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Collection[_Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        mounts: Collection[_Mount] = (),
        shared_volumes: Dict[str, _SharedVolume] = {},
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MB. This is a soft limit.
        timeout: Optional[int] = 86400,  # Maximum execution time of the function in seconds.
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, auto.
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
        from .functions import _Function, _FunctionHandle

        info = FunctionInfo(raw_f)
        base_mounts = [_get_client_mount()]
        for key, mount in info.get_mounts().items():
            base_mounts.append(mount)

        function_handle = _FunctionHandle._new()

        function = _Function(
            function_handle,
            info,
            _stub=self,
            image=self,
            secret=secret,
            secrets=secrets,
            gpu=gpu,
            base_mounts=base_mounts,
            mounts=mounts,
            shared_volumes=shared_volumes,
            memory=memory,
            timeout=timeout,
            cpu=cpu,
            cloud=cloud,
        )
        return self.extend(build_function=function)

    def env(self, vars: dict[str, str]) -> "_Image":
        """Sets the environmental variables of the image.

        **Example**

        ```python
        image = (
            modal.Image.conda()
                .env({"CONDA_OVERRIDE_CUDA": "11.2"})
                .conda_install("jax", "cuda-nvcc", channels=["conda-forge", "nvidia"])
                .pip_install("dm-haiku", "optax")
        )
        ```
        """
        return self.extend(
            dockerfile_commands=["FROM base"] + [f"ENV {key}={shlex.quote(val)}" for (key, val) in vars.items()]
        )


synchronize_apis(_ImageHandle)
Image, AioImage = synchronize_apis(_Image)
