# Copyright Modal Labs 2022
import contextlib
import os
import shlex
import sys
import typing
import warnings
from datetime import date
from inspect import isfunction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import toml
from google.protobuf.message import Message
from grpclib.exceptions import GRPCError, StreamTerminatedError

from modal._types import typechecked
from modal_proto import api_pb2
from modal_utils.async_utils import synchronize_api
from modal_utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors, unary_stream

from ._function_utils import FunctionInfo
from ._resolver import Resolver
from ._serialization import serialize
from .app import is_local
from .config import config, logger
from .exception import InvalidError, NotFoundError, RemoteError, deprecation_error
from .gpu import GPU_T, parse_gpu_config
from .mount import _Mount, python_standalone_mount_name
from .network_file_system import _NetworkFileSystem
from .object import _Object
from .secret import _Secret

_from_dockerhub_deprecation_msg = "`Image.from_dockerhub` is deprecated. Use `Image.from_registry` instead."


def _validate_python_version(version: str) -> None:
    components = version.split(".")
    supported_versions = {"3.11", "3.10", "3.9", "3.8", "3.7"}
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

    parts = python_version.split(".")

    if len(parts) > 2:
        return python_version

    # We use the same major/minor version, but the highest micro version
    # See https://hub.docker.com/_/python
    latest_micro_version = {
        "3.11": "0",
        "3.10": "8",
        "3.9": "15",
        "3.8": "15",
        "3.7": "15",
    }
    major_minor_version = ".".join(parts[:2])
    python_version = major_minor_version + "." + latest_micro_version[major_minor_version]
    return python_version


def _get_client_requirements_path():
    # Locate Modal client requirements.txt
    import modal

    modal_path = modal.__path__[0]
    return os.path.join(modal_path, "requirements.txt")


def _flatten_str_args(function_name: str, arg_name: str, args: Tuple[Union[str, List[str]], ...]) -> List[str]:
    """Takes a tuple of strings, or string lists, and flattens it.

    Raises an error if any of the elements are not strings or string lists.
    """
    # TODO(erikbern): maybe we can just build somthing intelligent that checks
    # based on type annotations in real time?
    # Or use something like this? https://github.com/FelixTheC/strongtyping

    def is_str_list(x):
        return isinstance(x, list) and all(isinstance(y, str) for y in x)

    ret: List[str] = []
    for x in args:
        if isinstance(x, str):
            ret.append(x)
        elif is_str_list(x):
            ret.extend(x)
        else:
            raise InvalidError(f"{function_name}: {arg_name} must only contain strings")
    return ret


def _make_pip_install_args(
    find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
    index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
    extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
    pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
) -> str:
    flags = [
        ("--find-links", find_links),  # TODO(erikbern): allow multiple?
        ("--index-url", index_url),
        ("--extra-index-url", extra_index_url),  # TODO(erikbern): allow multiple?
    ]

    args = " ".join(flag + " " + shlex.quote(value) for flag, value in flags if value is not None)
    if pre:
        args += " --pre"

    return args


class _ImageRegistryConfig:
    """mdmd:hidden"""

    def __init__(
        self,
        registry_type: int = api_pb2.RegistryType.DOCKERHUB,
        secret: Optional[_Secret] = None,
    ):
        self.registry_type = registry_type
        self.secret = secret

    def get_proto(self) -> api_pb2.ImageRegistryConfig:
        return api_pb2.ImageRegistryConfig(
            registry_type=self.registry_type,
            secret_id=(self.secret.object_id if self.secret else None),
        )


if typing.TYPE_CHECKING:
    import modal.functions


class _Image(_Object, type_prefix="im"):
    """Base class for container images to run functions in.

    Do not construct this class directly; instead use one of its static factory methods,
    such as `modal.Image.debian_slim`, `modal.Image.from_registry`, or `modal.Image.conda`.
    """

    force_build: bool
    inside_exceptions: List[Exception]

    def _initialize_from_empty(self):
        self.inside_exceptions = []

    def _hydrate_metadata(self, message: Optional[Message]):
        env_image_id = config.get("image_id")
        if env_image_id == self.object_id:
            for exc in self.inside_exceptions:
                raise exc

    @staticmethod
    @typechecked
    def _from_args(
        base_images={},
        context_files={},
        dockerfile_commands: Union[List[str], Callable[[], List[str]]] = [],
        secrets: Sequence[_Secret] = [],
        gpu_config: Optional[api_pb2.GPUConfig] = None,
        build_function: Optional["modal.functions._Function"] = None,
        context_mount: Optional[_Mount] = None,
        image_registry_config: Optional[_ImageRegistryConfig] = None,
        force_build: bool = False,
        # For internal use only.
        _namespace: int = api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
    ):
        if gpu_config is None:
            gpu_config = api_pb2.GPUConfig()
        if image_registry_config is None:
            image_registry_config = _ImageRegistryConfig()

        if not dockerfile_commands and not build_function:
            raise InvalidError(
                "No commands were provided for the image — have you tried using modal.Image.debian_slim()?"
            )
        for secret in secrets:
            if not isinstance(secret, _Secret):
                raise InvalidError("All secrets of an image needs to be modal.Secret/AioSecret instances")

        if build_function and dockerfile_commands:
            raise InvalidError("Cannot provide both a build function and Dockerfile commands!")

        if build_function and len(base_images) != 1:
            raise InvalidError("Cannot run a build function with multiple base images!")

        def _deps() -> List[_Object]:
            deps: List[_Object] = list(base_images.values()) + list(secrets)
            if build_function:
                deps.append(build_function)
            if context_mount:
                deps.append(context_mount)
            if image_registry_config.secret:
                deps.append(image_registry_config.secret)
            return deps

        async def _load(provider: _Image, resolver: Resolver, existing_object_id: Optional[str]):
            base_images_pb2s = [
                api_pb2.BaseImage(
                    docker_tag=docker_tag,
                    image_id=image.object_id,
                )
                for docker_tag, image in base_images.items()
            ]

            context_file_pb2s = []
            for filename, path in context_files.items():
                with open(path, "rb") as f:
                    context_file_pb2s.append(api_pb2.ImageContextFile(filename=filename, data=f.read()))

            if build_function:
                build_function_def = build_function.get_build_def()
                build_function_id = build_function.object_id

                globals = build_function._get_info().get_globals()
                filtered_globals = {}
                for k, v in globals.items():
                    if isfunction(v):
                        continue
                    try:
                        serialize(v)
                    except Exception:
                        # Skip unserializable values for now.
                        logger.warning(
                            f"Skipping unserializable global variable {k} for {build_function._get_info().function_name}. Changes to this variable won't invalidate the image."
                        )
                        continue
                    filtered_globals[k] = v

                # Cloudpickle function serialization produces unstable values.
                # TODO: better way to filter out types that don't have a stable hash?
                build_function_globals = serialize(filtered_globals) if filtered_globals else None
            else:
                build_function_def = None
                build_function_id = None
                build_function_globals = None

            dockerfile_commands_list: List[str]
            if callable(dockerfile_commands):
                # It's a closure (see DockerfileImage)
                dockerfile_commands_list = dockerfile_commands()
            else:
                dockerfile_commands_list = dockerfile_commands

            image_definition = api_pb2.Image(
                base_images=base_images_pb2s,
                dockerfile_commands=dockerfile_commands_list,
                context_files=context_file_pb2s,
                secret_ids=[secret.object_id for secret in secrets],
                gpu=bool(gpu_config.type),  # Note: as of 2023-01-27, server still uses this
                build_function_def=build_function_def,
                build_function_globals=build_function_globals,
                context_mount_id=(context_mount.object_id if context_mount else None),
                gpu_config=gpu_config,  # Note: as of 2023-01-27, server ignores this
                image_registry_config=image_registry_config.get_proto(),
                runtime=config.get("function_runtime"),
                runtime_debug=config.get("function_runtime_debug"),
            )

            req = api_pb2.ImageGetOrCreateRequest(
                app_id=resolver.app_id,
                image=image_definition,
                existing_image_id=existing_object_id,  # TODO: ignored
                build_function_id=build_function_id,
                force_build=force_build,
                namespace=_namespace,
            )
            resp = await retry_transient_errors(resolver.client.stub.ImageGetOrCreate, req)
            image_id = resp.image_id

            logger.debug("Waiting for image %s" % image_id)
            last_entry_id: Optional[str] = None
            result: Optional[api_pb2.GenericResult] = None

            async def join():
                nonlocal last_entry_id, result

                request = api_pb2.ImageJoinStreamingRequest(image_id=image_id, timeout=55, last_entry_id=last_entry_id)
                async for response in unary_stream(resolver.client.stub.ImageJoinStreaming, request):
                    if response.entry_id:
                        last_entry_id = response.entry_id
                    if response.result.status:
                        result = response.result
                    for task_log in response.task_logs:
                        if task_log.task_progress.pos or task_log.task_progress.len:
                            assert task_log.task_progress.progress_type == api_pb2.IMAGE_SNAPSHOT_UPLOAD
                            resolver.image_snapshot_update(image_id, task_log.task_progress)
                        elif task_log.data:
                            await resolver.console_write(task_log)
                resolver.console_flush()

            # Handle up to n exceptions while fetching logs
            retry_count = 0
            while result is None:
                try:
                    await join()
                except (StreamTerminatedError, GRPCError) as exc:
                    if isinstance(exc, GRPCError) and exc.status not in RETRYABLE_GRPC_STATUS_CODES:
                        raise exc
                    retry_count += 1
                    if retry_count >= 3:
                        raise exc

            if result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE:
                raise RemoteError(f"Image build for {image_id} failed with the exception:\n{result.exception}")
            elif result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED:
                raise RemoteError(f"Image build for {image_id} terminated due to external shut-down. Please try again.")
            elif result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
                raise RemoteError(
                    f"Image build for {image_id} timed out. Please try again with a larger `timeout` parameter."
                )
            elif result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
                pass
            else:
                raise RemoteError("Unknown status %s!" % result.status)

            provider._hydrate(image_id, resolver.client, None)

        rep = f"Image({dockerfile_commands})"
        obj = _Image._from_loader(_load, rep, deps=_deps)
        obj.force_build = force_build
        return obj

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

    @typechecked
    def copy_mount(self, mount: _Mount, remote_path: Union[str, Path] = ".") -> "_Image":
        """Copy the entire contents of a `modal.Mount` into an image.
        Useful when files only available locally are required during the image
        build process.

        **Example**

        ```python
        static_images_dir = "./static"
        # place all static images in root of mount
        mount = modal.Mount.from_local_dir(static_images_dir, remote_path="/")
        # place mount's contents into /static directory of image.
        image = modal.Image.debian_slim().copy_mount(mount, remote_path="/static")
        ```
        """
        if not isinstance(mount, _Mount):
            raise InvalidError("The mount argument to copy has to be a Modal Mount object")
        return self.extend(
            dockerfile_commands=["FROM base", f"COPY . {remote_path}"],  # copy everything from the supplied mount
            context_mount=mount,
        )

    def copy_local_file(self, local_path: Union[str, Path], remote_path: Union[str, Path] = "./") -> "_Image":
        """Copy a file into the image as a part of building it.

        This works in a similar way to [`COPY`](https://docs.docker.com/engine/reference/builder/#copy) in a `Dockerfile`.
        """
        basename = str(Path(local_path).name)
        mount = _Mount.from_local_file(local_path, remote_path=f"/{basename}")
        return self.extend(
            dockerfile_commands=["FROM base", f"COPY {basename} {remote_path}"],
            context_mount=mount,
        )

    def copy_local_dir(self, local_path: Union[str, Path], remote_path: Union[str, Path] = ".") -> "_Image":
        """Copy a directory into the image as a part of building the image.

        This works in a similar way to [`COPY`](https://docs.docker.com/engine/reference/builder/#copy) in a `Dockerfile`.
        """
        mount = _Mount.from_local_dir(local_path, remote_path="/")
        return self.extend(
            dockerfile_commands=["FROM base", f"COPY . {remote_path}"],
            context_mount=mount,
        )

    @typechecked
    def pip_install(
        self,
        *packages: Union[str, List[str]],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
        find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
        index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
        extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
        pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
        force_build: bool = False,
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
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

        extra_args = _make_pip_install_args(find_links, index_url, extra_index_url, pre)
        package_args = " ".join(shlex.quote(pkg) for pkg in sorted(pkgs))

        dockerfile_commands = [
            "FROM base",
            f"RUN python -m pip install {package_args} {extra_args}",
            # TODO(erikbern): if extra_args is empty, we add a superfluous space at the end.
            # However removing it at this point would cause image hashes to change.
            # Maybe let's remove it later when/if client requirements change.
        ]
        gpu_config = parse_gpu_config(gpu)
        return self.extend(
            dockerfile_commands=dockerfile_commands,
            force_build=self.force_build or force_build,
            gpu_config=gpu_config,
            secrets=secrets,
        )

    @typechecked
    def pip_install_private_repos(
        self,
        *repositories: str,
        git_user: str,
        find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
        index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
        extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
        pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
        gpu: GPU_T = None,
        secrets: Sequence[_Secret] = [],
        force_build: bool = False,
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

        extra_args = _make_pip_install_args(find_links, index_url, extra_index_url, pre)

        dockerfile_commands.extend(["RUN apt-get update && apt-get install -y git"])
        dockerfile_commands.extend([f'RUN python3 -m pip install "{url}" {extra_args}' for url in install_urls])

        gpu_config = parse_gpu_config(gpu)

        return self.extend(
            dockerfile_commands=dockerfile_commands,
            secrets=secrets,
            gpu_config=gpu_config,
            force_build=self.force_build or force_build,
        )

    @typechecked
    def pip_install_from_requirements(
        self,
        requirements_txt: str,  # Path to a requirements.txt file.
        find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
        *,
        index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
        extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
        pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
        force_build: bool = False,
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
    ) -> "_Image":
        """Install a list of Python packages from a local `requirements.txt` file."""

        requirements_txt = os.path.expanduser(requirements_txt)

        find_links_arg = f"-f {find_links}" if find_links else ""
        context_files = {"/.requirements.txt": requirements_txt}

        extra_args = _make_pip_install_args(find_links, index_url, extra_index_url, pre)

        dockerfile_commands = [
            "FROM base",
            "COPY /.requirements.txt /.requirements.txt",
            f"RUN python -m pip install -r /.requirements.txt {find_links_arg} {extra_args}",
        ]

        return self.extend(
            dockerfile_commands=dockerfile_commands,
            context_files=context_files,
            force_build=self.force_build or force_build,
            gpu_config=parse_gpu_config(gpu),
            secrets=secrets,
        )

    @typechecked
    def pip_install_from_pyproject(
        self,
        pyproject_toml: str,
        optional_dependencies: List[str] = [],
        *,
        find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
        index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
        extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
        pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
        force_build: bool = False,
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
    ) -> "_Image":
        """Install dependencies specified by a local `pyproject.toml` file.

        `optional_dependencies` is a list of the keys of the
        optional-dependencies section(s) of the `pyproject.toml` file
        (e.g. test, doc, experiment, etc). When provided,
        all of the packages in each listed section are installed as well."""
        from modal.app import is_local

        # Don't re-run inside container.
        if not is_local():
            return self

        pyproject_toml = os.path.expanduser(pyproject_toml)

        config = toml.load(pyproject_toml)

        dependencies = []
        dependencies.extend(config["project"]["dependencies"])
        if optional_dependencies:
            optionals = config["project"]["optional-dependencies"]
            for dep_group_name in optional_dependencies:
                if dep_group_name in optionals:
                    dependencies.extend(optionals[dep_group_name])

        return self.pip_install(
            *dependencies,
            find_links=find_links,
            index_url=index_url,
            extra_index_url=extra_index_url,
            pre=pre,
            force_build=self.force_build or force_build,
            secrets=secrets,
            gpu=gpu,
        )

    @typechecked
    def poetry_install_from_file(
        self,
        poetry_pyproject_toml: str,
        # Path to the lockfile. If not provided, uses poetry.lock in the same directory.
        poetry_lockfile: Optional[str] = None,
        # If set to True, it will not use poetry.lock
        ignore_lockfile: bool = False,
        # If set to True, use old installer. See https://github.com/python-poetry/poetry/issues/3336
        old_installer: bool = False,
        force_build: bool = False,
        # Selected optional dependency groups to install (See https://python-poetry.org/docs/cli/#install)
        with_: List[str] = [],
        # Selected optional dependency groups to exclude (See https://python-poetry.org/docs/cli/#install)
        without: List[str] = [],
        # Only install dependency groups specifed in this list.
        only: List[str] = [],
        *,
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
    ) -> "_Image":
        """Install poetry *dependencies* specified by a local `pyproject.toml` file.

        If not provided as argument the path to the lockfile is inferred. However, the
        file has to exist, unless `ignore_lockfile` is set to `True`.

        Note that the root project of the poetry project is not installed,
        only the dependencies. For including local packages see `modal.Mount.from_local_python_packages`
        """
        if not is_local():
            # existence checks can fail in global scope of the containers
            return self

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

        # Indentation for back-compat
        install_cmd = "  poetry install --no-root"

        if with_:
            install_cmd += f" --with {','.join(with_)}"

        if without:
            install_cmd += f" --without {','.join(without)}"

        if only:
            install_cmd += f" --only {','.join(only)}"

        dockerfile_commands += [
            "COPY /.pyproject.toml /tmp/poetry/pyproject.toml",
            "RUN cd /tmp/poetry && \\ ",
            "  poetry config virtualenvs.create false && \\ ",
            install_cmd,
        ]

        return self.extend(
            dockerfile_commands=dockerfile_commands,
            context_files=context_files,
            force_build=self.force_build or force_build,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu),
        )

    @typechecked
    def dockerfile_commands(
        self,
        *dockerfile_commands: Union[str, List[str]],
        context_files: Dict[str, str] = {},
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
        # modal.Mount with local files to supply as build context for COPY commands
        context_mount: Optional[_Mount] = None,
        force_build: bool = False,
    ) -> "_Image":
        """Extend an image with arbitrary Dockerfile-like commands."""
        cmds = _flatten_str_args("dockerfile_commands", "dockerfile_commands", dockerfile_commands)
        if not cmds:
            return self

        _dockerfile_commands = ["FROM base", *cmds]

        return self.extend(
            dockerfile_commands=_dockerfile_commands,
            context_files=context_files,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu, raise_on_true=False),
            context_mount=context_mount,
            force_build=self.force_build or force_build,
        )

    @typechecked
    def run_commands(
        self,
        *commands: Union[str, List[str]],
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
        force_build: bool = False,
    ) -> "_Image":
        """Extend an image with a list of shell commands to run."""
        cmds = _flatten_str_args("run_commands", "commands", commands)
        if not cmds:
            return self

        dockerfile_commands = ["FROM base"] + [f"RUN {cmd}" for cmd in cmds]

        return self.extend(
            dockerfile_commands=dockerfile_commands,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu, raise_on_true=False),
            force_build=self.force_build or force_build,
        )

    @staticmethod
    @typechecked
    def conda(python_version: str = "3.9", force_build: bool = False) -> "_Image":
        """
        A Conda base image, using miniconda3 and derived from the official Docker Hub image.
        In most cases, using [`Image.micromamba()`](/docs/reference/modal.Image#micromamba) with [`micromamba_install`](/docs/reference/modal.Image#micromamba_install) is recommended over `Image.conda()`, as it leads to significantly faster image build times.
        """
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
            force_build=force_build,
            _namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
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

    @typechecked
    def conda_install(
        self,
        *packages: Union[str, List[str]],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
        channels: List[str] = [],  # A list of Conda channels, eg. ["conda-forge", "nvidia"]
        force_build: bool = False,
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
    ) -> "_Image":
        """Install a list of additional packages using Conda. Note that in most cases, using [`Image.micromamba()`](/docs/reference/modal.Image#micromamba) with [`micromamba_install`](/docs/reference/modal.Image#micromamba_install)
        is recommended over `conda_install`, as it leads to significantly faster image build times."""

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

        gpu_config = parse_gpu_config(gpu)
        return self.extend(
            dockerfile_commands=dockerfile_commands,
            force_build=self.force_build or force_build,
            secrets=secrets,
            gpu_config=gpu_config,
        )

    @typechecked
    def conda_update_from_environment(
        self,
        environment_yml: str,
        force_build: bool = False,
        *,
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
    ) -> "_Image":
        """Update a Conda environment using dependencies from a given environment.yml file."""

        environment_yml = os.path.expanduser(environment_yml)

        context_files = {"/environment.yml": environment_yml}

        dockerfile_commands = [
            "FROM base",
            "COPY /environment.yml /environment.yml",
            "RUN conda env update --name base -f /environment.yml \\ ",
            "&& conda clean --yes --index-cache --tarballs --tempfiles --logfiles",
        ]

        return self.extend(
            dockerfile_commands=dockerfile_commands,
            context_files=context_files,
            force_build=self.force_build or force_build,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu),
        )

    @staticmethod
    @typechecked
    def micromamba(
        python_version: str = "3.9",
        force_build: bool = False,
    ) -> "_Image":
        """
        A Micromamba base image. Micromamba allows for fast building of small Conda-based containers.
        In most cases it will be faster than using [`Image.conda()`](/docs/reference/modal.Image#conda).
        """
        _validate_python_version(python_version)

        return _Image.from_registry(
            "mambaorg/micromamba:1.3.1-bullseye-slim",
            setup_dockerfile_commands=[
                'SHELL ["/usr/local/bin/_dockerfile_shell.sh"]',
                "ENV MAMBA_DOCKERFILE_ACTIVATE=1",
                f"RUN micromamba install -n base -y python={python_version} pip -c conda-forge",
            ],
            force_build=force_build,
            _namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        )

    @typechecked
    def micromamba_install(
        self,
        # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
        *packages: Union[str, List[str]],
        # A list of Conda channels, eg. ["conda-forge", "nvidia"]
        channels: List[str] = [],
        force_build: bool = False,
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
    ) -> "_Image":
        """Install a list of additional packages using micromamba."""

        pkgs = _flatten_str_args("micromamba_install", "packages", packages)
        if not pkgs:
            return self

        package_args = " ".join(shlex.quote(pkg) for pkg in pkgs)
        channel_args = "".join(f" -c {channel}" for channel in channels)

        dockerfile_commands = [
            "FROM base",
            f"RUN micromamba install {package_args}{channel_args} --yes",
        ]

        return self.extend(
            dockerfile_commands=dockerfile_commands,
            force_build=self.force_build or force_build,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu),
        )

    @staticmethod
    def _registry_setup_commands(
        tag: str, setup_dockerfile_commands: List[str], add_python: Optional[str]
    ) -> List[str]:
        add_python_commands: List[str] = []
        if add_python:
            add_python_commands = [
                "COPY /python/. /usr/local",
                "RUN ln -s /usr/local/bin/python3 /usr/local/bin/python",
                "ENV TERMINFO_DIRS=/etc/terminfo:/lib/terminfo:/usr/share/terminfo:/usr/lib/terminfo",
            ]
        return [
            f"FROM {tag}",
            *add_python_commands,
            *setup_dockerfile_commands,
            "COPY /modal_requirements.txt /modal_requirements.txt",
            "RUN python -m pip install --upgrade pip",
            "RUN python -m pip install -r /modal_requirements.txt",
            # TODO: We should add this next line at some point to clean up the image, but it would
            # trigger a hash change, so batch it with the next rebuild-triggering change.
            #
            # "RUN rm /modal_requirements.txt",
        ]

    @staticmethod
    @typechecked
    def from_registry(
        tag: str,
        *,
        setup_dockerfile_commands: List[str] = [],
        force_build: bool = False,
        add_python: Optional[str] = None,
        **kwargs,
    ) -> "_Image":
        """Build a Modal image from a public image registry, such as Docker Hub.

        The image must be built for the `linux/amd64` platform and have Python 3.7 or above
        installed and available on PATH as `python`. It should also have `pip`.

        If your image does not come with Python installed, you can use the `add_python` parameter
        to specify a version of Python to add to the image. Supported versions are `3.8`, `3.9`,
        `3.10`, and `3.11`. For Alpine-based images, use `3.8-musl` through `3.11-musl`, which
        are statically-linked Python installations.

        You may also use `setup_dockerfile_commands` to run Dockerfile commands before the
        remaining commands run. This might be useful if you want a custom Python installation or to
        set a `SHELL`. Prefer `run_commands()` when possible though.

        **Examples**

        ```python
        modal.Image.from_registry("python:3.11-slim-bookworm")
        modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
        modal.Image.from_registry("alpine:3.18.3", add_python="3.11-musl")
        ```
        """
        requirements_path = _get_client_requirements_path()
        dockerfile_commands = _Image._registry_setup_commands(tag, setup_dockerfile_commands, add_python)

        context_mount = None
        if add_python:
            context_mount = _Mount.from_name(
                python_standalone_mount_name(add_python),
                namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
            )

        return _Image._from_args(
            dockerfile_commands=dockerfile_commands,
            context_mount=context_mount,
            context_files={"/modal_requirements.txt": requirements_path},
            force_build=force_build,
            **kwargs,
        )

    @staticmethod
    @typechecked
    def from_dockerhub(
        tag: str,
        setup_dockerfile_commands: List[str] = [],
        force_build: bool = False,
        **kwargs,
    ):
        f"""{_from_dockerhub_deprecation_msg}"""
        deprecation_error(date(2023, 8, 25), _from_dockerhub_deprecation_msg)

    @staticmethod
    @typechecked
    def from_gcp_artifact_registry(
        tag: str,
        secret: Optional[_Secret] = None,
        *,
        setup_dockerfile_commands: List[str] = [],
        force_build: bool = False,
        add_python: Optional[str] = None,
        **kwargs,
    ) -> "_Image":
        """Build a Modal image from a private image in GCP Artifact Registry.

        You will need to pass a `modal.Secret` containing your GCP service account key
        as `SERVICE_ACCOUNT_JSON`. This can be done from the [Secrets](/secrets) page.
        The service account needs to have at least an "Artifact Registry Reader" role.

        See `Image.from_registry()` for information about the other parameters.

        **Example**

        ```python
        modal.Image.from_gcp_artifact_registry(
            "us-east1-docker.pkg.dev/my-project-1234/my-repo/my-image:my-version",
            secret=modal.Secret.from_name("my-gcp-secret"),
            add_python="3.11",
        )
        ```
        """
        image_registry_config = _ImageRegistryConfig(api_pb2.RegistryType.GCP_ARTIFACT_REGISTRY, secret)
        return _Image.from_registry(
            tag,
            setup_dockerfile_commands=setup_dockerfile_commands,
            force_build=force_build,
            add_python=add_python,
            image_registry_config=image_registry_config,
            **kwargs,
        )

    @staticmethod
    @typechecked
    def from_aws_ecr(
        tag: str,
        secret: Optional[_Secret] = None,
        *,
        setup_dockerfile_commands: List[str] = [],
        force_build: bool = False,
        add_python: Optional[str] = None,
        **kwargs,
    ) -> "_Image":
        """Build a Modal image from a private image in AWS Elastic Container Registry (ECR).

        You will need to pass a `modal.Secret` containing an AWS key (`AWS_ACCESS_KEY_ID`) and
        secret (`AWS_SECRET_ACCESS_KEY`) with permissions to access the target ECR registry.

        IAM configuration details can be found in the AWS documentation for
        ["Private repository policies"](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-policies.html).

        See `Image.from_registry()` for information about the other parameters.

        **Example**

        ```python
        modal.Image.from_aws_ecr(
            "000000000000.dkr.ecr.us-east-1.amazonaws.com/my-private-registry:my-version",
            secret=modal.Secret.from_name("aws"),
            add_python="3.11",
        )
        ```
        """
        image_registry_config = _ImageRegistryConfig(api_pb2.RegistryType.ECR, secret)
        return _Image.from_registry(
            tag,
            setup_dockerfile_commands=setup_dockerfile_commands,
            force_build=force_build,
            add_python=add_python,
            image_registry_config=image_registry_config,
            **kwargs,
        )

    @staticmethod
    @typechecked
    def from_dockerfile(
        path: Union[str, Path],
        context_mount: Optional[
            _Mount
        ] = None,  # modal.Mount with local files to supply as build context for COPY commands
        force_build: bool = False,
        *,
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
        add_python: Optional[str] = None,
    ) -> "_Image":
        """Build a Modal image from a local Dockerfile.

        If your Dockerfile does not have Python installed, you can use the `add_python` parameter
        to specify a version of Python to add to the image. Supported versions are `3.8`, `3.9`,
        `3.10`, and `3.11`. For Alpine-based images, use `3.8-musl` through `3.11-musl`, which
        are statically-linked Python installations.

        **Example**

        ```python
        image = modal.Image.from_dockerfile("./Dockerfile", add_python="3.10")
        ```
        """

        path = os.path.expanduser(path)

        def base_dockerfile_commands():
            # Make it a closure so that it's only invoked locally
            with open(path) as f:
                return f.read().split("\n")

        gpu_config = parse_gpu_config(gpu)
        base_image = _Image._from_args(
            dockerfile_commands=base_dockerfile_commands,
            context_mount=context_mount,
            gpu_config=gpu_config,
            secrets=secrets,
        )

        requirements_path = _get_client_requirements_path()

        context_mount = None
        add_python_commands = []
        if add_python:
            add_python_commands = [
                "COPY /python/. /usr/local",
                "RUN ln -s /usr/local/bin/python3 /usr/local/bin/python",
                "ENV TERMINFO_DIRS=/etc/terminfo:/lib/terminfo:/usr/share/terminfo:/usr/lib/terminfo",
            ]
            context_mount = _Mount.from_name(
                python_standalone_mount_name(add_python),
                namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
            )

        dockerfile_commands = [
            "FROM base",
            *add_python_commands,
            "COPY /modal_requirements.txt /modal_requirements.txt",
            "RUN python -m pip install --upgrade pip",
            "RUN python -m pip install -r /modal_requirements.txt",
        ]

        return base_image.extend(
            dockerfile_commands=dockerfile_commands,
            context_files={"/modal_requirements.txt": requirements_path},
            context_mount=context_mount,
            force_build=force_build,
        )

    @staticmethod
    @typechecked
    def debian_slim(python_version: Optional[str] = None, force_build: bool = False) -> "_Image":
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
            force_build=force_build,
            _namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        )

    @typechecked
    def apt_install(
        self,
        *packages: Union[str, List[str]],  # A list of packages, e.g. ["ssh", "libpq-dev"]
        force_build: bool = False,
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
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

        return self.extend(
            dockerfile_commands=dockerfile_commands,
            force_build=self.force_build or force_build,
            gpu_config=parse_gpu_config(gpu),
            secrets=secrets,
        )

    @typechecked
    def run_function(
        self,
        raw_f: Callable[[], Any],
        *,
        secret: Optional[_Secret] = None,  # An optional Modal Secret with environment variables for the container
        secrets: Sequence[_Secret] = (),  # Plural version of `secret` when multiple secrets are needed
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        mounts: Sequence[_Mount] = (),
        shared_volumes: Dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
        network_file_systems: Dict[Union[str, os.PathLike], _NetworkFileSystem] = {},
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        timeout: Optional[int] = 86400,  # Maximum execution time of the function in seconds.
        force_build: bool = False,
    ) -> "_Image":
        """Run user-defined function `raw_function` as an image build step. The function runs just like an ordinary Modal
        function, and any kwargs accepted by `@stub.function` (such as `Mount`s, `NetworkFileSystem`s, and resource requests) can
        be supplied to it. After it finishes execution, a snapshot of the resulting container file system is saved as an image.

        **Note**

        Only the source code of `raw_function`, the contents of `**kwargs`, and any referenced *global* variables are used to determine whether the image has changed
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
        from .functions import _Function

        info = FunctionInfo(raw_f)

        if shared_volumes or network_file_systems:
            warnings.warn(
                "Mounting NetworkFileSystems or Volumes is usually not advised with `run_function`."
                " If you are trying to download model weights, downloading it to the image itself is recommended and sufficient."
            )

        function = _Function.from_args(
            info,
            stub=None,
            image=self,
            secret=secret,
            secrets=secrets,
            gpu=gpu,
            mounts=mounts,
            network_file_systems=network_file_systems,
            memory=memory,
            timeout=timeout,
            cpu=cpu,
            is_builder_function=True,
        )
        return self.extend(build_function=function, force_build=self.force_build or force_build)

    @typechecked
    def env(self, vars: Dict[str, str]) -> "_Image":
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

    @typechecked
    def workdir(self, path: str) -> "_Image":
        """Sets the working directory for subequent image build steps.

        **Example**

        ```python
        image = (
            modal.Image.debian_slim()
                .run_commands("git clone https://xyz app")
                .workdir("/app")
                .run_commands("yarn install")
        )
        ```
        """
        return self.extend(dockerfile_commands=["FROM base"] + [f"WORKDIR {shlex.quote(path)}"])

    # Live handle methods

    def _is_inside(self) -> bool:
        """Returns whether this container is active or not.

        This is not meant to be called directly: see app.is_inside(image)
        """
        env_image_id = config.get("image_id")
        logger.debug(f"Image._is_inside(): env_image_id={env_image_id} self.object_id={self.object_id}")
        return self.object_id == env_image_id

    @contextlib.contextmanager
    def run_inside(self):
        env_image_id = config.get("image_id")
        try:
            yield
        except Exception as exc:
            if self.object_id is None:
                # Might be initialized later
                self.inside_exceptions.append(exc)
            elif env_image_id == self.object_id:
                # Image is already initialized (we can remove this case later
                # when we don't hydrate objects so early)
                raise


Image = synchronize_api(_Image)
