# Copyright Modal Labs 2022
import contextlib
import os
import re
import shlex
import sys
import typing
import warnings
from dataclasses import dataclass
from inspect import isfunction
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union, get_args

from google.protobuf.message import Message
from grpclib.exceptions import GRPCError, StreamTerminatedError

from modal_proto import api_pb2

from ._resolver import Resolver
from ._serialization import serialize
from ._utils.async_utils import synchronize_api
from ._utils.blob_utils import MAX_OBJECT_SIZE_BYTES
from ._utils.function_utils import FunctionInfo
from ._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors, unary_stream
from .config import config, logger, user_config_path
from .exception import InvalidError, NotFoundError, RemoteError, VersionError, deprecation_error, deprecation_warning
from .gpu import GPU_T, parse_gpu_config
from .mount import _Mount, python_standalone_mount_name
from .network_file_system import _NetworkFileSystem
from .object import _Object
from .secret import _Secret

if typing.TYPE_CHECKING:
    import modal.functions


# This is used for both type checking and runtime validation
ImageBuilderVersion = Literal["2023.12", "2024.04"]

# Note: we also define supported Python versions via logic at the top of the package __init__.py
# so that we fail fast / clearly in unsupported containers. Additionally, we enumerate the supported
# Python versions in mount.py where we specify the "standalone Python versions" we create mounts for.
# Consider consolidating these multiple sources of truth?
SUPPORTED_PYTHON_SERIES: Set[str] = {"3.8", "3.9", "3.10", "3.11", "3.12"}

CONTAINER_REQUIREMENTS_PATH = "/modal_requirements.txt"


def _validate_python_version(version: Optional[str], allow_micro_granularity: bool = True) -> str:
    if version is None:
        # If Python version is unspecified, match the local version, up to the minor component
        version = series_version = "{0}.{1}".format(*sys.version_info)
    elif not isinstance(version, str):
        raise InvalidError(f"Python version must be specified as a string, not {type(version).__name__}")
    elif not re.match(r"^3(?:\.\d{1,2}){1,2}$", version):
        raise InvalidError(f"Invalid Python version: {version!r}")
    else:
        components = version.split(".")
        if len(components) == 3 and not allow_micro_granularity:
            raise InvalidError(
                "Python version must be specified as 'major.minor' for this interface;"
                f" micro-level specification ({version!r}) is not valid."
            )
        series_version = "{0}.{1}".format(*components)

    if series_version not in SUPPORTED_PYTHON_SERIES:
        raise InvalidError(
            f"Unsupported Python version: {version!r}."
            f" Modal supports versions in the following series: {SUPPORTED_PYTHON_SERIES!r}."
        )
    return version


def _dockerhub_python_version(builder_version: ImageBuilderVersion, python_version: Optional[str] = None) -> str:
    python_version = _validate_python_version(python_version)
    components = python_version.split(".")

    # When user specifies a full Python version, use that
    if len(components) > 2:
        return python_version

    # Otherwise, use the same series, but a specific micro version, corresponding to the latest
    # available from https://hub.docker.com/_/python at the time of each image builder release.
    latest_micro_version = {
        "2023.12": {
            "3.12": "1",
            "3.11": "0",
            "3.10": "8",
            "3.9": "15",
            "3.8": "15",
        },
        "2024.04": {
            "3.12": "2",
            "3.11": "8",
            "3.10": "14",
            "3.9": "19",
            "3.8": "19",
        },
    }
    python_series = "{0}.{1}".format(*components)
    micro_version = latest_micro_version[builder_version][python_series]
    python_version = f"{python_series}.{micro_version}"
    return python_version


def _dockerhub_debian_codename(builder_version: ImageBuilderVersion) -> str:
    return {"2023.12": "bullseye", "2024.04": "bookworm"}[builder_version]


def _get_modal_requirements_path(builder_version: ImageBuilderVersion, python_version: Optional[str] = None) -> str:
    # Locate Modal client requirements data
    import modal

    modal_path = Path(modal.__path__[0])

    # When we added Python 3.12 support, we needed to update a few dependencies but did not yet
    # support versioned builds, so we put them in a separate 3.12-specific requirements file.
    # When the python_version is not specified in the Image API, we fall back to the local version.
    # Note that this is buggy if you're using a registry or dockerfile Image that (implicitly) contains 3.12
    # and have a different local version. We can't really fix that; but users can update their image builder.
    # We can get rid of this complexity entirely when we drop support for 2023.12.
    python_version = python_version or sys.version
    suffix = ".312" if builder_version == "2023.12" and python_version.startswith("3.12") else ""

    return str(modal_path / "requirements" / f"{builder_version}{suffix}.txt")


def _get_modal_requirements_command(version: ImageBuilderVersion) -> str:
    command = "pip install"
    if version <= "2023.12":
        args = f"-r {CONTAINER_REQUIREMENTS_PATH}"
    else:
        args = f"--no-cache --no-deps -r {CONTAINER_REQUIREMENTS_PATH}"
    return f"{command} {args}"


def _flatten_str_args(function_name: str, arg_name: str, args: Tuple[Union[str, List[str]], ...]) -> List[str]:
    """Takes a tuple of strings, or string lists, and flattens it.

    Raises an error if any of the elements are not strings or string lists.
    """

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

    args = " ".join(f"{flag} {shlex.quote(value)}" for flag, value in flags if value is not None)
    if pre:
        args += " --pre"

    return args


def _get_image_builder_version(client_version: str) -> ImageBuilderVersion:
    if config_version := config.get("image_builder_version"):
        version = config_version
        if (env_var := "MODAL_IMAGE_BUILDER_VERSION") in os.environ:
            version_source = f" (based on your `{env_var}` environment variable)"
        else:
            version_source = f" (based on your local config file at `{user_config_path}`)"
    else:
        version = client_version
        version_source = ""

    supported_versions: Set[ImageBuilderVersion] = set(get_args(ImageBuilderVersion))
    if version not in supported_versions:
        if config_version is not None:
            update_suggestion = "or remove your local configuration"
        elif version < min(supported_versions):
            update_suggestion = "your image builder version using the Modal dashboard"
        else:
            update_suggestion = "your client library (pip install --upgrade modal)"
        raise VersionError(
            "This version of the modal client supports the following image builder versions:"
            f" {supported_versions!r}."
            f"\n\nYou are using {version!r}{version_source}."
            f" Please update {update_suggestion}."
        )

    return version


class _ImageRegistryConfig:
    """mdmd:hidden"""

    def __init__(
        self,
        # TODO: change to _PUBLIC after worker starts handling it.
        registry_auth_type: int = api_pb2.REGISTRY_AUTH_TYPE_UNSPECIFIED,
        secret: Optional[_Secret] = None,
    ):
        self.registry_auth_type = registry_auth_type
        self.secret = secret

    def get_proto(self) -> api_pb2.ImageRegistryConfig:
        return api_pb2.ImageRegistryConfig(
            registry_auth_type=self.registry_auth_type,
            secret_id=(self.secret.object_id if self.secret else None),
        )


@dataclass
class DockerfileSpec:
    # Ideally we would use field() with default_factory=, but doesn't work with synchronicity type-stub gen
    commands: List[str]
    context_files: Dict[str, str]


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
    def _from_args(
        *,
        base_images: Optional[Dict[str, "_Image"]] = None,
        dockerfile_function: Optional[Callable[[ImageBuilderVersion], DockerfileSpec]] = None,
        secrets: Optional[Sequence[_Secret]] = None,
        gpu_config: Optional[api_pb2.GPUConfig] = None,
        build_function: Optional["modal.functions._Function"] = None,
        build_function_input: Optional[api_pb2.FunctionInput] = None,
        image_registry_config: Optional[_ImageRegistryConfig] = None,
        context_mount: Optional[_Mount] = None,
        force_build: bool = False,
        # For internal use only.
        _namespace: int = api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
    ):
        if base_images is None:
            base_images = {}
        if secrets is None:
            secrets = []
        if gpu_config is None:
            gpu_config = api_pb2.GPUConfig()
        if image_registry_config is None:
            image_registry_config = _ImageRegistryConfig()

        for secret in secrets:
            if not isinstance(secret, _Secret):
                raise InvalidError("All secrets of an image needs to be modal.Secret/AioSecret instances")

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

        async def _load(self: _Image, resolver: Resolver, existing_object_id: Optional[str]):
            builder_version = _get_image_builder_version(resolver.client.image_builder_version)

            if dockerfile_function is None:
                dockerfile = DockerfileSpec(commands=[], context_files={})
            else:
                dockerfile = dockerfile_function(builder_version)

            if not dockerfile.commands and not build_function:
                raise InvalidError(
                    "No commands were provided for the image — have you tried using modal.Image.debian_slim()?"
                )
            if dockerfile.commands and build_function:
                raise InvalidError("Cannot provide both a build function and Dockerfile commands!")

            base_images_pb2s = [
                api_pb2.BaseImage(
                    docker_tag=docker_tag,
                    image_id=image.object_id,
                )
                for docker_tag, image in base_images.items()
            ]

            context_file_pb2s = []
            for filename, path in dockerfile.context_files.items():
                with open(path, "rb") as f:
                    context_file_pb2s.append(api_pb2.ImageContextFile(filename=filename, data=f.read()))

            if build_function:
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
                _build_function = api_pb2.BuildFunction(
                    definition=build_function.get_build_def(),
                    globals=build_function_globals,
                    input=build_function_input,
                )
            else:
                build_function_id = None
                _build_function = None

            image_definition = api_pb2.Image(
                base_images=base_images_pb2s,
                dockerfile_commands=dockerfile.commands,
                context_files=context_file_pb2s,
                secret_ids=[secret.object_id for secret in secrets],
                gpu=bool(gpu_config.type),  # Note: as of 2023-01-27, server still uses this
                context_mount_id=(context_mount.object_id if context_mount else None),
                gpu_config=gpu_config,  # Note: as of 2023-01-27, server ignores this
                image_registry_config=image_registry_config.get_proto(),
                runtime=config.get("function_runtime"),
                runtime_debug=config.get("function_runtime_debug"),
                build_function=_build_function,
            )

            req = api_pb2.ImageGetOrCreateRequest(
                app_id=resolver.app_id,
                image=image_definition,
                existing_image_id=existing_object_id,  # TODO: ignored
                build_function_id=build_function_id,
                force_build=config.get("force_build") or force_build,
                namespace=_namespace,
                builder_version=builder_version,
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

            self._hydrate(image_id, resolver.client, None)

        rep = "Image()"
        obj = _Image._from_loader(_load, rep, deps=_deps)
        obj.force_build = force_build
        return obj

    def extend(self, **kwargs) -> "_Image":
        """Deprecated! This is a low-level method not intended to be part of the public API."""
        deprecation_warning(
            (2024, 3, 7),
            "`Image.extend` is deprecated; please use a higher-level method, such as `Image.dockerfile_commands`.",
        )

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            return DockerfileSpec(
                commands=kwargs.pop("dockerfile_commands", []),
                context_files=kwargs.pop("context_files", {}),
            )

        return _Image._from_args(base_images={"base": self}, dockerfile_function=build_dockerfile, **kwargs)

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

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            commands = ["FROM base", f"COPY . {remote_path}"]  # copy everything from the supplied mount
            return DockerfileSpec(commands=commands, context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            context_mount=mount,
        )

    def copy_local_file(self, local_path: Union[str, Path], remote_path: Union[str, Path] = "./") -> "_Image":
        """Copy a file into the image as a part of building it.

        This works in a similar way to [`COPY`](https://docs.docker.com/engine/reference/builder/#copy) in a `Dockerfile`.
        """
        basename = str(Path(local_path).name)
        mount = _Mount.from_local_file(local_path, remote_path=f"/{basename}")

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            return DockerfileSpec(commands=["FROM base", f"COPY {basename} {remote_path}"], context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            context_mount=mount,
        )

    def copy_local_dir(self, local_path: Union[str, Path], remote_path: Union[str, Path] = ".") -> "_Image":
        """Copy a directory into the image as a part of building the image.

        This works in a similar way to [`COPY`](https://docs.docker.com/engine/reference/builder/#copy) in a `Dockerfile`.
        """
        mount = _Mount.from_local_dir(local_path, remote_path="/")

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            return DockerfileSpec(commands=["FROM base", f"COPY . {remote_path}"], context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            context_mount=mount,
        )

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

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            package_args = " ".join(shlex.quote(pkg) for pkg in sorted(pkgs))
            extra_args = _make_pip_install_args(find_links, index_url, extra_index_url, pre)
            commands = ["FROM base", f"RUN python -m pip install {package_args} {extra_args}"]
            if version > "2023.12":  # Back-compat for legacy trailing space with empty extra_args
                commands = [cmd.strip() for cmd in commands]
            return DockerfileSpec(commands=commands, context_files={})

        gpu_config = parse_gpu_config(gpu)
        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            force_build=self.force_build or force_build,
            gpu_config=gpu_config,
            secrets=secrets,
        )

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

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            commands = ["FROM base"]
            if any(r.startswith("github") for r in repositories):
                commands.append(
                    f"RUN bash -c \"[[ -v GITHUB_TOKEN ]] || (echo 'GITHUB_TOKEN env var not set by provided modal.Secret(s): {secret_names}' && exit 1)\"",
                )
            if any(r.startswith("gitlab") for r in repositories):
                commands.append(
                    f"RUN bash -c \"[[ -v GITLAB_TOKEN ]] || (echo 'GITLAB_TOKEN env var not set by provided modal.Secret(s): {secret_names}' && exit 1)\"",
                )

            extra_args = _make_pip_install_args(find_links, index_url, extra_index_url, pre)
            commands.extend(["RUN apt-get update && apt-get install -y git"])
            commands.extend([f'RUN python3 -m pip install "{url}" {extra_args}' for url in install_urls])
            if version > "2023.12":  # Back-compat for legacy trailing space with empty extra_args
                commands = [cmd.strip() for cmd in commands]
            return DockerfileSpec(commands=commands, context_files={})

        gpu_config = parse_gpu_config(gpu)

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            secrets=secrets,
            gpu_config=gpu_config,
            force_build=self.force_build or force_build,
        )

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

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            requirements_txt_path = os.path.expanduser(requirements_txt)
            context_files = {"/.requirements.txt": requirements_txt_path}

            null_find_links_arg = " " if version == "2023.12" else ""
            find_links_arg = f" -f {find_links}" if find_links else null_find_links_arg
            extra_args = _make_pip_install_args(find_links, index_url, extra_index_url, pre)

            commands = [
                "FROM base",
                "COPY /.requirements.txt /.requirements.txt",
                f"RUN python -m pip install -r /.requirements.txt{find_links_arg} {extra_args}",
            ]
            if version > "2023.12":  # Back-compat for legacy whitespace with empty find_link / extra args
                commands = [cmd.strip() for cmd in commands]
            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            force_build=self.force_build or force_build,
            gpu_config=parse_gpu_config(gpu),
            secrets=secrets,
        )

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
        all of the packages in each listed section are installed as well.
        """

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            # Defer toml import so we don't need it in the container runtime environment
            import toml

            config = toml.load(os.path.expanduser(pyproject_toml))

            dependencies = []
            if "project" not in config or "dependencies" not in config["project"]:
                msg = (
                    "No [project.dependencies] section in pyproject.toml file. "
                    "If your pyproject.toml instead declares [tool.poetry.dependencies], use `Image.poetry_install_from_file()`. "
                    "See https://packaging.python.org/en/latest/guides/writing-pyproject-toml for further file format guidelines."
                )
                raise ValueError(msg)
            else:
                dependencies.extend(config["project"]["dependencies"])
            if optional_dependencies:
                optionals = config["project"]["optional-dependencies"]
                for dep_group_name in optional_dependencies:
                    if dep_group_name in optionals:
                        dependencies.extend(optionals[dep_group_name])

            extra_args = _make_pip_install_args(find_links, index_url, extra_index_url, pre)
            package_args = " ".join(shlex.quote(pkg) for pkg in sorted(dependencies))
            commands = ["FROM base", f"RUN python -m pip install {package_args} {extra_args}"]
            if version > "2023.12":  # Back-compat for legacy trailing space
                commands = [cmd.strip() for cmd in commands]

            return DockerfileSpec(commands=commands, context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            force_build=self.force_build or force_build,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu),
        )

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

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            context_files = {"/.pyproject.toml": os.path.expanduser(poetry_pyproject_toml)}

            commands = ["FROM base", "RUN python -m pip install poetry~=1.7"]

            if old_installer:
                commands += ["RUN poetry config experimental.new-installer false"]

            if not ignore_lockfile:
                nonlocal poetry_lockfile
                if poetry_lockfile is None:
                    p = Path(poetry_pyproject_toml).parent / "poetry.lock"
                    if not p.exists():
                        raise NotFoundError(
                            f"poetry.lock not found at inferred location: {p.absolute()}. If a lockfile is not needed, `ignore_lockfile=True` can be used."
                        )
                    poetry_lockfile = p.as_posix()
                context_files["/.poetry.lock"] = poetry_lockfile
                commands += ["COPY /.poetry.lock /tmp/poetry/poetry.lock"]

            install_cmd = "poetry install --no-root"
            if version == "2023.12":
                # Backwards compatability for previous string, which started with whitespace
                install_cmd = "  " + install_cmd

            if with_:
                install_cmd += f" --with {','.join(with_)}"

            if without:
                install_cmd += f" --without {','.join(without)}"

            if only:
                install_cmd += f" --only {','.join(only)}"
            install_cmd += " --compile"  # no .pyc compilation slows down cold-start.

            commands += [
                "COPY /.pyproject.toml /tmp/poetry/pyproject.toml",
                "RUN cd /tmp/poetry && \\ ",
                "  poetry config virtualenvs.create false && \\ ",
                install_cmd,
            ]
            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            force_build=self.force_build or force_build,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu),
        )

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

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            return DockerfileSpec(commands=["FROM base", *cmds], context_files=context_files)

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu, raise_on_true=False),
            context_mount=context_mount,
            force_build=self.force_build or force_build,
        )

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

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            return DockerfileSpec(commands=["FROM base"] + [f"RUN {cmd}" for cmd in cmds], context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu, raise_on_true=False),
            force_build=self.force_build or force_build,
        )

    @staticmethod
    def conda(python_version: Optional[str] = None, force_build: bool = False) -> "_Image":
        """
        A Conda base image, using miniconda3 and derived from the official Docker Hub image.
        In most cases, using [`Image.micromamba()`](/docs/reference/modal.Image#micromamba) with [`micromamba_install`](/docs/reference/modal.Image#micromamba_install) is recommended over `Image.conda()`, as it leads to significantly faster image build times.
        """

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            nonlocal python_version
            if version == "2023.12" and python_version is None:
                python_version = "3.9"  # Backcompat for old hardcoded default param
            validated_python_version = _validate_python_version(python_version)
            debian_codename = _dockerhub_debian_codename(version)
            requirements_path = _get_modal_requirements_path(version, python_version)
            context_files = {CONTAINER_REQUIREMENTS_PATH: requirements_path}

            # Doesn't use the official continuumio/miniconda3 image as a base. That image has maintenance
            # issues (https://github.com/ContinuumIO/docker-images/issues) and building our own is more flexible.
            conda_install_script = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            commands = [
                f"FROM debian:{debian_codename}",  # the -slim images lack files required by Conda.
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
                f"&& conda install --yes --channel conda-forge python={validated_python_version} \\ ",
                "&& conda update conda \\ ",
                # Remove now unneeded packages and files.
                "&& apt-get --quiet --yes remove curl bzip2 \\ ",
                "&& apt-get --quiet --yes autoremove \\ ",
                "&& apt-get autoclean \\ ",
                "&& rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \\ ",
                "&& conda clean --all --yes",
                # Setup .bashrc for conda.
                "RUN conda init bash --verbose",
                f"COPY {CONTAINER_REQUIREMENTS_PATH} {CONTAINER_REQUIREMENTS_PATH}",
                # .bashrc is explicitly sourced because RUN is a non-login shell and doesn't run bash.
                "RUN . /root/.bashrc && conda activate base \\ ",
                # Ensure that packaging tools are up to date and install client dependenices
                f"&& python -m pip install --upgrade {'pip' if version == '2023.12' else 'pip wheel uv'} \\ ",
                f"&& python -m {_get_modal_requirements_command(version)}",
            ]
            if version > "2023.12":
                commands.append(f"RUN rm {CONTAINER_REQUIREMENTS_PATH}")
            return DockerfileSpec(commands=commands, context_files=context_files)

        base = _Image._from_args(
            dockerfile_function=build_dockerfile,
            force_build=force_build,
            _namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        )

        return base.dockerfile_commands(
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

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            package_args = " ".join(shlex.quote(pkg) for pkg in pkgs)
            channel_args = "".join(f" -c {channel}" for channel in channels)

            commands = [
                "FROM base",
                f"RUN conda install {package_args}{channel_args} --yes \\ ",
                "&& conda clean --yes --index-cache --tarballs --tempfiles --logfiles",
            ]
            return DockerfileSpec(commands=commands, context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            force_build=self.force_build or force_build,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu),
        )

    def conda_update_from_environment(
        self,
        environment_yml: str,
        force_build: bool = False,
        *,
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
    ) -> "_Image":
        """Update a Conda environment using dependencies from a given environment.yml file."""

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            context_files = {"/environment.yml": os.path.expanduser(environment_yml)}

            commands = [
                "FROM base",
                "COPY /environment.yml /environment.yml",
                "RUN conda env update --name base -f /environment.yml \\ ",
                "&& conda clean --yes --index-cache --tarballs --tempfiles --logfiles",
            ]
            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            force_build=self.force_build or force_build,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu),
        )

    @staticmethod
    def micromamba(
        python_version: Optional[str] = None,
        force_build: bool = False,
    ) -> "_Image":
        """
        A Micromamba base image. Micromamba allows for fast building of small Conda-based containers.
        In most cases it will be faster than using [`Image.conda()`](/docs/reference/modal.Image#conda).
        """

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            nonlocal python_version
            if version == "2023.12" and python_version is None:
                python_version = "3.9"  # Backcompat for old hardcoded default param
            validated_python_version = _validate_python_version(python_version)
            micromamba_version = {"2023.12": "1.3.1", "2024.04": "1.5.8"}[version]
            debian_codename = _dockerhub_debian_codename(version)
            tag = f"mambaorg/micromamba:{micromamba_version}-{debian_codename}-slim"
            setup_commands = [
                'SHELL ["/usr/local/bin/_dockerfile_shell.sh"]',
                "ENV MAMBA_DOCKERFILE_ACTIVATE=1",
                f"RUN micromamba install -n base -y python={validated_python_version} pip -c conda-forge",
            ]
            commands = _Image._registry_setup_commands(tag, version, setup_commands)
            context_files = {CONTAINER_REQUIREMENTS_PATH: _get_modal_requirements_path(version, python_version)}
            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            dockerfile_function=build_dockerfile,
            force_build=force_build,
            _namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        )

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

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            package_args = " ".join(shlex.quote(pkg) for pkg in pkgs)
            channel_args = "".join(f" -c {channel}" for channel in channels)

            commands = [
                "FROM base",
                f"RUN micromamba install {package_args}{channel_args} --yes",
            ]
            return DockerfileSpec(commands=commands, context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            force_build=self.force_build or force_build,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu),
        )

    @staticmethod
    def _registry_setup_commands(
        tag: str,
        builder_version: ImageBuilderVersion,
        setup_commands: List[str],
        add_python: Optional[str] = None,
    ) -> List[str]:
        add_python_commands: List[str] = []
        if add_python:
            _validate_python_version(add_python, allow_micro_granularity=False)
            add_python_commands = [
                "COPY /python/. /usr/local",
                "RUN ln -s /usr/local/bin/python3 /usr/local/bin/python",
                "ENV TERMINFO_DIRS=/etc/terminfo:/lib/terminfo:/usr/share/terminfo:/usr/lib/terminfo",
            ]

        modal_requirements_commands = [
            f"COPY {CONTAINER_REQUIREMENTS_PATH} {CONTAINER_REQUIREMENTS_PATH}",
            f"RUN python -m pip install --upgrade {'pip' if builder_version == '2023.12' else 'pip wheel uv'}",
            f"RUN python -m {_get_modal_requirements_command(builder_version)}",
        ]
        if builder_version > "2023.12":
            modal_requirements_commands.append(f"RUN rm {CONTAINER_REQUIREMENTS_PATH}")

        return [
            f"FROM {tag}",
            *add_python_commands,
            *setup_commands,
            *modal_requirements_commands,
        ]

    @staticmethod
    def from_registry(
        tag: str,
        *,
        secret: Optional[_Secret] = None,
        setup_dockerfile_commands: List[str] = [],
        force_build: bool = False,
        add_python: Optional[str] = None,
        **kwargs,
    ) -> "_Image":
        """Build a Modal image from a public or private image registry, such as Docker Hub.

        The image must be built for the `linux/amd64` platform.

        If your image does not come with Python installed, you can use the `add_python` parameter
        to specify a version of Python to add to the image. Supported versions are `3.8`, `3.9`,
        `3.10`, `3.11`, and `3.12`. Otherwise, the image is expected to have Python>3.8 available
        on PATH as `python`, along with `pip`.

        You may also use `setup_dockerfile_commands` to run Dockerfile commands before the
        remaining commands run. This might be useful if you want a custom Python installation or to
        set a `SHELL`. Prefer `run_commands()` when possible though.

        To authenticate against a private registry with static credentials, you must set the `secret` parameter to
        a `modal.Secret` containing a username (`REGISTRY_USERNAME`) and an access token or password (`REGISTRY_PASSWORD`).

        To authenticate against private registries with credentials from a cloud provider, use `Image.from_gcp_artifact_registry()`
        or `Image.from_aws_ecr()`.

        **Examples**

        ```python
        modal.Image.from_registry("python:3.11-slim-bookworm")
        modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
        modal.Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
        ```
        """
        context_mount = None
        if add_python:
            context_mount = _Mount.from_name(
                python_standalone_mount_name(add_python),
                namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
            )

        if "image_registry_config" not in kwargs and secret is not None:
            kwargs["image_registry_config"] = _ImageRegistryConfig(api_pb2.REGISTRY_AUTH_TYPE_STATIC_CREDS, secret)

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            commands = _Image._registry_setup_commands(tag, version, setup_dockerfile_commands, add_python)
            context_files = {CONTAINER_REQUIREMENTS_PATH: _get_modal_requirements_path(version, add_python)}
            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            dockerfile_function=build_dockerfile,
            context_mount=context_mount,
            force_build=force_build,
            **kwargs,
        )

    @staticmethod
    def from_gcp_artifact_registry(
        tag: str,
        secret: Optional[_Secret] = None,
        *,
        setup_dockerfile_commands: List[str] = [],
        force_build: bool = False,
        add_python: Optional[str] = None,
        **kwargs,
    ) -> "_Image":
        """Build a Modal image from a private image in Google Cloud Platform (GCP) Artifact Registry.

        You will need to pass a `modal.Secret` containing [your GCP service account key data](https://cloud.google.com/iam/docs/keys-create-delete#creating)
        as `SERVICE_ACCOUNT_JSON`. This can be done from the [Secrets](/secrets) page. Your service account should be granted a specific
        role depending on the GCP registry used:

        - For Artifact Registry images (`pkg.dev` domains) use the ["Artifact Registry Reader"](https://cloud.google.com/artifact-registry/docs/access-control#roles) role
        - For Container Registry images (`gcr.io` domains) use the ["Storage Object Viewer"](https://cloud.google.com/artifact-registry/docs/transition/setup-gcr-repo#permissions) role

        **Note:** This method does not use `GOOGLE_APPLICATION_CREDENTIALS` as that variable accepts a path to a JSON file, not the actual JSON string.

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
        if "secrets" in kwargs:
            raise TypeError("Passing a list of 'secrets' is not supported; use the singular 'secret' argument.")
        image_registry_config = _ImageRegistryConfig(api_pb2.REGISTRY_AUTH_TYPE_GCP, secret)
        return _Image.from_registry(
            tag,
            setup_dockerfile_commands=setup_dockerfile_commands,
            force_build=force_build,
            add_python=add_python,
            image_registry_config=image_registry_config,
            **kwargs,
        )

    @staticmethod
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
        if "secrets" in kwargs:
            raise TypeError("Passing a list of 'secrets' is not supported; use the singular 'secret' argument.")
        image_registry_config = _ImageRegistryConfig(api_pb2.REGISTRY_AUTH_TYPE_AWS, secret)
        return _Image.from_registry(
            tag,
            setup_dockerfile_commands=setup_dockerfile_commands,
            force_build=force_build,
            add_python=add_python,
            image_registry_config=image_registry_config,
            **kwargs,
        )

    @staticmethod
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
        `3.10`, `3.11`, and `3.12`.

        **Example**

        ```python
        image = modal.Image.from_dockerfile("./Dockerfile", add_python="3.12")
        ```
        """

        # --- Build the base dockerfile

        def build_dockerfile_base(version: ImageBuilderVersion) -> DockerfileSpec:
            with open(os.path.expanduser(path)) as f:
                commands = f.read().split("\n")
            return DockerfileSpec(commands=commands, context_files={})

        gpu_config = parse_gpu_config(gpu)
        base_image = _Image._from_args(
            dockerfile_function=build_dockerfile_base,
            context_mount=context_mount,
            gpu_config=gpu_config,
            secrets=secrets,
        )

        # --- Now add in the modal dependencies, and, optionally a Python distribution
        # This happening in two steps is probably a vestigial consequence of previous limitations,
        # but it will be difficult to merge them without forcing rebuilds of images.

        if add_python:
            context_mount = _Mount.from_name(
                python_standalone_mount_name(add_python),
                namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
            )
        else:
            context_mount = None

        def build_dockerfile_python(version: ImageBuilderVersion) -> DockerfileSpec:
            commands = _Image._registry_setup_commands("base", version, [], add_python)
            requirements_path = _get_modal_requirements_path(version, add_python)
            context_files = {CONTAINER_REQUIREMENTS_PATH: requirements_path}
            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            base_images={"base": base_image},
            dockerfile_function=build_dockerfile_python,
            context_mount=context_mount,
            force_build=force_build,
        )

    @staticmethod
    def debian_slim(python_version: Optional[str] = None, force_build: bool = False) -> "_Image":
        """Default image, based on the official `python` Docker images."""

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            requirements_path = _get_modal_requirements_path(version, python_version)
            context_files = {CONTAINER_REQUIREMENTS_PATH: requirements_path}
            full_python_version = _dockerhub_python_version(version, python_version)
            debian_codename = _dockerhub_debian_codename(version)

            commands = [
                f"FROM python:{full_python_version}-slim-{debian_codename}",
                f"COPY {CONTAINER_REQUIREMENTS_PATH} {CONTAINER_REQUIREMENTS_PATH}",
                "RUN apt-get update",
                "RUN apt-get install -y gcc gfortran build-essential",
                f"RUN pip install --upgrade {'pip' if version == '2023.12' else 'pip wheel uv'}",
                f"RUN {_get_modal_requirements_command(version)}",
                # Set debian front-end to non-interactive to avoid users getting stuck with input prompts.
                "RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections",
            ]
            if version > "2023.12":
                commands.append(f"RUN rm {CONTAINER_REQUIREMENTS_PATH}")
            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            dockerfile_function=build_dockerfile,
            force_build=force_build,
            _namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        )

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

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            commands = [
                "FROM base",
                "RUN apt-get update",
                f"RUN apt-get install -y {package_args}",
            ]
            return DockerfileSpec(commands=commands, context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            force_build=self.force_build or force_build,
            gpu_config=parse_gpu_config(gpu),
            secrets=secrets,
        )

    def run_function(
        self,
        raw_f: Callable,
        secrets: Sequence[_Secret] = (),  # Optional Modal Secret objects with environment variables for the container
        gpu: GPU_T = None,  # GPU specification as string ("any", "T4", "A10G", ...) or object (`modal.GPU.A100()`, ...)
        mounts: Sequence[_Mount] = (),
        shared_volumes: Dict[Union[str, PurePosixPath], _NetworkFileSystem] = {},
        network_file_systems: Dict[Union[str, PurePosixPath], _NetworkFileSystem] = {},
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        timeout: Optional[int] = 86400,  # Maximum execution time of the function in seconds.
        force_build: bool = False,
        secret: Optional[_Secret] = None,  # Deprecated: use `secrets`.
        args: Sequence[Any] = (),  # Positional arguments to the function.
        kwargs: Dict[str, Any] = {},  # Keyword arguments to the function.
    ) -> "_Image":
        """Run user-defined function `raw_f` as an image build step. The function runs just like an ordinary Modal
        function, and any kwargs accepted by `@app.function` (such as `Mount`s, `NetworkFileSystem`s, and resource requests) can
        be supplied to it. After it finishes execution, a snapshot of the resulting container file system is saved as an image.

        **Note**

        Only the source code of `raw_f`, the contents of `**kwargs`, and any referenced *global* variables are used to determine whether the image has changed
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
            app=None,
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
        if len(args) + len(kwargs) > 0:
            args_serialized = serialize((args, kwargs))
            if len(args_serialized) > MAX_OBJECT_SIZE_BYTES:
                raise InvalidError(
                    f"Arguments to `run_function` are too large ({len(args_serialized)} bytes). "
                    f"Maximum size is {MAX_OBJECT_SIZE_BYTES} bytes."
                )
            build_function_input = api_pb2.FunctionInput(args=args_serialized, data_format=api_pb2.DATA_FORMAT_PICKLE)
        else:
            build_function_input = None
        return _Image._from_args(
            base_images={"base": self},
            build_function=function,
            build_function_input=build_function_input,
            force_build=self.force_build or force_build,
        )

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

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            commands = ["FROM base"] + [f"ENV {key}={shlex.quote(val)}" for (key, val) in vars.items()]
            return DockerfileSpec(commands=commands, context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )

    def workdir(self, path: str) -> "_Image":
        """Set the working directory for subsequent image build steps and function execution.

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

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            commands = ["FROM base", f"WORKDIR {shlex.quote(path)}"]
            return DockerfileSpec(commands=commands, context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )

    # Live handle methods

    @contextlib.contextmanager
    def imports(self):
        """
        Used to import packages in global scope that are only available when running remotely.
        By using this context manager you can avoid an `ImportError` due to not having certain
        packages installed locally.

        **Usage:**

        ```python notest
        with image.imports():
            import torch
        ```
        """
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
            if not isinstance(exc, ImportError):
                warnings.warn(f"Warning: caught a non-ImportError exception in an `imports()` block: {repr(exc)}")

    def run_inside(self):
        """`Image.run_inside` is deprecated - use `Image.imports` instead.

        **Usage:**

        ```python notest
        with image.imports():
            import torch
        ```
        """
        deprecation_error((2023, 12, 15), Image.run_inside.__doc__)


Image = synchronize_api(_Image)
