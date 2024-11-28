# Copyright Modal Labs 2022
import contextlib
import json
import os
import re
import shlex
import sys
import typing
import warnings
from dataclasses import dataclass
from inspect import isfunction
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
    get_args,
)

from google.protobuf.message import Message
from grpclib.exceptions import GRPCError, StreamTerminatedError

from modal_proto import api_pb2

from ._resolver import Resolver
from ._serialization import serialize
from ._utils.async_utils import synchronize_api
from ._utils.blob_utils import MAX_OBJECT_SIZE_BYTES
from ._utils.function_utils import FunctionInfo
from ._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES, retry_transient_errors
from .client import _Client
from .cloud_bucket_mount import _CloudBucketMount
from .config import config, logger, user_config_path
from .environments import _get_environment_cached
from .exception import InvalidError, NotFoundError, RemoteError, VersionError, deprecation_error, deprecation_warning
from .gpu import GPU_T, parse_gpu_config
from .mount import _Mount, python_standalone_mount_name
from .network_file_system import _NetworkFileSystem
from .object import _Object, live_method_gen
from .output import _get_output_manager
from .scheduler_placement import SchedulerPlacement
from .secret import _Secret
from .volume import _Volume

if typing.TYPE_CHECKING:
    import modal.functions

# This is used for both type checking and runtime validation
ImageBuilderVersion = Literal["2023.12", "2024.04", "2024.10"]

# Note: we also define supported Python versions via logic at the top of the package __init__.py
# so that we fail fast / clearly in unsupported containers. Additionally, we enumerate the supported
# Python versions in mount.py where we specify the "standalone Python versions" we create mounts for.
# Consider consolidating these multiple sources of truth?
SUPPORTED_PYTHON_SERIES: Dict[ImageBuilderVersion, List[str]] = {
    "2024.10": ["3.9", "3.10", "3.11", "3.12", "3.13"],
    "2024.04": ["3.9", "3.10", "3.11", "3.12"],
    "2023.12": ["3.9", "3.10", "3.11", "3.12"],
}

LOCAL_REQUIREMENTS_DIR = Path(__file__).parent / "requirements"
CONTAINER_REQUIREMENTS_PATH = "/modal_requirements.txt"


def _validate_python_version(
    python_version: Optional[str], builder_version: ImageBuilderVersion, allow_micro_granularity: bool = True
) -> str:
    if python_version is None:
        # If Python version is unspecified, match the local version, up to the minor component
        python_version = series_version = "{0}.{1}".format(*sys.version_info)
    elif not isinstance(python_version, str):
        raise InvalidError(f"Python version must be specified as a string, not {type(python_version).__name__}")
    elif not re.match(r"^3(?:\.\d{1,2}){1,2}(rc\d*)?$", python_version):
        raise InvalidError(f"Invalid Python version: {python_version!r}")
    else:
        components = python_version.split(".")
        if len(components) == 3 and not allow_micro_granularity:
            raise InvalidError(
                "Python version must be specified as 'major.minor' for this interface;"
                f" micro-level specification ({python_version!r}) is not valid."
            )
        series_version = "{0}.{1}".format(*components)

    supported_series = SUPPORTED_PYTHON_SERIES[builder_version]
    if series_version not in supported_series:
        raise InvalidError(
            f"Unsupported Python version: {python_version!r}."
            f" When using the {builder_version!r} Image builder, Modal supports the following series:"
            f" {supported_series!r}."
        )
    return python_version


def _dockerhub_python_version(builder_version: ImageBuilderVersion, python_version: Optional[str] = None) -> str:
    python_version = _validate_python_version(python_version, builder_version)
    version_components = python_version.split(".")

    # When user specifies a full Python version, use that
    if len(version_components) > 2:
        return python_version

    # Otherwise, use the same series, but a specific micro version, corresponding to the latest
    # available from https://hub.docker.com/_/python at the time of each image builder release.
    # This allows us to publish one pre-built debian-slim image per Python series.
    python_versions = _base_image_config("python", builder_version)
    series_to_micro_version = dict(tuple(v.rsplit(".", 1)) for v in python_versions)
    python_series_requested = "{0}.{1}".format(*version_components)
    micro_version = series_to_micro_version[python_series_requested]
    return f"{python_series_requested}.{micro_version}"


def _base_image_config(group: str, builder_version: ImageBuilderVersion) -> Any:
    with open(LOCAL_REQUIREMENTS_DIR / "base-images.json", "r") as f:
        data = json.load(f)
    return data[group][builder_version]


def _get_modal_requirements_path(builder_version: ImageBuilderVersion, python_version: Optional[str] = None) -> str:
    # When we added Python 3.12 support, we needed to update a few dependencies but did not yet
    # support versioned builds, so we put them in a separate 3.12-specific requirements file.
    # When the python_version is not specified in the Image API, we fall back to the local version.
    # Note that this is buggy if you're using a registry or dockerfile Image that (implicitly) contains 3.12
    # and have a different local version. We can't really fix that; but users can update their image builder.
    # We can get rid of this complexity entirely when we drop support for 2023.12.
    python_version = python_version or sys.version
    suffix = ".312" if builder_version == "2023.12" and python_version.startswith("3.12") else ""

    return str(LOCAL_REQUIREMENTS_DIR / f"{builder_version}{suffix}.txt")


def _get_modal_requirements_command(version: ImageBuilderVersion) -> str:
    if version == "2023.12":
        prefix = "pip install"
    elif version == "2024.04":
        prefix = "pip install --no-cache --no-deps"
    else:  # Currently, 2024.10+
        prefix = "uv pip install --system --compile-bytecode --no-cache --no-deps"

    return f"{prefix} -r {CONTAINER_REQUIREMENTS_PATH}"


def _flatten_str_args(function_name: str, arg_name: str, args: Sequence[Union[str, List[str]]]) -> List[str]:
    """Takes a sequence of strings, or string lists, and flattens it.

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


def _validate_packages(packages: List[str]) -> bool:
    """Validates that a list of packages does not contain any command-line options."""
    return not any(pkg.startswith("-") for pkg in packages)


def _warn_invalid_packages(old_command: str) -> None:
    deprecation_warning(
        (2024, 7, 3),
        "Passing flags to `pip` via the `packages` argument of `pip_install` is deprecated."
        " Please pass flags via the `extra_options` argument instead."
        "\nNote that this will cause a rebuild of this image layer."
        " To avoid rebuilding, you can pass the following to `run_commands` instead:"
        f'\n`image.run_commands("{old_command}")`',
        show_source=False,
    )


def _make_pip_install_args(
    find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
    index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
    extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
    pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
    extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation --no-clean"
) -> str:
    flags = [
        ("--find-links", find_links),  # TODO(erikbern): allow multiple?
        ("--index-url", index_url),
        ("--extra-index-url", extra_index_url),  # TODO(erikbern): allow multiple?
    ]

    args = " ".join(f"{flag} {shlex.quote(value)}" for flag, value in flags if value is not None)
    if pre:
        args += " --pre"  # TODO: remove extra whitespace in future image builder version

    if extra_options:
        if args:
            args += " "
        args += f"{extra_options}"

    return args


def _get_image_builder_version(server_version: ImageBuilderVersion) -> ImageBuilderVersion:
    if local_config_version := config.get("image_builder_version"):
        version = local_config_version
        if (env_var := "MODAL_IMAGE_BUILDER_VERSION") in os.environ:
            version_source = f" (based on your `{env_var}` environment variable)"
        else:
            version_source = f" (based on your local config file at `{user_config_path}`)"
    else:
        version_source = ""
        version = server_version

    supported_versions: Set[ImageBuilderVersion] = set(get_args(ImageBuilderVersion))
    if version not in supported_versions:
        if local_config_version is not None:
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
        registry_auth_type: "api_pb2.RegistryAuthType.ValueType" = api_pb2.REGISTRY_AUTH_TYPE_UNSPECIFIED,
        secret: Optional[_Secret] = None,
    ):
        self.registry_auth_type = registry_auth_type
        self.secret = secret

    def get_proto(self) -> api_pb2.ImageRegistryConfig:
        return api_pb2.ImageRegistryConfig(
            registry_auth_type=self.registry_auth_type,
            secret_id=(self.secret.object_id if self.secret else ""),
        )


@dataclass
class DockerfileSpec:
    # Ideally we would use field() with default_factory=, but doesn't work with synchronicity type-stub gen
    commands: List[str]
    context_files: Dict[str, str]


async def _image_await_build_result(image_id: str, client: _Client) -> api_pb2.ImageJoinStreamingResponse:
    last_entry_id: str = ""
    result_response: Optional[api_pb2.ImageJoinStreamingResponse] = None

    async def join():
        nonlocal last_entry_id, result_response

        request = api_pb2.ImageJoinStreamingRequest(image_id=image_id, timeout=55, last_entry_id=last_entry_id)
        async for response in client.stub.ImageJoinStreaming.unary_stream(request):
            if response.entry_id:
                last_entry_id = response.entry_id
            if response.result.status:
                result_response = response
                # can't return yet, since there may still be logs streaming back in subsequent responses
            for task_log in response.task_logs:
                if task_log.task_progress.pos or task_log.task_progress.len:
                    assert task_log.task_progress.progress_type == api_pb2.IMAGE_SNAPSHOT_UPLOAD
                    if output_mgr := _get_output_manager():
                        output_mgr.update_snapshot_progress(image_id, task_log.task_progress)
                elif task_log.data:
                    if output_mgr := _get_output_manager():
                        await output_mgr.put_log_content(task_log)
        if output_mgr := _get_output_manager():
            output_mgr.flush_lines()

    # Handle up to n exceptions while fetching logs
    retry_count = 0
    while result_response is None:
        try:
            await join()
        except (StreamTerminatedError, GRPCError) as exc:
            if isinstance(exc, GRPCError) and exc.status not in RETRYABLE_GRPC_STATUS_CODES:
                raise exc
            retry_count += 1
            if retry_count >= 3:
                raise exc
    return result_response


class _Image(_Object, type_prefix="im"):
    """Base class for container images to run functions in.

    Do not construct this class directly; instead use one of its static factory methods,
    such as `modal.Image.debian_slim`, `modal.Image.from_registry`, or `modal.Image.micromamba`.
    """

    force_build: bool
    inside_exceptions: List[Exception]
    _serve_mounts: typing.FrozenSet[_Mount]  # used for mounts watching in `modal serve`
    _deferred_mounts: Sequence[
        _Mount
    ]  # added as mounts on any container referencing the Image, see `def _mount_layers`
    _metadata: Optional[api_pb2.ImageMetadata] = None  # set on hydration, private for now

    def _initialize_from_empty(self):
        self.inside_exceptions = []
        self._serve_mounts = frozenset()
        self._deferred_mounts = ()
        self.force_build = False

    def _initialize_from_other(self, other: "_Image"):
        # used by .clone()
        self.inside_exceptions = other.inside_exceptions
        self.force_build = other.force_build
        self._serve_mounts = other._serve_mounts
        self._deferred_mounts = other._deferred_mounts

    def _hydrate_metadata(self, metadata: Optional[Message]):
        env_image_id = config.get("image_id")  # set as an env var in containers
        if env_image_id == self.object_id:
            for exc in self.inside_exceptions:
                # This raises exceptions from `with image.imports()` blocks
                # if the hydrated image is the one used by the container
                raise exc

        if metadata:
            assert isinstance(metadata, api_pb2.ImageMetadata)
            self._metadata = metadata

    def _add_mount_layer_or_copy(self, mount: _Mount, copy: bool = False):
        if copy:
            return self.copy_mount(mount, remote_path="/")

        base_image = self

        async def _load(self2: "_Image", resolver: Resolver, existing_object_id: Optional[str]):
            self2._hydrate_from_other(base_image)  # same image id as base image as long as it's lazy
            self2._deferred_mounts = tuple(base_image._deferred_mounts) + (mount,)
            self2._serve_mounts = base_image._serve_mounts | ({mount} if mount.is_local() else set())

        return _Image._from_loader(_load, "Image(local files)", deps=lambda: [base_image, mount])

    @property
    def _mount_layers(self) -> typing.Sequence[_Mount]:
        """Non-evaluated mount layers on the image

        When the image is used by a Modal container, these mounts need to be attached as well to
        represent the full image content, as they haven't yet been represented as a layer in the
        image.

        When the image is used as a base image for a new layer (that is not itself a mount layer)
        these mounts need to first be inserted as a copy operation (.copy_mount) into the image.
        """
        return self._deferred_mounts

    def _assert_no_mount_layers(self):
        if self._mount_layers:
            raise InvalidError(
                "An image tried to run a build step after using `image.add_local_*` to include local files.\n"
                "\n"
                "Run `image.add_local_*` commands last in your image build to avoid rebuilding images with every local "
                "file change. Modal will then add these files to containers on startup instead, saving build time.\n"
                "If you need to run other build steps after adding local files, set `copy=True` to copy the files "
                "directly into the image, at the expense of some added build time.\n"
                "\n"
                "Example:\n"
                "\n"
                "my_image = (\n"
                "    Image.debian_slim()\n"
                '    .add_local_file("data.json", copy=True)\n'
                '    .run_commands("python -m mypak")  # this now works!\n'
                ")\n"
            )

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
        _namespace: "api_pb2.DeploymentNamespace.ValueType" = api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        _do_assert_no_mount_layers: bool = True,
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

        def _deps() -> Sequence[_Object]:
            deps = tuple(base_images.values()) + tuple(secrets)
            if build_function:
                deps += (build_function,)
            if context_mount:
                deps += (context_mount,)
            if image_registry_config and image_registry_config.secret:
                deps += (image_registry_config.secret,)
            return deps

        async def _load(self: _Image, resolver: Resolver, existing_object_id: Optional[str]):
            if _do_assert_no_mount_layers:
                for image in base_images.values():
                    # base images can't have
                    image._assert_no_mount_layers()

            assert resolver.app_id  # type narrowing
            environment = await _get_environment_cached(resolver.environment_name or "", resolver.client)
            # A bit hacky,but assume that the environment provides a valid builder version
            image_builder_version = cast(ImageBuilderVersion, environment._settings.image_builder_version)
            builder_version = _get_image_builder_version(image_builder_version)

            if dockerfile_function is None:
                dockerfile = DockerfileSpec(commands=[], context_files={})
            else:
                dockerfile = dockerfile_function(builder_version)

            if not dockerfile.commands and not build_function:
                raise InvalidError(
                    "No commands were provided for the image — have you tried using modal.Image.debian_slim()?"
                )
            if dockerfile.commands and build_function:
                raise InvalidError(
                    "Cannot provide both build function and Dockerfile commands in the same image layer!"
                )

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
                attrs = build_function._get_info().get_cls_var_attrs()
                globals = {**globals, **attrs}
                filtered_globals = {}
                for k, v in globals.items():
                    if isfunction(v):
                        continue
                    try:
                        serialize(v)
                    except Exception:
                        # Skip unserializable values for now.
                        logger.warning(
                            f"Skipping unserializable global variable {k} for "
                            f"{build_function._get_info().function_name}. "
                            "Changes to this variable won't invalidate the image."
                        )
                        continue
                    filtered_globals[k] = v

                # Cloudpickle function serialization produces unstable values.
                # TODO: better way to filter out types that don't have a stable hash?
                build_function_globals = serialize(filtered_globals) if filtered_globals else b""
                _build_function = api_pb2.BuildFunction(
                    definition=build_function.get_build_def(),
                    globals=build_function_globals,
                    input=build_function_input,
                )
            else:
                build_function_id = ""
                _build_function = None

            image_definition = api_pb2.Image(
                base_images=base_images_pb2s,
                dockerfile_commands=dockerfile.commands,
                context_files=context_file_pb2s,
                secret_ids=[secret.object_id for secret in secrets],
                gpu=bool(gpu_config.type),  # Note: as of 2023-01-27, server still uses this
                context_mount_id=(context_mount.object_id if context_mount else ""),
                gpu_config=gpu_config,  # Note: as of 2023-01-27, server ignores this
                image_registry_config=image_registry_config.get_proto(),
                runtime=config.get("function_runtime"),
                runtime_debug=config.get("function_runtime_debug"),
                build_function=_build_function,
            )

            req = api_pb2.ImageGetOrCreateRequest(
                app_id=resolver.app_id,
                image=image_definition,
                existing_image_id=existing_object_id or "",  # TODO: ignored
                build_function_id=build_function_id,
                force_build=config.get("force_build") or force_build,
                namespace=_namespace,
                builder_version=builder_version,
                # Failsafe mechanism to prevent inadvertant updates to the global images.
                # Only admins can publish to the global namespace, but they have to additionally request it.
                allow_global_deployment=os.environ.get("MODAL_IMAGE_ALLOW_GLOBAL_DEPLOYMENT", "0") == "1",
            )
            resp = await retry_transient_errors(resolver.client.stub.ImageGetOrCreate, req)
            image_id = resp.image_id
            result: api_pb2.GenericResult
            metadata: Optional[api_pb2.ImageMetadata] = None

            if resp.result.status:
                # image already built
                result = resp.result
                if resp.HasField("metadata"):
                    metadata = resp.metadata
            else:
                # not built or in the process of building - wait for build
                logger.debug("Waiting for image %s" % image_id)
                resp = await _image_await_build_result(image_id, resolver.client)
                result = resp.result
                if resp.HasField("metadata"):
                    metadata = resp.metadata

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

            self._hydrate(image_id, resolver.client, metadata)
            local_mounts = set()
            for base in base_images.values():
                local_mounts |= base._serve_mounts
            if context_mount and context_mount.is_local():
                local_mounts.add(context_mount)
            self._serve_mounts = frozenset(local_mounts)

        rep = f"Image({dockerfile_function})"
        obj = _Image._from_loader(_load, rep, deps=_deps)
        obj.force_build = force_build
        return obj

    def extend(self, **kwargs) -> "_Image":
        """mdmd:hidden"""
        deprecation_error(
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

    def add_local_file(self, local_path: Union[str, Path], remote_path: str, *, copy: bool = False) -> "_Image":
        """Adds a local file to the image at `remote_path` within the container

        By default (`copy=False`), the files are added to containers on startup and are not built into the actual Image,
        which speeds up deployment.

        Set `copy=True` to copy the files into an Image layer at build time instead, similar to how
        [`COPY`](https://docs.docker.com/engine/reference/builder/#copy) works in a `Dockerfile`.

        copy=True can slow down iteration since it requires a rebuild of the Image and any subsequent
        build steps whenever the included files change, but it is required if you want to run additional
        build steps after this one.
        """
        if not PurePosixPath(remote_path).is_absolute():
            # TODO(elias): implement relative to absolute resolution using image workdir metadata
            #  + make default remote_path="./"
            #  This requires deferring the Mount creation until after "self" (the base image) has been resolved
            #  so we know the workdir of the operation.
            raise InvalidError("image.add_local_file() currently only supports absolute remote_path values")

        if remote_path.endswith("/"):
            remote_path = remote_path + Path(local_path).name

        mount = _Mount.from_local_file(local_path, remote_path)
        return self._add_mount_layer_or_copy(mount, copy=copy)

    def add_local_dir(self, local_path: Union[str, Path], remote_path: str, *, copy: bool = False) -> "_Image":
        """Adds a local directory's content to the image at `remote_path` within the container

        By default (`copy=False`), the files are added to containers on startup and are not built into the actual Image,
        which speeds up deployment.

        Set `copy=True` to copy the files into an Image layer at build time instead, similar to how
        [`COPY`](https://docs.docker.com/engine/reference/builder/#copy) works in a `Dockerfile`.

        copy=True can slow down iteration since it requires a rebuild of the Image and any subsequent
        build steps whenever the included files change, but it is required if you want to run additional
        build steps after this one.
        """
        if not PurePosixPath(remote_path).is_absolute():
            # TODO(elias): implement relative to absolute resolution using image workdir metadata
            #  + make default remote_path="./"
            raise InvalidError("image.add_local_dir() currently only supports absolute remote_path values")
        mount = _Mount.from_local_dir(local_path, remote_path=remote_path)
        return self._add_mount_layer_or_copy(mount, copy=copy)

    def copy_local_file(self, local_path: Union[str, Path], remote_path: Union[str, Path] = "./") -> "_Image":
        """Copy a file into the image as a part of building it.

        This works in a similar way to [`COPY`](https://docs.docker.com/engine/reference/builder/#copy)
        works in a `Dockerfile`.
        """
        # TODO(elias): add pending deprecation with suggestion to use add_* instead
        basename = str(Path(local_path).name)
        mount = _Mount.from_local_file(local_path, remote_path=f"/{basename}")

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            return DockerfileSpec(commands=["FROM base", f"COPY {basename} {remote_path}"], context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            context_mount=mount,
        )

    def _add_local_python_packages(self, *packages: str, copy: bool = False) -> "_Image":
        """Adds Python package files to containers

        Adds all files from the specified Python packages to containers running the Image.

        Packages are added to the `/root` directory of containers, which is on the `PYTHONPATH`
        of any executed Modal Functions.

        By default (`copy=False`), the files are added to containers on startup and are not built into the actual Image,
        which speeds up deployment.

        Set `copy=True` to copy the files into an Image layer at build time instead. This can slow down iteration since
        it requires a rebuild of the Image and any subsequent build steps whenever the included files change, but it is
        required if you want to run additional build steps after this one.

        **Note:** This excludes all dot-prefixed subdirectories or files and all `.pyc`/`__pycache__` files.
        To add full directories with finer control, use `.add_local_dir()` instead.
        """
        mount = _Mount.from_local_python_packages(*packages)
        return self._add_mount_layer_or_copy(mount, copy=copy)

    def copy_local_dir(self, local_path: Union[str, Path], remote_path: Union[str, Path] = ".") -> "_Image":
        """Copy a directory into the image as a part of building the image.

        This works in a similar way to [`COPY`](https://docs.docker.com/engine/reference/builder/#copy)
        works in a `Dockerfile`.
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
        extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation --no-clean"
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
    ) -> "_Image":
        """Install a list of Python packages using pip.

        **Examples**

        Simple installation:
        ```python
        image = modal.Image.debian_slim().pip_install("click", "httpx~=0.23.3")
        ```

        More complex installation:
        ```python
        image = (
            modal.Image.from_registry(
                "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
            )
            .pip_install(
                "ninja",
                "packaging",
                "wheel",
                "transformers==4.40.2",
            )
            .pip_install(
                "flash-attn==2.5.8", extra_options="--no-build-isolation"
            )
        )
        ```
        """
        pkgs = _flatten_str_args("pip_install", "packages", packages)
        if not pkgs:
            return self

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            package_args = " ".join(shlex.quote(pkg) for pkg in sorted(pkgs))
            extra_args = _make_pip_install_args(find_links, index_url, extra_index_url, pre, extra_options)
            commands = ["FROM base", f"RUN python -m pip install {package_args} {extra_args}"]
            if not _validate_packages(pkgs):
                _warn_invalid_packages(commands[-1].split("RUN ")[-1])
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
        extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation --no-clean"
        gpu: GPU_T = None,
        secrets: Sequence[_Secret] = [],
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
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
                "No secrets provided to function. "
                "Installing private packages requires tokens to be passed via modal.Secret objects."
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
                    'RUN bash -c "[[ -v GITHUB_TOKEN ]] || '
                    f"(echo 'GITHUB_TOKEN env var not set by provided modal.Secret(s): {secret_names}' && exit 1)\"",
                )
            if any(r.startswith("gitlab") for r in repositories):
                commands.append(
                    'RUN bash -c "[[ -v GITLAB_TOKEN ]] || '
                    f"(echo 'GITLAB_TOKEN env var not set by provided modal.Secret(s): {secret_names}' && exit 1)\"",
                )

            extra_args = _make_pip_install_args(find_links, index_url, extra_index_url, pre, extra_options)
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
        extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation --no-clean"
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
    ) -> "_Image":
        """Install a list of Python packages from a local `requirements.txt` file."""

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            requirements_txt_path = os.path.expanduser(requirements_txt)
            context_files = {"/.requirements.txt": requirements_txt_path}

            null_find_links_arg = " " if version == "2023.12" else ""
            find_links_arg = f" -f {find_links}" if find_links else null_find_links_arg
            extra_args = _make_pip_install_args(find_links, index_url, extra_index_url, pre, extra_options)

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
        extra_options: str = "",  # Additional options to pass to pip install, e.g. "--no-build-isolation --no-clean"
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
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
                    "If your pyproject.toml instead declares [tool.poetry.dependencies], "
                    "use `Image.poetry_install_from_file()`. "
                    "See https://packaging.python.org/en/latest/guides/writing-pyproject-toml "
                    "for further file format guidelines."
                )
                raise ValueError(msg)
            else:
                dependencies.extend(config["project"]["dependencies"])
            if optional_dependencies:
                optionals = config["project"]["optional-dependencies"]
                for dep_group_name in optional_dependencies:
                    if dep_group_name in optionals:
                        dependencies.extend(optionals[dep_group_name])

            extra_args = _make_pip_install_args(find_links, index_url, extra_index_url, pre, extra_options)
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
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
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
                            f"poetry.lock not found at inferred location: {p.absolute()}. "
                            "If a lockfile is not needed, `ignore_lockfile=True` can be used."
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
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
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
            gpu_config=parse_gpu_config(gpu),
            context_mount=context_mount,
            force_build=self.force_build or force_build,
        )

    def entrypoint(
        self,
        entrypoint_commands: List[str],
    ) -> "_Image":
        """Set the entrypoint for the image."""
        args_str = _flatten_str_args("entrypoint", "entrypoint_files", entrypoint_commands)
        args_str = '"' + '", "'.join(args_str) + '"' if args_str else ""
        dockerfile_cmd = f"ENTRYPOINT [{args_str}]"

        return self.dockerfile_commands(dockerfile_cmd)

    def shell(
        self,
        shell_commands: List[str],
    ) -> "_Image":
        """Overwrite default shell for the image."""
        args_str = _flatten_str_args("shell", "shell_commands", shell_commands)
        args_str = '"' + '", "'.join(args_str) + '"' if args_str else ""
        dockerfile_cmd = f"SHELL [{args_str}]"

        return self.dockerfile_commands(dockerfile_cmd)

    def run_commands(
        self,
        *commands: Union[str, List[str]],
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
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
            gpu_config=parse_gpu_config(gpu),
            force_build=self.force_build or force_build,
        )

    @staticmethod
    def conda(python_version: Optional[str] = None, force_build: bool = False):
        """mdmd:hidden"""
        message = (
            "`Image.conda` is deprecated."
            " Please use the faster and more reliable `Image.micromamba` constructor instead."
        )
        deprecation_error((2024, 5, 2), message)

    def conda_install(
        self,
        *packages: Union[str, List[str]],  # A list of Python packages, eg. ["numpy", "matplotlib>=3.5.0"]
        channels: List[str] = [],  # A list of Conda channels, eg. ["conda-forge", "nvidia"]
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
    ):
        """mdmd:hidden"""
        message = (
            "`Image.conda_install` is deprecated."
            " Please use the faster and more reliable `Image.micromamba_install` instead."
        )
        deprecation_error((2024, 5, 2), message)

    def conda_update_from_environment(
        self,
        environment_yml: str,
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
        *,
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
    ):
        """mdmd:hidden"""
        message = (
            "Image.conda_update_from_environment` is deprecated."
            " Please use the `Image.micromamba_install` method (with the `spec_file` parameter) instead."
        )
        deprecation_error((2024, 5, 2), message)

    @staticmethod
    def micromamba(
        python_version: Optional[str] = None,
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
    ) -> "_Image":
        """A Micromamba base image. Micromamba allows for fast building of small Conda-based containers."""

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            nonlocal python_version
            if version == "2023.12" and python_version is None:
                python_version = "3.9"  # Backcompat for old hardcoded default param
            validated_python_version = _validate_python_version(python_version, version)
            micromamba_version = _base_image_config("micromamba", version)
            debian_codename = _base_image_config("debian", version)
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
        # A local path to a file containing package specifications
        spec_file: Optional[str] = None,
        # A list of Conda channels, eg. ["conda-forge", "nvidia"].
        channels: List[str] = [],
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
    ) -> "_Image":
        """Install a list of additional packages using micromamba."""
        pkgs = _flatten_str_args("micromamba_install", "packages", packages)
        if not pkgs and spec_file is None:
            return self

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            package_args = " ".join(shlex.quote(pkg) for pkg in pkgs)
            channel_args = "".join(f" -c {channel}" for channel in channels)

            space = " " if package_args else ""
            remote_spec_file = "" if spec_file is None else f"/{os.path.basename(spec_file)}"
            file_arg = "" if spec_file is None else f"{space}-f {remote_spec_file} -n base"
            copy_commands = [] if spec_file is None else [f"COPY {remote_spec_file} {remote_spec_file}"]

            commands = [
                "FROM base",
                *copy_commands,
                f"RUN micromamba install {package_args}{file_arg}{channel_args} --yes",
            ]
            context_files = {} if spec_file is None else {remote_spec_file: os.path.expanduser(spec_file)}
            return DockerfileSpec(commands=commands, context_files=context_files)

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
            _validate_python_version(add_python, builder_version, allow_micro_granularity=False)
            add_python_commands = [
                "COPY /python/. /usr/local",
                "RUN ln -s /usr/local/bin/python3 /usr/local/bin/python",
                "ENV TERMINFO_DIRS=/etc/terminfo:/lib/terminfo:/usr/share/terminfo:/usr/lib/terminfo",
            ]

        # Note: this change is because we install dependencies with uv in 2024.10+
        requirements_prefix = "python -m " if builder_version < "2024.10" else ""
        modal_requirements_commands = [
            f"COPY {CONTAINER_REQUIREMENTS_PATH} {CONTAINER_REQUIREMENTS_PATH}",
            f"RUN python -m pip install --upgrade {_base_image_config('package_tools', builder_version)}",
            f"RUN {requirements_prefix}{_get_modal_requirements_command(builder_version)}",
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
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
        add_python: Optional[str] = None,
        **kwargs,
    ) -> "_Image":
        """Build a Modal image from a public or private image registry, such as Docker Hub.

        The image must be built for the `linux/amd64` platform.

        If your image does not come with Python installed, you can use the `add_python` parameter
        to specify a version of Python to add to the image. Otherwise, the image is expected to
        have Python on PATH as `python`, along with `pip`.

        You may also use `setup_dockerfile_commands` to run Dockerfile commands before the
        remaining commands run. This might be useful if you want a custom Python installation or to
        set a `SHELL`. Prefer `run_commands()` when possible though.

        To authenticate against a private registry with static credentials, you must set the `secret` parameter to
        a `modal.Secret` containing a username (`REGISTRY_USERNAME`) and
        an access token or password (`REGISTRY_PASSWORD`).

        To authenticate against private registries with credentials from a cloud provider,
        use `Image.from_gcp_artifact_registry()` or `Image.from_aws_ecr()`.

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
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
        add_python: Optional[str] = None,
        **kwargs,
    ) -> "_Image":
        """Build a Modal image from a private image in Google Cloud Platform (GCP) Artifact Registry.

        You will need to pass a `modal.Secret` containing [your GCP service account key data](https://cloud.google.com/iam/docs/keys-create-delete#creating)
        as `SERVICE_ACCOUNT_JSON`. This can be done from the [Secrets](/secrets) page.
        Your service account should be granted a specific role depending on the GCP registry used:

        - For Artifact Registry images (`pkg.dev` domains) use
          the ["Artifact Registry Reader"](https://cloud.google.com/artifact-registry/docs/access-control#roles) role
        - For Container Registry images (`gcr.io` domains) use
          the ["Storage Object Viewer"](https://cloud.google.com/artifact-registry/docs/transition/setup-gcr-repo) role

        **Note:** This method does not use `GOOGLE_APPLICATION_CREDENTIALS` as that
        variable accepts a path to a JSON file, not the actual JSON string.

        See `Image.from_registry()` for information about the other parameters.

        **Example**

        ```python
        modal.Image.from_gcp_artifact_registry(
            "us-east1-docker.pkg.dev/my-project-1234/my-repo/my-image:my-version",
            secret=modal.Secret.from_name(
                "my-gcp-secret",
                required_keys=["SERVICE_ACCOUNT_JSON"],
            ),
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
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
        add_python: Optional[str] = None,
        **kwargs,
    ) -> "_Image":
        """Build a Modal image from a private image in AWS Elastic Container Registry (ECR).

        You will need to pass a `modal.Secret` containing `AWS_ACCESS_KEY_ID`,
        `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` to access the target ECR registry.

        IAM configuration details can be found in the AWS documentation for
        ["Private repository policies"](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-policies.html).

        See `Image.from_registry()` for information about the other parameters.

        **Example**

        ```python
        modal.Image.from_aws_ecr(
            "000000000000.dkr.ecr.us-east-1.amazonaws.com/my-private-registry:my-version",
            secret=modal.Secret.from_name(
                "aws",
                required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
            ),
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
        # Filepath to Dockerfile.
        path: Union[str, Path],
        # modal.Mount with local files to supply as build context for COPY commands.
        # NOTE: The remote_path of the Mount should match the Dockerfile's WORKDIR.
        context_mount: Optional[_Mount] = None,
        # Ignore cached builds, similar to 'docker build --no-cache'
        force_build: bool = False,
        *,
        secrets: Sequence[_Secret] = [],
        gpu: GPU_T = None,
        add_python: Optional[str] = None,
    ) -> "_Image":
        """Build a Modal image from a local Dockerfile.

        If your Dockerfile does not have Python installed, you can use the `add_python` parameter
        to specify a version of Python to add to the image.

        **Example**

        ```python
        image = modal.Image.from_dockerfile("./Dockerfile", add_python="3.12")
        ```

        If your Dockerfile uses `COPY` instructions which copy data from the local context of the
        build into the image, this local data must be uploaded to Modal via a context mount:

        ```python
        image = modal.Image.from_dockerfile(
            "./Dockerfile",
            context_mount=modal.Mount.from_local_dir(
                local_path="src",
                remote_path=".",  # to current WORKDIR
            ),
        )
        ```

        The context mount will allow a `COPY src/ src/` instruction to succeed in Modal's remote builder.
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
        if isinstance(python_version, float):
            raise TypeError("The `python_version` argument should be a string, not a float.")

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            requirements_path = _get_modal_requirements_path(version, python_version)
            context_files = {CONTAINER_REQUIREMENTS_PATH: requirements_path}
            full_python_version = _dockerhub_python_version(version, python_version)
            debian_codename = _base_image_config("debian", version)

            commands = [
                f"FROM python:{full_python_version}-slim-{debian_codename}",
                f"COPY {CONTAINER_REQUIREMENTS_PATH} {CONTAINER_REQUIREMENTS_PATH}",
                "RUN apt-get update",
                "RUN apt-get install -y gcc gfortran build-essential",
                f"RUN pip install --upgrade {_base_image_config('package_tools', version)}",
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
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
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
        raw_f: Callable[..., Any],
        secrets: Sequence[_Secret] = (),  # Optional Modal Secret objects with environment variables for the container
        gpu: Union[
            GPU_T, List[GPU_T]
        ] = None,  # GPU request as string ("any", "T4", ...), object (`modal.GPU.A100()`, ...), or a list of either
        mounts: Sequence[_Mount] = (),  # Mounts attached to the function
        volumes: Dict[Union[str, PurePosixPath], Union[_Volume, _CloudBucketMount]] = {},  # Volume mount paths
        network_file_systems: Dict[Union[str, PurePosixPath], _NetworkFileSystem] = {},  # NFS mount paths
        cpu: Optional[float] = None,  # How many CPU cores to request. This is a soft limit.
        memory: Optional[int] = None,  # How much memory to request, in MiB. This is a soft limit.
        timeout: Optional[int] = 60 * 60,  # Maximum execution time of the function in seconds.
        force_build: bool = False,  # Ignore cached builds, similar to 'docker build --no-cache'
        cloud: Optional[str] = None,  # Cloud provider to run the function on. Possible values are aws, gcp, oci, auto.
        region: Optional[Union[str, Sequence[str]]] = None,  # Region or regions to run the function on.
        args: Sequence[Any] = (),  # Positional arguments to the function.
        kwargs: Dict[str, Any] = {},  # Keyword arguments to the function.
    ) -> "_Image":
        """Run user-defined function `raw_f` as an image build step. The function runs just like an ordinary Modal
        function, and any kwargs accepted by `@app.function` (such as `Mount`s, `NetworkFileSystem`s,
        and resource requests) can be supplied to it.
        After it finishes execution, a snapshot of the resulting container file system is saved as an image.

        **Note**

        Only the source code of `raw_f`, the contents of `**kwargs`, and any referenced *global* variables
        are used to determine whether the image has changed and needs to be rebuilt.
        If this function references other functions or variables, the image will not be rebuilt if you
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

        if not callable(raw_f):
            raise InvalidError(f"Argument to Image.run_function must be a function, not {type(raw_f).__name__}.")
        elif raw_f.__name__ == "<lambda>":
            # It may be possible to support lambdas eventually, but for now we don't handle them well, so reject quickly
            raise InvalidError("Image.run_function does not support lambda functions.")

        scheduler_placement = SchedulerPlacement(region=region) if region else None

        info = FunctionInfo(raw_f)

        function = _Function.from_args(
            info,
            app=None,
            image=self,  # type: ignore[reportArgumentType]  # TODO: probably conflict with type stub?
            secrets=secrets,
            gpu=gpu,
            mounts=mounts,
            volumes=volumes,
            network_file_systems=network_file_systems,
            cloud=cloud,
            scheduler_placement=scheduler_placement,
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
        """Sets the environment variables in an Image.

        **Example**

        ```python
        image = (
            modal.Image.debian_slim()
            .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
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

    def workdir(self, path: Union[str, PurePosixPath]) -> "_Image":
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
            commands = ["FROM base", f"WORKDIR {shlex.quote(str(path))}"]
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

    @live_method_gen
    async def _logs(self) -> typing.AsyncGenerator[str, None]:
        """Streams logs from an image, or returns logs from an already completed image.

        This method is considered private since its interface may change - use it at your own risk!
        """
        last_entry_id: str = ""

        request = api_pb2.ImageJoinStreamingRequest(
            image_id=self._object_id, timeout=55, last_entry_id=last_entry_id, include_logs_for_finished=True
        )
        async for response in self._client.stub.ImageJoinStreaming.unary_stream(request):
            if response.result.status:
                return
            if response.entry_id:
                last_entry_id = response.entry_id
            for task_log in response.task_logs:
                if task_log.data:
                    yield task_log.data


Image = synchronize_api(_Image)
