# Copyright Modal Labs 2022
import contextlib
import json
import os
import re
import shlex
import sys
import typing
import warnings
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from functools import wraps
from inspect import isfunction
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Concatenate,
    Literal,
    cast,
    get_args,
)

import typing_extensions
from google.protobuf.message import Message
from grpclib.exceptions import StreamTerminatedError
from typing_extensions import Self

from modal._serialization import serialize_data_format
from modal_proto import api_pb2

from ._environments import _get_environment_cached
from ._load_context import LoadContext
from ._object import _Object, live_method_gen
from ._resolver import Resolver
from ._serialization import get_preferred_payload_format, serialize
from ._utils.async_utils import TaskContext, deprecate_aio_usage, synchronizer
from ._utils.blob_utils import MAX_OBJECT_SIZE_BYTES
from ._utils.docker_utils import (
    extract_copy_command_patterns,
    find_dockerignore_file,
)
from ._utils.function_utils import FunctionInfo, parse_gpu_config
from ._utils.mount_utils import validate_only_modal_volumes, validate_volumes_by_object_id
from ._utils.name_utils import check_object_name
from .client import _Client
from .cloud_bucket_mount import _CloudBucketMount
from .config import config, logger, user_config_path
from .exception import (
    ExecutionError,
    InternalError,
    InvalidError,
    NotFoundError,
    RemoteError,
    ServiceError,
    VersionError,
)
from .file_pattern_matcher import NON_PYTHON_FILES, FilePatternMatcher, _ignore_fn
from .mount import _Mount, python_standalone_mount_name
from .network_file_system import _NetworkFileSystem
from .output import OutputManager
from .secret import _Secret
from .volume import _Volume, _volume_to_mount_proto

if typing.TYPE_CHECKING:
    import modal
    import modal._functions
    import modal.client

# This is used for both type checking and runtime validation
ImageBuilderVersion = Literal["2023.12", "2024.04", "2024.10", "2025.06", "PREVIEW"]

# Note: we also define supported Python versions via logic at the top of the package __init__.py
# so that we fail fast / clearly in unsupported containers. Additionally, we enumerate the supported
# Python versions in mount.py where we specify the "standalone Python versions" we create mounts for.
# Consider consolidating these multiple sources of truth?
SUPPORTED_PYTHON_SERIES: dict[ImageBuilderVersion, list[str]] = {
    "PREVIEW": ["3.10", "3.11", "3.12", "3.13", "3.14", "3.14t"],
    "2025.06": ["3.10", "3.11", "3.12", "3.13", "3.14", "3.14t"],
    "2024.10": ["3.10", "3.11", "3.12", "3.13"],
    "2024.04": ["3.10", "3.11", "3.12"],
    "2023.12": ["3.10", "3.11", "3.12"],
}

LOCAL_REQUIREMENTS_DIR = Path(__file__).parent / "builder"
CONTAINER_REQUIREMENTS_PATH = "/modal_requirements.txt"


class _AutoDockerIgnoreSentinel:
    def __repr__(self) -> str:
        return f"{__name__}.AUTO_DOCKERIGNORE"

    def __call__(self, _: Path) -> bool:
        raise NotImplementedError("This is only a placeholder. Do not call")


AUTO_DOCKERIGNORE = _AutoDockerIgnoreSentinel()

COPY_DEPRECATION_MESSAGE_PATTERN = """modal.Image.copy_* methods will soon be deprecated.

Use {replacement} instead, which is functionally and performance-wise equivalent.

See https://modal.com/docs/guide/modal-1-0-migration for more details.
"""

P = typing_extensions.ParamSpec("P")


def _validate_image_name(name: str) -> None:
    check_object_name(name, "Image")
    # Reserve the "im-" prefix for image IDs so CLI args can be parsed
    # unambiguously as either an image ID or an image name.
    if name.startswith("im-"):
        raise InvalidError("Image name cannot start with 'im-' (reserved for image IDs).")


def _validate_image_tag(tag: str) -> None:
    check_object_name(tag, "Image tag")


def _parse_named_image_ref(name: str) -> tuple[str, str]:
    """Parse an image reference, returning (namespace_prefix, name_tag).

    If the name contains a '/', the part before the last '/' is extracted as
    a namespace prefix (intended for environment/name or workspace/env/name
    syntax). The actual image name (after the last '/') is validated as a
    standard object name.

    Returns a tuple of (prefix, "full_name:tag") where prefix is empty string
    if no '/' is present.
    """
    image_name, sep, tag = name.partition(":")
    if not sep:
        tag = "latest"

    prefix = ""
    if "/" in image_name:
        prefix, image_name = image_name.rsplit("/", 1)
        if not prefix:
            raise InvalidError("Invalid Image name: '/' prefix must be non-empty.")
        if not image_name:
            raise InvalidError("Invalid Image name: name after '/' must be non-empty.")

    _validate_image_name(image_name)
    _validate_image_tag(tag)

    full_name = f"{prefix}/{image_name}" if prefix else image_name
    return prefix, f"{full_name}:{tag}"


def _validate_python_version(
    python_version: str | None,
    builder_version: ImageBuilderVersion,
    allow_micro_granularity: bool = True,
    allow_free_threading: bool = False,
    caller_name: str = "",
) -> str:
    if python_version is None:
        # If Python version is unspecified, match the local version, up to the minor component
        python_version = series_version = "{}.{}".format(*sys.version_info)
    elif not isinstance(python_version, str):
        raise InvalidError(f"Python version must be specified as a string, not {type(python_version).__name__}")
    elif not re.match(r"^3(?:\.\d{1,2}){1,2}(rc\d*)?t?$", python_version):
        raise InvalidError(f"Invalid Python version: {python_version!r}")
    elif not allow_free_threading and python_version.endswith("t"):
        context = f"with {caller_name}" if caller_name else ""
        raise InvalidError(f"Free threaded Python is not supported {context}")
    else:
        components = python_version.split(".")
        if len(components) == 3 and not allow_micro_granularity:
            raise InvalidError(
                "Python version must be specified as 'major.minor' for this interface;"
                f" micro-level specification ({python_version!r}) is not valid."
            )
        suffix = "t" if len(components) == 3 and python_version.endswith("t") else ""
        series_version = f"{components[0]}.{components[1]}{suffix}"

    supported_series = SUPPORTED_PYTHON_SERIES[builder_version]
    if series_version not in supported_series:
        raise InvalidError(
            f"Unsupported Python version: {python_version!r}."
            f" When using the {builder_version!r} Image builder, Modal supports the following series:"
            f" {supported_series!r}."
        )
    return python_version


def _dockerhub_python_version(
    builder_version: ImageBuilderVersion,
    python_version: str | None = None,
    allow_free_threading: bool = False,
    caller_name: str = "",
) -> str:
    python_version = _validate_python_version(
        python_version, builder_version, allow_free_threading=allow_free_threading, caller_name=caller_name
    )
    version_components = python_version.split(".")

    # When user specifies a full Python version, use that
    if len(version_components) > 2:
        return python_version

    # Otherwise, use the same series, but a specific micro version, corresponding to the latest
    # available from https://hub.docker.com/_/python at the time of each image builder release.
    # This allows us to publish one pre-built debian-slim image per Python series.
    python_versions = _base_image_config("python", builder_version)
    series_to_micro_version = dict(tuple(v.rsplit(".", 1)) for v in python_versions)
    python_series_requested = "{}.{}".format(*version_components)
    micro_version = series_to_micro_version[python_series_requested]
    return f"{python_series_requested}.{micro_version}"


def _base_image_config(group: str, builder_version: ImageBuilderVersion) -> Any:
    with open(LOCAL_REQUIREMENTS_DIR / "base-images.json") as f:
        data = json.load(f)
    return data[group][builder_version]


def _get_modal_requirements_path(builder_version: ImageBuilderVersion, python_version: str | None = None) -> str:
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


def _flatten_str_args(function_name: str, arg_name: str, args: Sequence[str | list[str]]) -> list[str]:
    """Takes a sequence of strings, or string lists, and flattens it.

    Raises an error if any of the elements are not strings or string lists.
    """

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


def _validate_packages(packages: list[str]) -> bool:
    """Validates that a list of packages does not contain any command-line options."""
    return not any(pkg.startswith("-") for pkg in packages)


def _make_pip_install_args(
    find_links: str | None = None,  # Passes -f (--find-links) pip install
    index_url: str | None = None,  # Passes -i (--index-url) to pip install
    extra_index_url: str | None = None,  # Passes --extra-index-url to pip install
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

    supported_versions: set[ImageBuilderVersion] = set(get_args(ImageBuilderVersion))
    if version not in supported_versions:
        if local_config_version is not None:
            update_suggestion = "or remove your local configuration"
        elif version < min(supported_versions):
            update_suggestion = "your image builder version using the Modal dashboard"
        else:
            update_suggestion = "your client library (pip install --upgrade modal)"
        preview_versions: set[ImageBuilderVersion] = {"PREVIEW"}
        suggested_versions = supported_versions - preview_versions
        raise VersionError(
            "This version of the modal client supports the following image builder versions:"
            f" {suggested_versions!r}."
            f"\n\nYou are using {version!r}{version_source}."
            f" Please update {update_suggestion}."
        )

    return version


def _create_context_mount(
    docker_commands: Sequence[str],
    ignore_fn: Callable[[Path], bool],
    context_dir: Path,
) -> _Mount | None:
    """
    Creates a context mount from a list of docker commands.

    1. Paths are evaluated relative to context_dir.
    2. First selects inclusions based on COPY commands in the list of commands.
    3. Then ignore any files as per the ignore predicate.
    """
    copy_patterns = extract_copy_command_patterns(docker_commands)
    if not copy_patterns:
        return None  # no mount needed
    include_fn = FilePatternMatcher(*copy_patterns)

    def ignore_with_include(source: Path) -> bool:
        if source.is_absolute():
            source = source.relative_to(context_dir)
        if not include_fn(source) or ignore_fn(source):
            return True

        return False

    return _Mount._add_local_dir(context_dir, PurePosixPath("/"), ignore=ignore_with_include)


def _create_context_mount_function(
    ignore: Sequence[str] | Callable[[Path], bool] | _AutoDockerIgnoreSentinel,
    dockerfile_cmds: list[str] = [],
    dockerfile_path: Path | None = None,
    context_dir: Path | str | None = None,
):
    if dockerfile_path and dockerfile_cmds:
        raise InvalidError("Cannot provide both dockerfile and docker commands")

    if ignore is AUTO_DOCKERIGNORE:

        def auto_created_context_mount_fn() -> _Mount | None:
            nonlocal context_dir
            context_dir = Path.cwd() if context_dir is None else Path(context_dir).absolute()
            dockerignore_file = find_dockerignore_file(context_dir, dockerfile_path)
            ignore_fn = (
                FilePatternMatcher(*dockerignore_file.read_text("utf8").splitlines())
                if dockerignore_file
                else _ignore_fn(())
            )

            cmds = dockerfile_path.read_text("utf8").splitlines() if dockerfile_path else dockerfile_cmds
            return _create_context_mount(cmds, ignore_fn=ignore_fn, context_dir=context_dir)

        return auto_created_context_mount_fn

    else:

        def auto_created_context_mount_fn() -> _Mount | None:
            # use COPY commands and ignore patterns to construct implicit context mount
            nonlocal context_dir
            context_dir = Path.cwd() if context_dir is None else Path(context_dir).absolute()
            cmds = dockerfile_path.read_text("utf8").splitlines() if dockerfile_path else dockerfile_cmds
            return _create_context_mount(cmds, ignore_fn=_ignore_fn(ignore), context_dir=context_dir)

        return auto_created_context_mount_fn


class _ImageRegistryConfig:
    """mdmd:hidden"""

    def __init__(
        self,
        # TODO: change to _PUBLIC after worker starts handling it.
        registry_auth_type: "api_pb2.RegistryAuthType.ValueType" = api_pb2.REGISTRY_AUTH_TYPE_UNSPECIFIED,
        secret: _Secret | None = None,
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
    commands: list[str] = field(default_factory=list)
    context_files: dict[str, str] = field(default_factory=dict)


async def _image_await_build_result(image_id: str, client: _Client) -> api_pb2.ImageJoinStreamingResponse:
    last_entry_id: str = ""
    result_response: api_pb2.ImageJoinStreamingResponse | None = None

    async def join():
        nonlocal last_entry_id, result_response

        request = api_pb2.ImageJoinStreamingRequest(image_id=image_id, timeout=55, last_entry_id=last_entry_id)
        async for response in client.stub.ImageJoinStreaming.unary_stream(request):
            if response.entry_id:
                last_entry_id = response.entry_id
            if response.result.status:
                result_response = response
                # can't return yet, since there may still be logs streaming back in subsequent responses
            output_mgr = OutputManager.get()
            for task_log in response.task_logs:
                if task_log.task_progress.pos or task_log.task_progress.len:
                    assert task_log.task_progress.progress_type == api_pb2.IMAGE_SNAPSHOT_UPLOAD
                    output_mgr.update_snapshot_progress(image_id, task_log.task_progress)
                elif task_log.data:
                    await output_mgr.put_streaming_log(task_log)
        OutputManager.get().flush_lines()

    # Handle up to n exceptions while fetching logs
    retry_count = 0
    while result_response is None:
        try:
            await join()
        except (ServiceError, InternalError, StreamTerminatedError) as exc:
            retry_count += 1
            if retry_count >= 3:
                raise exc
    return result_response


def _requires_image_instance(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not isinstance(self, _Image):
            raise InvalidError(
                "Image has not been constructed yet. "
                f"Use one of the static factory methods prior to calling {method.__name__} "
                "like `modal.Image.debian_slim`, `modal.Image.from_registry`, or `modal.Image.micromamba`"
            )
        return method(self, *args, **kwargs)

    return wrapper


class _Image(_Object, type_prefix="im"):
    """Base class for container images to run functions in.

    Do not construct this class directly; instead use one of its static factory methods,
    such as `modal.Image.debian_slim`, `modal.Image.from_registry`, or `modal.Image.micromamba`.
    """

    force_build: bool
    inside_exceptions: list[Exception]
    _serve_mounts: frozenset[_Mount]  # used for mounts watching in `modal serve`
    _deferred_mounts: Sequence[
        _Mount
    ]  # added as mounts on any container referencing the Image, see `def _mount_layers`
    _added_python_source_set: frozenset[str]  # used to warn about missing mounts during auto-mount deprecation
    _metadata: api_pb2.ImageMetadata | None = None  # set on hydration, private for now
    _is_empty: bool

    def _initialize_from_empty(self):
        self.inside_exceptions = []
        self._serve_mounts = frozenset()
        self._deferred_mounts = ()
        self._added_python_source_set = frozenset()
        self.force_build = False
        self._is_empty = False

    def _initialize_from_other(self, other: "_Image"):
        self.inside_exceptions = other.inside_exceptions
        self.force_build = other.force_build
        self._serve_mounts = other._serve_mounts
        self._deferred_mounts = other._deferred_mounts
        self._added_python_source_set = other._added_python_source_set
        self._is_empty = False

    def _get_metadata(self) -> Message | None:
        return self._metadata

    def _hydrate_metadata(self, metadata: Message | None):
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
            return self._copy_mount(mount, remote_path="/")

        base_image = self

        async def _load(self2: "_Image", resolver: Resolver, load_context: LoadContext, existing_object_id: str | None):
            self2._hydrate_from_other(base_image)  # same image id as base image as long as it's lazy
            self2._deferred_mounts = tuple(base_image._deferred_mounts) + (mount,)
            self2._serve_mounts = base_image._serve_mounts | ({mount} if mount.is_local() else set())

        img = _Image._from_loader(
            _load, "Image(local files)", deps=lambda: [base_image, mount], load_context_overrides=LoadContext.empty()
        )
        img._added_python_source_set = base_image._added_python_source_set
        return img

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
        base_images: dict[str, "_Image"] | None = None,
        dockerfile_function: Callable[[ImageBuilderVersion], DockerfileSpec] | None = None,
        secrets: Collection[_Secret] | None = None,
        gpu_config: api_pb2.GPUConfig | None = None,
        build_function: "modal._functions._Function | None" = None,
        build_function_input: api_pb2.FunctionInput | None = None,
        image_registry_config: _ImageRegistryConfig | None = None,
        context_mount_function: Callable[[], _Mount | None] | None = None,
        force_build: bool = False,
        build_args: dict[str, str] = {},
        validated_volumes: Sequence[tuple[str, _Volume]] | None = None,
        # For internal use only.
        _namespace: "api_pb2.DeploymentNamespace.ValueType" = api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
        _do_assert_no_mount_layers: bool = True,
    ):
        if base_images is None:
            base_images = {}

        if validated_volumes is None:
            validated_volumes = []

        if secrets is None:
            secrets = []
        if gpu_config is None:
            gpu_config = api_pb2.GPUConfig()
        if image_registry_config is None:
            image_registry_config = _ImageRegistryConfig()

        for secret in secrets:
            if not isinstance(secret, _Secret):
                raise InvalidError("All secrets of an Image need to be modal.Secret instances.")

        if build_function and len(base_images) != 1:
            raise InvalidError("Cannot run a build function with multiple base images!")

        def _deps() -> Sequence[_Object]:
            deps: tuple[_Object, ...] = tuple(base_images.values()) + tuple(secrets)
            if build_function:
                deps += (build_function,)
            if image_registry_config and image_registry_config.secret:
                deps += (image_registry_config.secret,)
            for _, vol in validated_volumes:
                deps += (vol,)
            return deps

        async def _load(self: _Image, resolver: Resolver, load_context: LoadContext, existing_object_id: str | None):
            context_mount = context_mount_function() if context_mount_function else None
            if context_mount:
                await resolver.load(context_mount, load_context)

            if _do_assert_no_mount_layers:
                for image in base_images.values():
                    # base images can't have
                    image._assert_no_mount_layers()

            assert load_context.app_id  # type narrowing
            environment = await _get_environment_cached(load_context.environment_name or "", load_context.client)
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

            # Validate that the same volume (by object_id) isn't mounted at multiple paths
            validate_volumes_by_object_id(validated_volumes)

            # Relies on dicts being ordered (true as of Python 3.6).
            volume_mounts = [_volume_to_mount_proto(path, volume) for path, volume in validated_volumes]

            image_definition = api_pb2.Image(
                base_images=base_images_pb2s,
                dockerfile_commands=dockerfile.commands,
                context_files=context_file_pb2s,
                secret_ids=[secret.object_id for secret in secrets],
                context_mount_id=(context_mount.object_id if context_mount else ""),
                gpu_config=gpu_config,
                image_registry_config=image_registry_config.get_proto(),
                runtime=config.get("function_runtime"),
                runtime_debug=config.get("function_runtime_debug"),
                build_function=_build_function,
                build_args=build_args,
                volume_mounts=volume_mounts,
            )

            req = api_pb2.ImageGetOrCreateRequest(
                app_id=load_context.app_id,
                image=image_definition,
                existing_image_id=existing_object_id or "",  # TODO: ignored
                build_function_id=build_function_id,
                force_build=config.get("force_build") or force_build,
                namespace=_namespace,
                builder_version=builder_version,
                # Failsafe mechanism to prevent inadvertant updates to the global images.
                # Only admins can publish to the global namespace, but they have to additionally request it.
                allow_global_deployment=os.environ.get("MODAL_IMAGE_ALLOW_GLOBAL_DEPLOYMENT") == "1",
                ignore_cache=config.get("ignore_cache"),
            )
            resp = await load_context.client.stub.ImageGetOrCreate(req)
            image_id = resp.image_id
            result: api_pb2.GenericResult
            metadata: api_pb2.ImageMetadata | None = None

            if resp.result.status:
                # image already built
                result = resp.result
                if resp.HasField("metadata"):
                    metadata = resp.metadata
            else:
                # not built or in the process of building - wait for build
                logger.debug("Waiting for image %s" % image_id)
                build_resp = await _image_await_build_result(image_id, load_context.client)
                result = build_resp.result
                if build_resp.HasField("metadata"):
                    metadata = build_resp.metadata

            if result.status == api_pb2.GenericResult.GENERIC_STATUS_FAILURE:
                if result.exception:
                    raise RemoteError(f"Image build for {image_id} failed with the exception:\n{result.exception}")
                else:
                    msg = f"Image build for {image_id} failed. See build logs for more details."
                    if not OutputManager.get().is_enabled:
                        msg += " (Hint: Use `modal.enable_output()` to see logs from the process building the Image.)"
                    raise RemoteError(msg)
            elif result.status == api_pb2.GenericResult.GENERIC_STATUS_TERMINATED:
                msg = f"Image build for {image_id} terminated due to external shut-down. Please try again."
                if result.exception:
                    msg = (
                        f"Image build for {image_id} terminated due to external shut-down with the exception:\n"
                        f"{result.exception}"
                    )
                raise RemoteError(msg)
            elif result.status == api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT:
                raise RemoteError(
                    f"Image build for {image_id} timed out. Please try again with a larger `timeout` parameter."
                )
            elif result.status == api_pb2.GenericResult.GENERIC_STATUS_SUCCESS:
                pass
            else:
                raise RemoteError("Unknown status %s!" % result.status)

            self._hydrate(image_id, load_context.client, metadata)
            local_mounts: set[_Mount] = set()
            for base in base_images.values():
                local_mounts |= base._serve_mounts
            if context_mount and context_mount.is_local():
                local_mounts.add(context_mount)
            self._serve_mounts = frozenset(local_mounts)

        rep = f"Image({dockerfile_function})"
        obj = _Image._from_loader(_load, rep, deps=_deps, load_context_overrides=LoadContext.empty())
        obj.force_build = force_build
        obj._added_python_source_set = frozenset.union(
            frozenset(), *(base._added_python_source_set for base in base_images.values())
        )
        return obj

    def _copy_mount(self, mount: _Mount, remote_path: str | Path = ".") -> "_Image":
        """mdmd:hidden
        Internal
        """
        if not isinstance(mount, _Mount):
            raise InvalidError("The mount argument to copy has to be a Modal Mount object")

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            commands = ["FROM base", f"COPY . {remote_path}"]  # copy everything from the supplied mount
            return DockerfileSpec(commands=commands, context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            context_mount_function=lambda: mount,
        )

    @_requires_image_instance
    def add_local_file(self, local_path: str | Path, remote_path: str, *, copy: bool = False) -> "_Image":
        """Adds a local file to the image at `remote_path` within the container.

        By default (`copy=False`), the files are added to containers on startup and are not built into the actual Image,
        which speeds up deployment.

        Set `copy=True` to copy the files into an Image layer at build time instead, similar to how
        [`COPY`](https://docs.docker.com/engine/reference/builder/#copy) works in a `Dockerfile`.

        copy=True can slow down iteration since it requires a rebuild of the Image and any subsequent
        build steps whenever the included files change, but it is required if you want to run additional
        build steps after this one.

        *Added in v0.66.40*: This method replaces the deprecated `modal.Image.copy_local_file` method.

        Args:
            local_path: Path to the file on the local machine.
            remote_path: Absolute path inside the container where the file should appear.
            copy: If True, bake the file into an image layer at build time; if False, mount at container startup.

        Returns:
            A new `Image` with the file layer or mount applied.
        """
        if not PurePosixPath(remote_path).is_absolute():
            # TODO(elias): implement relative to absolute resolution using image workdir metadata
            #  + make default remote_path="./"
            #  This requires deferring the Mount creation until after "self" (the base image) has been resolved
            #  so we know the workdir of the operation.
            raise InvalidError("image.add_local_file() currently only supports absolute remote_path values")

        if remote_path.endswith("/"):
            remote_path = remote_path + Path(local_path).name

        mount = _Mount._from_local_file(local_path, remote_path)
        return self._add_mount_layer_or_copy(mount, copy=copy)

    @_requires_image_instance
    def add_local_dir(
        self,
        local_path: str | Path,
        remote_path: str,
        *,
        copy: bool = False,
        ignore: Sequence[str] | Callable[[Path], bool] = [],
    ) -> "_Image":
        """Adds a local directory's content to the image at `remote_path` within the container.

        By default (`copy=False`), the files are added to containers on startup and are not built into the actual Image,
        which speeds up deployment.

        Set `copy=True` to copy the files into an Image layer at build time instead, similar to how
        [`COPY`](https://docs.docker.com/engine/reference/builder/#copy) works in a `Dockerfile`.

        copy=True can slow down iteration since it requires a rebuild of the Image and any subsequent
        build steps whenever the included files change, but it is required if you want to run additional
        build steps after this one.

        *Added in v0.66.40*: This method replaces the deprecated `modal.Image.copy_local_dir` method.

        Args:
            local_path: Path to the directory on the local machine.
            remote_path: Absolute path inside the container where the directory contents should appear.
            copy: If True, bake the tree into an image layer at build time; if False, mount at container startup.
            ignore:
                Predicate or pattern list for file exclusion (True means exclude). A sequence is converted to a
                dockerignore-style matcher.

        Returns:
            A new `Image` with the directory layer or mount applied.

        Examples:
            ```python
            from modal import FilePatternMatcher

            image = modal.Image.debian_slim().add_local_dir(
                "~/assets",
                remote_path="/assets",
                ignore=["*.venv"],
            )

            image = modal.Image.debian_slim().add_local_dir(
                "~/assets",
                remote_path="/assets",
                ignore=lambda p: p.is_relative_to(".venv"),
            )

            image = modal.Image.debian_slim().add_local_dir(
                "~/assets",
                remote_path="/assets",
                ignore=FilePatternMatcher("**/*.txt"),
            )

            # When including files is simpler than excluding them, you can use the `~` operator to invert the matcher.
            image = modal.Image.debian_slim().add_local_dir(
                "~/assets",
                remote_path="/assets",
                ignore=~FilePatternMatcher("**/*.py"),
            )

            # You can also read ignore patterns from a file.
            image = modal.Image.debian_slim().add_local_dir(
                "~/assets",
                remote_path="/assets",
                ignore=FilePatternMatcher.from_file("/path/to/ignorefile"),
            )
            ```
        """
        if not PurePosixPath(remote_path).is_absolute():
            # TODO(elias): implement relative to absolute resolution using image workdir metadata
            #  + make default remote_path="./"
            raise InvalidError("image.add_local_dir() currently only supports absolute remote_path values")

        mount = _Mount._add_local_dir(Path(local_path), PurePosixPath(remote_path), ignore=_ignore_fn(ignore))
        return self._add_mount_layer_or_copy(mount, copy=copy)

    @_requires_image_instance
    def add_local_python_source(
        self, *modules: str, copy: bool = False, ignore: Sequence[str] | Callable[[Path], bool] = NON_PYTHON_FILES
    ) -> "_Image":
        """Adds locally available Python packages/modules to containers.

        Adds all files from the specified Python package or module to containers running the Image.

        Packages are added to the `/root` directory of containers, which is on the `PYTHONPATH`
        of any executed Modal Functions, enabling import of the module by that name.

        By default (`copy=False`), the files are added to containers on startup and are not built into the actual Image,
        which speeds up deployment.

        Set `copy=True` to copy the files into an Image layer at build time instead. This can slow down iteration since
        it requires a rebuild of the Image and any subsequent build steps whenever the included files change, but it is
        required if you want to run additional build steps after this one.

        **Note:** This excludes all dot-prefixed subdirectories or files and all `.pyc`/`__pycache__` files.
        To add full directories with finer control, use `.add_local_dir()` instead and specify `/root` as
        the destination directory.

        By default only includes `.py`-files in the source modules. Set the `ignore` argument to a list of patterns
        or a callable to override this behavior.

        *Added in v0.67.28*: This method replaces the deprecated `modal.Mount.from_local_python_packages` pattern.

        Args:
            *modules: Python package or module names to include from the local project.
            copy: If True, bake sources into an image layer; if False, mount at container startup.
            ignore: Patterns or callable controlling which files to exclude.

        Returns:
            A new `Image` with the Python source mount or layer applied.

        Examples:
            ```py
            # includes everything except data.json
            modal.Image.debian_slim().add_local_python_source("mymodule", ignore=["data.json"])

            # exclude large files
            modal.Image.debian_slim().add_local_python_source(
                "mymodule",
                ignore=lambda p: p.stat().st_size > 1e9
            )
            ```
        """
        if not all(isinstance(module, str) for module in modules):
            raise InvalidError("Local Python modules must be specified as strings.")
        mount = _Mount._from_local_python_packages(*modules, ignore=ignore)
        img = self._add_mount_layer_or_copy(mount, copy=copy)
        img._added_python_source_set |= set(modules)
        return img

    @deprecate_aio_usage((2025, 11, 14), "Image.from_id")
    @classmethod
    def from_id(cls, image_id: str, client: "modal.client.Client | None" = None) -> typing_extensions.Self:
        """Construct an Image from an id and look up the Image result.

        The ID of an Image object can be accessed using `.object_id`.

        Args:
            image_id: Image object ID to load.
            client: Optional Modal client; uses the default synchronizer client when omitted.

        Returns:
            A hydrated `Image` handle for the given ID.
        """
        _client = typing.cast(_Client, synchronizer._translate_in(client))

        async def _load(self: _Image, resolver: Resolver, load_context: LoadContext, existing_object_id: str | None):
            resp = await load_context.client.stub.ImageFromId(api_pb2.ImageFromIdRequest(image_id=image_id))
            self._hydrate(resp.image_id, load_context.client, resp.metadata)

        rep = f"Image.from_id({image_id!r})"

        obj = _Image._from_loader(_load, rep, load_context_overrides=LoadContext(client=_client))
        obj._object_id = image_id

        return typing.cast(typing_extensions.Self, synchronizer._translate_out(obj))

    async def build(self, app: "modal.app._App") -> "_Image":
        """Eagerly build an image.

        If your image was previously built, then this method will not rebuild your image
        and your cached image is returned.

        For defining Modal functions, images are built automatically when deploying or running an App.
        You do not need to build the image explicitly in that case.

        Args:
            app: Initialized app used as the load context for the image build.

        Returns:
            This image after the build (and resolver load) completes.

        Examples:
            ```python
            image = modal.Image.debian_slim().uv_pip_install("scipy", "numpy")

            app = modal.App.lookup("build-image", create_if_missing=True)
            with modal.enable_output():  # To see logs in your local terminal
                image.build(app)

            # Save the image id
            my_image_id = image.object_id

            # Reference the image with the id or uses it another context.
            built_image = modal.Image.from_id(my_image_id)
            ```

            Alternatively, you can pre-build an image and use it in a sandbox:

            ```python notest
            app = modal.App.lookup("sandbox-example", create_if_missing=True)

            with modal.enable_output():
                image = modal.Image.debian_slim().uv_pip_install("scipy")
                image.build(app)

            sb = modal.Sandbox.create("python", "-c", "import scipy; print(scipy)", app=app, image=image)
            print(sb.stdout.read())
            sb.terminate()
            ```

            ```python notest
            app = modal.App()
            image = modal.Image.debian_slim()

            # No need to explicitly build the image for defining a function.
            @app.function(image=image)
            def f():
                ...
            ```
        """
        if app.app_id is None:
            raise InvalidError("App has not been initialized yet. Use the context manager `app.run()` or `App.lookup`")

        resolver = Resolver()
        async with TaskContext() as tc:
            load_context = LoadContext(task_context=tc).merged_with(app._root_load_context)
            await resolver.load(self, load_context)
        return self

    @_requires_image_instance
    def pip_install(
        self,
        *packages: str | list[str],
        find_links: str | None = None,
        index_url: str | None = None,
        extra_index_url: str | None = None,
        pre: bool = False,
        extra_options: str = "",
        force_build: bool = False,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        gpu: str | None = None,
    ) -> "_Image":
        """Install a list of Python packages using pip.

        Args:
            *packages: Python packages to install, e.g. ``numpy`` or ``matplotlib>=3.5.0``.
            find_links: Passed as ``--find-links`` to pip.
            index_url: Passed as ``--index-url`` to pip.
            extra_index_url: Passed as ``--extra-index-url`` to pip.
            pre: If True, allow pre-release versions (``--pre``).
            extra_options: Additional raw options for pip, e.g. ``--no-build-isolation``.
            force_build: If True, skip cached image builds (similar to ``docker build --no-cache``).
            env: Environment variables set in the build container.
            secrets: Secrets injected as environment variables during the build.
            gpu: GPU type to attach to the builder container.

        Returns:
            A new `Image` with the pip install layer applied.

        Examples:
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
        elif not _validate_packages(pkgs):
            raise InvalidError(
                "Package list for `Image.pip_install` cannot contain other arguments;"
                " try the `extra_options` parameter instead."
            )

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            package_args = shlex.join(sorted(pkgs))
            extra_args = _make_pip_install_args(find_links, index_url, extra_index_url, pre, extra_options)
            commands = ["FROM base", f"RUN python -m pip install {package_args} {extra_args}"]
            if version > "2023.12":  # Back-compat for legacy trailing space with empty extra_args
                commands = [cmd.strip() for cmd in commands]
            return DockerfileSpec(commands=commands, context_files={})

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

        gpu_config = parse_gpu_config(gpu)
        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            force_build=self.force_build or force_build,
            gpu_config=gpu_config,
            secrets=secrets,
        )

    @_requires_image_instance
    def pip_install_private_repos(
        self,
        *repositories: str,
        git_user: str,
        find_links: str | None = None,
        index_url: str | None = None,
        extra_index_url: str | None = None,
        pre: bool = False,
        extra_options: str = "",
        gpu: str | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        force_build: bool = False,
    ) -> "_Image":
        """Install a list of Python packages from private git repositories using pip.

        This method currently supports Github and Gitlab only.

        - **Github:** Provide a `modal.Secret` that contains a `GITHUB_TOKEN` key-value pair
        - **Gitlab:** Provide a `modal.Secret` that contains a `GITLAB_TOKEN` key-value pair

        These API tokens should have permissions to read the list of private repositories provided as arguments.

        We recommend using Github's ['fine-grained' access tokens](https://github.blog/2022-10-18-introducing-fine-grained-personal-access-tokens-for-github/).
        These tokens are repo-scoped, and avoid granting read permission across all of a user's private repos.

        Args:
            *repositories: Git URLs without scheme, e.g. ``github.com/org/repo@ref`` or with ``#subdirectory=``.
            git_user: Username embedded in HTTPS git URLs for authentication.
            find_links: Passed as ``--find-links`` to pip.
            index_url: Passed as ``--index-url`` to pip.
            extra_index_url: Passed as ``--extra-index-url`` to pip.
            pre: If True, allow pre-release versions.
            extra_options: Additional raw options for pip.
            gpu: GPU type to attach to the builder container.
            env: Environment variables set in the build container.
            secrets: Secrets that supply ``GITHUB_TOKEN`` / ``GITLAB_TOKEN`` as required.
            force_build: If True, skip cached image builds.

        Returns:
            A new `Image` with private repositories installed.

        Examples:
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

        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

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

    @_requires_image_instance
    def pip_install_from_requirements(
        self,
        requirements_txt: str,
        find_links: str | None = None,
        *,
        index_url: str | None = None,
        extra_index_url: str | None = None,
        pre: bool = False,
        extra_options: str = "",
        force_build: bool = False,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        gpu: str | None = None,
    ) -> "_Image":
        """Install a list of Python packages from a local `requirements.txt` file.

        Args:
            requirements_txt: Path to a ``requirements.txt`` file on the local machine.
            find_links: Passed as ``--find-links`` to pip.
            index_url: Passed as ``--index-url`` to pip.
            extra_index_url: Passed as ``--extra-index-url`` to pip.
            pre: If True, allow pre-release versions.
            extra_options: Additional raw options for pip.
            force_build: If True, skip cached image builds.
            env: Environment variables set in the build container.
            secrets: Secrets injected as environment variables during the build.
            gpu: GPU type to attach to the builder container.

        Returns:
            A new `Image` with requirements installed.
        """

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

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

    @_requires_image_instance
    def pip_install_from_pyproject(
        self,
        pyproject_toml: str,
        optional_dependencies: list[str] = [],
        *,
        find_links: str | None = None,
        index_url: str | None = None,
        extra_index_url: str | None = None,
        pre: bool = False,
        extra_options: str = "",
        force_build: bool = False,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        gpu: str | None = None,
    ) -> "_Image":
        """Install dependencies specified by a local `pyproject.toml` file.

        `optional_dependencies` is a list of the keys of the
        optional-dependencies section(s) of the `pyproject.toml` file
        (e.g. test, doc, experiment, etc). When provided,
        all of the packages in each listed section are installed as well.

        Args:
            pyproject_toml: Path to a ``pyproject.toml`` using PEP 621 ``[project.dependencies]``.
            optional_dependencies: Keys under ``[project.optional-dependencies]`` to install additionally.
            find_links: Passed as ``--find-links`` to pip.
            index_url: Passed as ``--index-url`` to pip.
            extra_index_url: Passed as ``--extra-index-url`` to pip.
            pre: If True, allow pre-release versions.
            extra_options: Additional raw options for pip.
            force_build: If True, skip cached image builds.
            env: Environment variables set in the build container.
            secrets: Secrets injected as environment variables during the build.
            gpu: GPU type to attach to the builder container.

        Returns:
            A new `Image` with project dependencies installed.
        """

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

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
            package_args = shlex.join(sorted(dependencies))
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

    @_requires_image_instance
    def uv_pip_install(
        self,
        *packages: str | list[str],
        requirements: list[str] | None = None,
        find_links: str | None = None,
        index_url: str | None = None,
        extra_index_url: str | None = None,
        pre: bool = False,
        extra_options: str = "",
        force_build: bool = False,
        uv_version: str | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        gpu: str | None = None,
    ) -> "_Image":
        """Install a list of Python packages using uv pip install.

        This method assumes that:
        - Python is on the ``$PATH`` and dependencies are installed with the first Python on the ``$PATH``.
        - The shell supports ``$()``-style substitution as used in the generated Dockerfile.
        - The ``command`` builtin is available on the ``$PATH``.

        Added in v1.1.0.

        Args:
            *packages: Python packages to pass to ``uv pip install``.
            requirements: Optional list of requirement file paths (passed as ``--requirements``).
            find_links: Passed as ``--find-links`` to ``uv pip``.
            index_url: Passed as ``--index-url`` to ``uv pip``.
            extra_index_url: Passed as ``--extra-index-url`` to ``uv pip``.
            pre: If True, allow pre-releases (``--prerelease allow``).
            extra_options: Additional raw options appended to the ``uv pip install`` invocation.
            force_build: If True, skip cached image builds.
            uv_version: Pin the uv binary version copied from ``ghcr.io/astral-sh/uv``.
            env: Environment variables set in the build container.
            secrets: Secrets injected as environment variables during the build.
            gpu: GPU type to attach to the builder container.

        Returns:
            A new `Image` with packages installed via uv.

        Examples:
            ```python
            image = modal.Image.debian_slim().uv_pip_install("torch==2.7.1", "numpy")
            ```
        """

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

        pkgs = _flatten_str_args("uv_pip_install", "packages", packages)

        if requirements is None or isinstance(requirements, list):
            requirements = requirements or []
        else:
            raise InvalidError("requirements must be None or a list of strings")

        if not pkgs and not requirements:
            return self
        elif not _validate_packages(pkgs):
            raise InvalidError(
                "Package list for `Image.uv_pip_install` cannot contain other arguments;"
                " try the `extra_options` parameter instead."
            )

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            commands = ["FROM base"]
            UV_ROOT = "/.uv"
            if uv_version is None:
                commands.append(f"COPY --from=ghcr.io/astral-sh/uv:latest /uv {UV_ROOT}/uv")
            else:
                commands.append(f"COPY --from=ghcr.io/astral-sh/uv:{uv_version} /uv {UV_ROOT}/uv")

            # NOTE: Using $(command -v python) assumes:
            # - python is on the PATH and uv is installing into the first python in the PATH
            # - the shell supports $() for substitution
            # - `command` command is on the PATH
            uv_pip_args = ["--python $(command -v python)", "--compile-bytecode"]
            context_files = {}

            if find_links:
                uv_pip_args.append(f"--find-links {shlex.quote(find_links)}")
            if index_url:
                uv_pip_args.append(f"--index-url {shlex.quote(index_url)}")
            if extra_index_url:
                uv_pip_args.append(f"--extra-index-url {shlex.quote(extra_index_url)}")
            if pre:
                uv_pip_args.append("--prerelease allow")
            if extra_options:
                uv_pip_args.append(extra_options)

            if requirements:

                def _generate_paths(idx: int, req: str) -> dict:
                    local_path = os.path.expanduser(req)
                    basename = os.path.basename(req)

                    # The requirement files can have the same name but in different directories:
                    # requirements=["test/requirements.txt", "a/b/c/requirements.txt"]
                    # To uniquely identify these files, we add a `idx` prefix to every file's basename
                    # - `test/requirements.txt` -> `/.0_requirements.txt` in context -> `/.uv/0/requirements.txt` to uv
                    # - `a/b/c/requirements.txt` -> `/.1_requirements.txt` in context -> `/.uv/1/requirements.txt` to uv
                    return {
                        "local_path": local_path,
                        "context_path": f"/.{idx}_{basename}",
                        "dest_path": f"{UV_ROOT}/{idx}/{basename}",
                    }

                requirement_paths = [_generate_paths(idx, req) for idx, req in enumerate(requirements)]
                requirements_cli = " ".join(f"--requirements {req['dest_path']}" for req in requirement_paths)
                uv_pip_args.append(requirements_cli)

                commands.extend([f"COPY {req['context_path']} {req['dest_path']}" for req in requirement_paths])
                context_files.update({req["context_path"]: req["local_path"] for req in requirement_paths})

            uv_pip_args.extend(shlex.quote(p) for p in sorted(pkgs))
            uv_pip_args_joined = " ".join(uv_pip_args)

            commands.append(f"RUN {UV_ROOT}/uv pip install {uv_pip_args_joined}")

            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            force_build=self.force_build or force_build,
            gpu_config=parse_gpu_config(gpu),
            secrets=secrets,
        )

    @_requires_image_instance
    def poetry_install_from_file(
        self,
        poetry_pyproject_toml: str,
        poetry_lockfile: str | None = None,
        *,
        ignore_lockfile: bool = False,
        force_build: bool = False,
        with_: list[str] = [],
        without: list[str] = [],
        only: list[str] = [],
        poetry_version: str | None = "latest",
        old_installer: bool = False,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        gpu: str | None = None,
    ) -> "_Image":
        """Install poetry *dependencies* specified by a local `pyproject.toml` file.

        If not provided as argument the path to the lockfile is inferred. However, the
        file has to exist, unless `ignore_lockfile` is set to `True`.

        Note that the root project of the poetry project is not installed, only the dependencies.
        For including local python source files see `add_local_python_source`

        Poetry will be installed to the Image (using pip) unless `poetry_version` is set to None.
        Note that the interpretation of `poetry_version="latest"` depends on the Modal Image Builder
        version, with versions 2024.10 and earlier limiting poetry to 1.x.

        Args:
            poetry_pyproject_toml: Path to a Poetry ``pyproject.toml`` file.
            poetry_lockfile: Path to ``poetry.lock``; if omitted, inferred next to the pyproject.
            ignore_lockfile: If True, do not copy or use a lockfile even when present.
            force_build: If True, skip cached image builds.
            with_: Optional dependency groups to include (``poetry install --with``).
            without: Optional dependency groups to exclude (``poetry install --without``).
            only: Only install dependency groups in this list (``poetry install --only``).
            poetry_version: Poetry version specifier to ``pip install``, or None to skip installing Poetry.
            old_installer: If True, use Poetry's legacy installer.
            env: Environment variables set in the build container.
            secrets: Secrets injected as environment variables during the build.
            gpu: GPU type to attach to the builder container.

        Returns:
            A new `Image` with Poetry dependencies installed.
        """

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            context_files = {"/.pyproject.toml": os.path.expanduser(poetry_pyproject_toml)}

            commands = ["FROM base"]
            if poetry_version is not None:
                if poetry_version == "latest":
                    poetry_spec = "~=1.7" if version <= "2024.10" else ""
                else:
                    poetry_spec = f"=={poetry_version}"  # TODO: support other versions
                commands += [f"RUN python -m pip install poetry{poetry_spec}"]

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

            install_cmd += " --compile"  # Always compile .pyc during build; avoid recompiling on every cold start

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

    @_requires_image_instance
    def uv_sync(
        self,
        uv_project_dir: str = "./",
        *,
        force_build: bool = False,
        groups: list[str] | None = None,
        extras: list[str] | None = None,
        frozen: bool = True,
        extra_options: str = "",
        uv_version: str | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        gpu: str | None = None,
    ) -> "_Image":
        """Creates a virtual environment with the dependencies in a uv managed project with `uv sync`.

        The `pyproject.toml` and `uv.lock` in `uv_project_dir` are automatically added to the build context. The
        `uv_project_dir` is relative to the current working directory of where `modal` is called.

        NOTE: This does *not* install the project itself into the environment (this is equivalent to the
        `--no-install-project` flag in the `uv sync` command) and you would be expected to add any local python source
        files using `Image.add_local_python_source` or similar methods after this call.

        This ensures that updates to your project code wouldn't require reinstalling third-party dependencies
        after every change.

        uv workspaces are currently not supported.

        Added in v1.1.0.

        Args:
            uv_project_dir: Path to the local uv project directory (contains ``pyproject.toml``).
            force_build: If True, skip cached image builds.
            groups: Dependency groups passed as ``uv sync --group``.
            extras: Optional extras passed as ``uv sync --extra``.
            frozen: If True and a ``uv.lock`` exists, run ``uv sync --frozen`` so the lock is not updated at build time.
            extra_options: Additional raw options appended to ``uv sync``.
            uv_version: Pin the uv binary version copied from ``ghcr.io/astral-sh/uv``.
            env: Environment variables set in the build container.
            secrets: Secrets injected as environment variables during the build.
            gpu: GPU type to attach to the builder container.

        Returns:
            A new `Image` with a uv-managed virtual environment.

        Examples:
            ```python
            image = modal.Image.debian_slim().uv_sync()
            ```
        """

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

        def _normalize_items(items, name) -> list[str]:
            if items is None:
                return []
            elif isinstance(items, list):
                return items
            else:
                raise InvalidError(f"{name} must be None or a list of strings")

        groups = _normalize_items(groups, "groups")
        extras = _normalize_items(extras, "extras")

        def _check_pyproject_toml(pyproject_toml: str, version: ImageBuilderVersion):
            if not os.path.exists(pyproject_toml):
                raise InvalidError(f"Expected {pyproject_toml} to exist")

            import toml

            with open(pyproject_toml) as f:
                pyproject_toml_content = toml.load(f)

            if (
                "tool" in pyproject_toml_content
                and "uv" in pyproject_toml_content["tool"]
                and "workspace" in pyproject_toml_content["tool"]["uv"]
            ):
                raise InvalidError("uv workspaces are not supported")

            if version > "2024.10":
                # For builder version > 2024.10, modal is mounted at runtime and is not
                # a requirement in `uv.lock`
                return

            try:
                dependencies = pyproject_toml_content["project"]["dependencies"]
            except KeyError as e:
                raise InvalidError(
                    f"Invalid pyproject.toml file: missing key {e} in {pyproject_toml}. "
                    "See https://packaging.python.org/en/latest/guides/writing-pyproject-toml for guidelines."
                )

            for group in groups:
                if (
                    "dependency-groups" in pyproject_toml_content
                    and group in pyproject_toml_content["dependency-groups"]
                ):
                    dependencies += pyproject_toml_content["dependency-groups"][group]

            for extra in extras:
                if (
                    "project" in pyproject_toml_content
                    and "optional-dependencies" in pyproject_toml_content["project"]
                    and extra in pyproject_toml_content["project"]["optional-dependencies"]
                ):
                    dependencies += pyproject_toml_content["project"]["optional-dependencies"][extra]

            PACKAGE_REGEX = re.compile(r"^[\w-]+")

            def _extract_package(package) -> str:
                m = PACKAGE_REGEX.match(package)
                return m.group(0) if m else ""

            if not any(_extract_package(dependency) == "modal" for dependency in dependencies):
                raise InvalidError(
                    "Image builder version <= 2024.10 requires modal to be specified in your pyproject.toml file"
                )

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            uv_project_dir_ = os.path.expanduser(uv_project_dir)
            pyproject_toml = os.path.join(uv_project_dir_, "pyproject.toml")

            UV_ROOT = "/.uv"
            uv_sync_args = [
                f"--project={UV_ROOT}",
                "--no-install-workspace",  # Do not install the root project or any "uv workspace"
                "--compile-bytecode",
            ]

            for group in groups:
                uv_sync_args.append(f"--group={group}")
            for extra in extras:
                uv_sync_args.append(f"--extra={extra}")
            if extra_options:
                uv_sync_args.append(extra_options)

            commands = ["FROM base"]

            if uv_version is None:
                commands.append(f"COPY --from=ghcr.io/astral-sh/uv:latest /uv {UV_ROOT}/uv")
            else:
                commands.append(f"COPY --from=ghcr.io/astral-sh/uv:{uv_version} /uv {UV_ROOT}/uv")

            context_files = {}

            _check_pyproject_toml(pyproject_toml, version)

            context_files["/.pyproject.toml"] = pyproject_toml
            commands.append(f"COPY /.pyproject.toml {UV_ROOT}/pyproject.toml")

            uv_lock = os.path.join(uv_project_dir_, "uv.lock")
            if os.path.exists(uv_lock):
                context_files["/.uv.lock"] = uv_lock
                commands.append(f"COPY /.uv.lock {UV_ROOT}/uv.lock")

                if frozen:
                    # Do not update `uv.lock` when we have one when `frozen=True`. This is the default because this
                    # ensures that the runtime environment matches the local `uv.lock`.
                    #
                    # If `frozen=False`, then `uv sync` will update the the dependencies in the `uv.lock` file
                    # during build time.
                    uv_sync_args.append("--frozen")

            uv_sync_args_joined = " ".join(uv_sync_args).strip()

            commands += [
                f"RUN {UV_ROOT}/uv sync {uv_sync_args_joined}",
                f"ENV PATH={UV_ROOT}/.venv/bin:$PATH",
            ]

            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            force_build=self.force_build or force_build,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu),
        )

    @_requires_image_instance
    def dockerfile_commands(
        self,
        *dockerfile_commands: str | list[str],
        context_files: dict[str, str] = {},
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        gpu: str | None = None,
        context_dir: Path | str | None = None,
        force_build: bool = False,
        ignore: Sequence[str] | Callable[[Path], bool] = AUTO_DOCKERIGNORE,
        build_args: dict[str, str] = {},
    ) -> "_Image":
        """Extend an image with arbitrary Dockerfile-like commands.

        Args:
            *dockerfile_commands: Dockerfile lines to append after ``FROM base`` (strings or nested lists).
            context_files: Map of container paths to local files to include in the build context.
            env: Environment variables set in the build container.
            secrets: Secrets injected as environment variables during the build.
            gpu: GPU type to attach to the builder container.
            context_dir: Root directory for resolving relative COPY paths in implicit context mounts.
            force_build: If True, skip cached image builds.
            ignore: Ignore rules for the implicit context mount (defaults to auto ``.dockerignore`` behavior).
            build_args: Dockerfile ``ARG`` values forwarded to the build.

        Returns:
            A new `Image` with the Dockerfile fragment applied.

        Examples:
            ```python
            from modal import FilePatternMatcher

            # By default a .dockerignore file is used if present in the current working directory
            image = modal.Image.debian_slim().dockerfile_commands(
                ["COPY data /data"],
            )

            image = modal.Image.debian_slim().dockerfile_commands(
                ["COPY data /data"],
                ignore=["*.venv"],
            )

            image = modal.Image.debian_slim().dockerfile_commands(
                ["COPY data /data"],
                ignore=lambda p: p.is_relative_to(".venv"),
            )

            image = modal.Image.debian_slim().dockerfile_commands(
                ["COPY data /data"],
                ignore=FilePatternMatcher("**/*.txt"),
            )

            # When including files is simpler than excluding them, you can use the `~` operator to invert the matcher.
            image = modal.Image.debian_slim().dockerfile_commands(
                ["COPY data /data"],
                ignore=~FilePatternMatcher("**/*.py"),
            )

            # You can also read ignore patterns from a file.
            image = modal.Image.debian_slim().dockerfile_commands(
                ["COPY data /data"],
                ignore=FilePatternMatcher.from_file("/path/to/dockerignore"),
            )
            ```
        """
        cmds = _flatten_str_args("dockerfile_commands", "dockerfile_commands", dockerfile_commands)
        if not cmds:
            return self

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            return DockerfileSpec(commands=["FROM base", *cmds], context_files=context_files)

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            secrets=secrets,
            gpu_config=parse_gpu_config(gpu),
            context_mount_function=_create_context_mount_function(
                ignore=ignore, dockerfile_cmds=cmds, context_dir=context_dir
            ),
            force_build=self.force_build or force_build,
            build_args=build_args,
        )

    @_requires_image_instance
    def entrypoint(
        self,
        entrypoint_commands: list[str],
    ) -> "_Image":
        """Set the ENTRYPOINT for the image.

        Args:
            entrypoint_commands: argv tokens for the ``ENTRYPOINT`` JSON array form.

        Returns:
            A new `Image` with the entrypoint Dockerfile directive applied.
        """
        if not isinstance(entrypoint_commands, list) or not all(isinstance(x, str) for x in entrypoint_commands):
            raise InvalidError("entrypoint_commands must be a list of strings.")
        args = _flatten_str_args("entrypoint", "entrypoint_commands", entrypoint_commands)
        args_str = '"' + '", "'.join(args) + '"' if args else ""
        dockerfile_cmd = f"ENTRYPOINT [{args_str}]"

        return self.dockerfile_commands(dockerfile_cmd)

    @_requires_image_instance
    def shell(
        self,
        shell_commands: list[str],
    ) -> "_Image":
        """Overwrite default shell for the image.

        Args:
            shell_commands: argv tokens for the ``SHELL`` JSON array form.

        Returns:
            A new `Image` with the shell Dockerfile directive applied.
        """
        if not isinstance(shell_commands, list) or not all(isinstance(x, str) for x in shell_commands):
            raise InvalidError("shell_commands must be a list of strings.")
        args = _flatten_str_args("shell", "shell_commands", shell_commands)
        args_str = '"' + '", "'.join(args) + '"' if args else ""
        dockerfile_cmd = f"SHELL [{args_str}]"

        return self.dockerfile_commands(dockerfile_cmd)

    @_requires_image_instance
    def run_commands(
        self,
        *commands: str | list[str],
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        volumes: dict[str | PurePosixPath, _Volume] | None = None,
        gpu: str | None = None,
        force_build: bool = False,
    ) -> "_Image":
        """Extend an image with a list of shell commands to run.

        Args:
            *commands: Shell commands to run as separate ``RUN`` lines (strings or nested lists).
            env: Environment variables set in the build container.
            secrets: Secrets injected as environment variables during the build.
            volumes: Modal volumes to attach during the build step.
            gpu: GPU type to attach to the builder container.
            force_build: If True, skip cached image builds.

        Returns:
            A new `Image` with the commands executed as layers.
        """

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

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
            validated_volumes=validate_only_modal_volumes(volumes, "Image.run_commands"),
        )

    @staticmethod
    def micromamba(
        python_version: str | None = None,
        force_build: bool = False,
    ) -> "_Image":
        """A Micromamba base image. Micromamba allows for fast building of small Conda-based containers.

        Args:
            python_version: Python series or full version to install in the base conda environment.
            force_build: If True, skip cached image builds.

        Returns:
            A Micromamba-based `Image`.
        """

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            validated_python_version = _validate_python_version(
                python_version, version, allow_free_threading=False, caller_name="Image.micromamba"
            )
            micromamba_version = _base_image_config("micromamba", version)
            tag = f"mambaorg/micromamba:{micromamba_version}"
            setup_commands = [
                'SHELL ["/usr/local/bin/_dockerfile_shell.sh"]',
                "ENV MAMBA_DOCKERFILE_ACTIVATE=1",
                f"RUN micromamba install -n base -y python={validated_python_version} pip -c conda-forge",
            ]
            commands = _Image._registry_setup_commands(tag, version, setup_commands)
            if version > "2024.10":
                # for convenience when launching in a sandbox: sleep for 48h
                commands.append(f'CMD ["sleep", "{48 * 3600}"]')
            context_files = {}
            if version <= "2024.10":
                context_files = {CONTAINER_REQUIREMENTS_PATH: _get_modal_requirements_path(version, python_version)}
            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            dockerfile_function=build_dockerfile,
            force_build=force_build,
            _namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        )

    @_requires_image_instance
    def micromamba_install(
        self,
        *packages: str | list[str],
        spec_file: str | None = None,
        channels: list[str] = [],
        force_build: bool = False,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        gpu: str | None = None,
    ) -> "_Image":
        """Install a list of additional packages using micromamba.

        Args:
            *packages: Conda packages to install, e.g. ``numpy`` or version constraints.
            spec_file: Optional local path to a conda spec file to pass with ``-f``.
            channels: Conda channels to pass with repeated ``-c`` flags.
            force_build: If True, skip cached image builds.
            env: Environment variables set in the build container.
            secrets: Secrets injected as environment variables during the build.
            gpu: GPU type to attach to the builder container.

        Returns:
            A new `Image` with micromamba packages installed.
        """

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

        pkgs = _flatten_str_args("micromamba_install", "packages", packages)
        if not pkgs and spec_file is None:
            return self

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            package_args = shlex.join(pkgs)
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
        setup_commands: list[str],
        add_python: str | None = None,
    ) -> list[str]:
        add_python_commands: list[str] = []
        if add_python:
            _validate_python_version(
                add_python, builder_version, allow_micro_granularity=False, allow_free_threading=True
            )
            add_python_commands = [
                "COPY /python/. /usr/local",
                "ENV TERMINFO_DIRS=/etc/terminfo:/lib/terminfo:/usr/share/terminfo:/usr/lib/terminfo",
            ]
            python_minor = add_python.split(".")[1]
            if python_minor.endswith("t"):
                python_minor = python_minor[:-1]
            if int(python_minor) < 13:
                # Previous versions did not include the `python` binary, but later ones do.
                # (The important factor is not the Python version itself, but the standalone dist version.)
                # We insert the command in the list at the position it was previously always added
                # for backwards compatibility with existing images.
                add_python_commands.insert(1, "RUN ln -s /usr/local/bin/python3 /usr/local/bin/python")

        # Note: this change is because we install dependencies with uv in 2024.10+
        requirements_prefix = "python -m " if builder_version < "2024.10" else ""
        modal_requirements_commands = []
        if builder_version <= "2024.10":
            # past 2024.10, client dependencies are mounted at runtime
            modal_requirements_commands.extend(
                [
                    f"COPY {CONTAINER_REQUIREMENTS_PATH} {CONTAINER_REQUIREMENTS_PATH}",
                    f"RUN python -m pip install --upgrade {_base_image_config('package_tools', builder_version)}",
                    f"RUN {requirements_prefix}{_get_modal_requirements_command(builder_version)}",
                ]
            )
        if "2024.10" >= builder_version > "2023.12":
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
        secret: _Secret | None = None,
        *,
        setup_dockerfile_commands: list[str] = [],
        force_build: bool = False,
        add_python: str | None = None,
        **kwargs,
    ) -> "_Image":
        """Build a Modal Image from a public or private image registry, such as Docker Hub.

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

        Args:
            tag: Registry image reference (e.g. ``python:3.11-slim``).
            secret: Optional secret for static registry credentials.
            setup_dockerfile_commands: Extra Dockerfile lines run after ``FROM`` during base setup.
            force_build: If True, skip cached image builds.
            add_python: Optional standalone Python series to inject when the base image lacks Python.
            **kwargs: Additional arguments forwarded to the internal image constructor (e.g. registry config).

        Returns:
            An `Image` based on the registry tag.

        Examples:
            ```python
            modal.Image.from_registry("python:3.11-slim-bookworm")
            modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
            modal.Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
            ```
        """

        def context_mount_function() -> _Mount | None:
            return (
                _Mount.from_name(
                    python_standalone_mount_name(add_python),
                    namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
                )
                if add_python
                else None
            )

        if "image_registry_config" not in kwargs and secret is not None:
            kwargs["image_registry_config"] = _ImageRegistryConfig(api_pb2.REGISTRY_AUTH_TYPE_STATIC_CREDS, secret)

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            commands = _Image._registry_setup_commands(tag, version, setup_dockerfile_commands, add_python)
            context_files = {}
            if version <= "2024.10":
                context_files = {CONTAINER_REQUIREMENTS_PATH: _get_modal_requirements_path(version, add_python)}
            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            dockerfile_function=build_dockerfile,
            context_mount_function=context_mount_function,
            force_build=force_build,
            **kwargs,
        )

    @staticmethod
    def from_gcp_artifact_registry(
        tag: str,
        secret: _Secret | None = None,
        *,
        setup_dockerfile_commands: list[str] = [],
        force_build: bool = False,
        add_python: str | None = None,
        **kwargs,
    ) -> "_Image":
        """Build a Modal image from a private image in Google Cloud Platform (GCP) Artifact Registry.

        You will need to pass a `modal.Secret` containing [your GCP service account key data](https://cloud.google.com/iam/docs/keys-create-delete#creating)
        as `SERVICE_ACCOUNT_JSON`. This can be done from the [Secrets](https://modal.com/secrets) page.
        Your service account should be granted a specific role depending on the GCP registry used:

        - For Artifact Registry images (`pkg.dev` domains) use
          the ["Artifact Registry Reader"](https://cloud.google.com/artifact-registry/docs/access-control#roles) role
        - For Container Registry images (`gcr.io` domains) use
          the ["Storage Object Viewer"](https://cloud.google.com/artifact-registry/docs/transition/setup-gcr-repo) role

        **Note:** This method does not use `GOOGLE_APPLICATION_CREDENTIALS` as that
        variable accepts a path to a JSON file, not the actual JSON string.

        See `Image.from_registry()` for information about the other parameters.

        Args:
            tag: Full GCP Artifact Registry image reference.
            secret: Secret containing ``SERVICE_ACCOUNT_JSON`` for registry authentication.
            setup_dockerfile_commands: Extra Dockerfile lines run after ``FROM`` during base setup.
            force_build: If True, skip cached image builds.
            add_python: Optional standalone Python series to inject when the base image lacks Python.
            **kwargs: Additional arguments forwarded to `from_registry`.

        Returns:
            An `Image` based on the private GCP artifact.

        Examples:
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
        secret: _Secret | None = None,
        *,
        setup_dockerfile_commands: list[str] = [],
        force_build: bool = False,
        add_python: str | None = None,
        **kwargs,
    ) -> "_Image":
        """Build a Modal image from a private image in AWS Elastic Container Registry (ECR).

        You will need to pass a `modal.Secret` containing either IAM user credentials or OIDC
        configuration to access the target ECR registry.

        For IAM user authentication, set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION`.

        For OIDC authentication, set `AWS_ROLE_ARN` and `AWS_REGION`.

        IAM configuration details can be found in the AWS documentation for
        ["Private repository policies"](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-policies.html).

        For more details on using an AWS role to access ECR, see the [OIDC integration guide](https://modal.com/docs/guide/oidc-integration).

        See `Image.from_registry()` for information about the other parameters.

        Args:
            tag: Full ECR image URI.
            secret: Secret with IAM or OIDC credentials for ECR.
            setup_dockerfile_commands: Extra Dockerfile lines run after ``FROM`` during base setup.
            force_build: If True, skip cached image builds.
            add_python: Optional standalone Python series to inject when the base image lacks Python.
            **kwargs: Additional arguments forwarded to `from_registry`.

        Returns:
            An `Image` based on the private ECR image.

        Examples:
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
        path: str | Path,
        *,
        force_build: bool = False,
        context_dir: Path | str | None = None,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        gpu: str | None = None,
        add_python: str | None = None,
        build_args: dict[str, str] = {},
        ignore: Sequence[str] | Callable[[Path], bool] = AUTO_DOCKERIGNORE,
    ) -> "_Image":
        """Build a Modal image from a local Dockerfile.

        If your Dockerfile does not have Python installed, you can use the `add_python` parameter
        to specify a version of Python to add to the image.

        Args:
            path: Path to the Dockerfile on the local machine.
            force_build: If True, skip cached image builds.
            context_dir: Build context directory for resolving relative COPY paths.
            env: Environment variables set in the build container.
            secrets: Secrets injected as environment variables during the build.
            gpu: GPU type to attach to the builder container.
            add_python: Standalone Python version to add when the Dockerfile does not install Python.
            build_args: Dockerfile ``ARG`` values forwarded to the build.
            ignore: Ignore rules for the implicit context mount (defaults to auto ``.dockerignore`` behavior).

        Returns:
            An `Image` built from the Dockerfile plus Modal runtime dependencies.

        Examples:
            ```python
            from modal import FilePatternMatcher

            # By default a .dockerignore file is used if present in the current working directory
            image = modal.Image.from_dockerfile(
                "./Dockerfile",
                add_python="3.12",
            )

            image = modal.Image.from_dockerfile(
                "./Dockerfile",
                add_python="3.12",
                ignore=["*.venv"],
            )

            image = modal.Image.from_dockerfile(
                "./Dockerfile",
                add_python="3.12",
                ignore=lambda p: p.is_relative_to(".venv"),
            )

            image = modal.Image.from_dockerfile(
                "./Dockerfile",
                add_python="3.12",
                ignore=FilePatternMatcher("**/*.txt"),
            )

            # When including files is simpler than excluding them, you can use the `~` operator to invert the matcher.
            image = modal.Image.from_dockerfile(
                "./Dockerfile",
                add_python="3.12",
                ignore=~FilePatternMatcher("**/*.py"),
            )

            # You can also read ignore patterns from a file.
            image = modal.Image.from_dockerfile(
                "./Dockerfile",
                add_python="3.12",
                ignore=FilePatternMatcher.from_file("/path/to/dockerignore"),
            )
            ```
        """

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

        # --- Build the base dockerfile

        def build_dockerfile_base(version: ImageBuilderVersion) -> DockerfileSpec:
            with open(os.path.expanduser(path)) as f:
                commands = f.read().split("\n")
            return DockerfileSpec(commands=commands, context_files={})

        gpu_config = parse_gpu_config(gpu)
        base_image = _Image._from_args(
            dockerfile_function=build_dockerfile_base,
            context_mount_function=_create_context_mount_function(
                ignore=ignore, dockerfile_path=Path(path), context_dir=context_dir
            ),
            gpu_config=gpu_config,
            secrets=secrets,
            build_args=build_args,
        )

        # --- Now add in the modal dependencies, and, optionally a Python distribution
        # This happening in two steps is probably a vestigial consequence of previous limitations,
        # but it will be difficult to merge them without forcing rebuilds of images.

        def add_python_mount():
            return (
                _Mount.from_name(
                    python_standalone_mount_name(add_python),
                    namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
                )
                if add_python
                else None
            )

        def build_dockerfile_python(version: ImageBuilderVersion) -> DockerfileSpec:
            commands = _Image._registry_setup_commands("base", version, [], add_python)
            context_files = {}
            if version <= "2024.10":
                requirements_path = _get_modal_requirements_path(version, add_python)
                context_files = {CONTAINER_REQUIREMENTS_PATH: requirements_path}
            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            base_images={"base": base_image},
            dockerfile_function=build_dockerfile_python,
            context_mount_function=add_python_mount,
            force_build=force_build,
        )

    @staticmethod
    def from_scratch(force_build: bool = False) -> "_Image":
        """Create an empty Image, equivalent to `FROM scratch` in Docker.

        The resulting Image has no operating system, shell, or package manager. It is
        primarily useful as a lightweight filesystem to mount into a Sandbox via
        `Sandbox.mount_image`.

        Note that since this Image doesn't contain Python or other standard OS utilities,
        higher-level Image build steps like `pip_install` cannot be chained onto it. It also
        cannot be used for `modal.Function` execution, which requires a Python interpreter.

        Args:
            force_build: If True, skip cached image builds.

        Returns:
            An empty `Image` suitable for minimal filesystem mounts.

        Examples:
            ```python notest
            image = modal.Image.from_scratch().add_local_file(local_path, "/bin/my_binary", copy=True)
            ```
        """

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            return DockerfileSpec(commands=["FROM scratch"], context_files={})

        image = _Image._from_args(
            dockerfile_function=build_dockerfile,
            force_build=force_build,
            _namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        )
        image._is_empty = True
        return image

    @staticmethod
    def debian_slim(python_version: str | None = None, force_build: bool = False) -> "_Image":
        """Default image, based on the official `python` Docker images.

        Args:
            python_version: Python series or full version to use from the Debian slim images.
            force_build: If True, skip cached image builds.

        Returns:
            The standard Debian slim Python `Image` used as Modal's default base.
        """
        if isinstance(python_version, float):
            raise TypeError("The `python_version` argument should be a string, not a float.")

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            context_files = {}
            if version <= "2024.10":
                requirements_path = _get_modal_requirements_path(version, python_version)
                context_files = {CONTAINER_REQUIREMENTS_PATH: requirements_path}
            full_python_version = _dockerhub_python_version(
                version, python_version, allow_free_threading=False, caller_name="Image.debian_slim"
            )
            debian_codename = _base_image_config("debian", version)

            commands = [
                f"FROM python:{full_python_version}-slim-{debian_codename}",
            ]
            if version <= "2024.10":
                commands.extend(
                    [
                        f"COPY {CONTAINER_REQUIREMENTS_PATH} {CONTAINER_REQUIREMENTS_PATH}",
                    ]
                )
            commands.extend(
                [
                    "RUN apt-get update",
                    "RUN apt-get install -y gcc gfortran build-essential",
                    f"RUN pip install --upgrade {_base_image_config('package_tools', version)}",
                ]
            )
            if version <= "2024.10":
                # after 2024.10, modal requirements are mounted at runtime
                commands.extend(
                    [
                        f"RUN {_get_modal_requirements_command(version)}",
                    ]
                )
            commands.extend(
                [
                    # Set debian front-end to non-interactive to avoid users getting stuck with input prompts.
                    "RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections",
                ]
            )
            if "2024.10" >= version > "2023.12":
                commands.append(f"RUN rm {CONTAINER_REQUIREMENTS_PATH}")
            if version > "2024.10":
                # for convenience when launching in a sandbox: sleep for 48h
                commands.append(f'CMD ["sleep", "{48 * 3600}"]')
            return DockerfileSpec(commands=commands, context_files=context_files)

        return _Image._from_args(
            dockerfile_function=build_dockerfile,
            force_build=force_build,
            _namespace=api_pb2.DEPLOYMENT_NAMESPACE_GLOBAL,
        )

    @_requires_image_instance
    def apt_install(
        self,
        *packages: str | list[str],
        force_build: bool = False,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        gpu: str | None = None,
    ) -> "_Image":
        """Install a list of Debian packages using `apt`.

        Args:
            *packages: Apt package names to install, e.g. ``git`` or ``libpq-dev``.
            force_build: If True, skip cached image builds.
            env: Environment variables set in the build container.
            secrets: Secrets injected as environment variables during the build.
            gpu: GPU type to attach to the builder container.

        Returns:
            A new `Image` with ``apt-get install`` layers applied.

        Examples:
            ```python
            image = modal.Image.debian_slim().apt_install("git")
            ```
        """
        pkgs = _flatten_str_args("apt_install", "packages", packages)
        if not pkgs:
            return self

        package_args = shlex.join(pkgs)

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            commands = [
                "FROM base",
                "RUN apt-get update",
                f"RUN apt-get install -y {package_args}",
            ]
            return DockerfileSpec(commands=commands, context_files={})

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
            force_build=self.force_build or force_build,
            gpu_config=parse_gpu_config(gpu),
            secrets=secrets,
        )

    @_requires_image_instance
    def run_function(
        self,
        raw_f: Callable[..., Any],
        *,
        env: dict[str, str | None] | None = None,
        secrets: Collection[_Secret] | None = None,
        volumes: dict[str | PurePosixPath, _Volume | _CloudBucketMount] = {},
        network_file_systems: dict[str | PurePosixPath, _NetworkFileSystem] = {},
        gpu: str | list[str] | None = None,
        cpu: float | None = None,
        memory: int | None = None,
        timeout: int = 60 * 60,
        cloud: str | None = None,
        region: str | Sequence[str] | None = None,
        force_build: bool = False,
        args: Sequence[Any] = (),
        kwargs: dict[str, Any] = {},
        include_source: bool = True,
    ) -> "_Image":
        """Run user-defined function `raw_f` as an image build step.

        The function runs like an ordinary Modal Function, accepting a resource configuration and integrating
        with Modal features like Secrets and Volumes. Unlike ordinary Modal Functions, any changes to the
        filesystem state will be captured on container exit and saved as a new Image.

        Only the source code of `raw_f`, the contents of `**kwargs`, and any referenced *global* variables
        are used to determine whether the image has changed and needs to be rebuilt.
        If this function references other functions or variables, the image will not be rebuilt if you
        make changes to them. You can force a rebuild by changing the function's source code itself.

        Args:
            raw_f: Callable executed remotely during the image build.
            env: Environment variables set in the builder container.
            secrets: Secrets available to the builder function.
            volumes: Volume and bucket mounts attached for the build.
            network_file_systems: Network file systems attached for the build.
            gpu: GPU type or list of types for the builder container.
            cpu: CPU cores to request (soft limit).
            memory: Memory to request in MiB (soft limit).
            timeout: Maximum build-step runtime in seconds.
            cloud: Cloud provider for the builder function.
            region: Region or regions for the builder function.
            force_build: If True, skip cached image builds.
            args: Positional arguments serialized to the builder function.
            kwargs: Keyword arguments serialized to the builder function.
            include_source: Whether to include the function's source in the builder image.

        Returns:
            A new `Image` capturing the filesystem after `raw_f` completes.

        Examples:
            ```python notest

            def my_build_function():
                open("model.pt", "w").write("parameters!")

            image = (
                modal.Image
                    .debian_slim()
                    .pip_install("torch")
                    .run_function(my_build_function, secrets=[...], volumes={...})
            )
            ```
        """

        secrets = secrets or []
        if env:
            secrets = [*secrets, _Secret.from_dict(env)]

        from ._functions import _Function

        if not callable(raw_f):
            raise InvalidError(f"Argument to Image.run_function must be a function, not {type(raw_f).__name__}.")
        elif raw_f.__name__ == "<lambda>":
            # It may be possible to support lambdas eventually, but for now we don't handle them well, so reject quickly
            raise InvalidError("Image.run_function does not support lambda functions.")

        info = FunctionInfo(raw_f)

        function = _Function.from_local(
            info,
            app=None,
            image=self,
            secrets=secrets,
            gpu=gpu,
            volumes=volumes,
            network_file_systems=network_file_systems,
            cloud=cloud,
            region=region,
            memory=memory,
            timeout=timeout,
            cpu=cpu,
            is_builder_function=True,
            include_source=include_source,
        )
        if len(args) + len(kwargs) > 0:
            data_format = get_preferred_payload_format()
            args_serialized = serialize_data_format((args, kwargs), data_format)

            if len(args_serialized) > MAX_OBJECT_SIZE_BYTES:
                raise InvalidError(
                    f"Arguments to `run_function` are too large ({len(args_serialized)} bytes). "
                    f"Maximum size is {MAX_OBJECT_SIZE_BYTES} bytes."
                )

            build_function_input = api_pb2.FunctionInput(
                args=args_serialized,
                data_format=data_format,
            )
        else:
            build_function_input = None
        return _Image._from_args(
            base_images={"base": self},
            build_function=function,
            build_function_input=build_function_input,
            force_build=self.force_build or force_build,
        )

    @_requires_image_instance
    def env(self, vars: dict[str, str]) -> "_Image":
        """Sets the environment variables in an Image.

        Args:
            vars: Map of environment variable names to string values.

        Returns:
            A new `Image` with ``ENV`` directives applied.

        Examples:
            ```python
            image = (
                modal.Image.debian_slim()
                .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
            )
            ```
        """
        non_str_keys = [key for key, val in vars.items() if not isinstance(val, str)]
        if non_str_keys:
            raise InvalidError(f"Image ENV variables must be strings. Invalid keys: {non_str_keys}")

        def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
            env_commands = [f"ENV {key}={shlex.quote(val)}" for (key, val) in vars.items()]
            return DockerfileSpec(commands=["FROM base"] + env_commands, context_files={})

        return _Image._from_args(
            base_images={"base": self},
            dockerfile_function=build_dockerfile,
        )

    @_requires_image_instance
    def workdir(self, path: str | PurePosixPath) -> "_Image":
        """Set the working directory for subsequent image build steps and function execution.

        Args:
            path: Working directory path inside the image.

        Returns:
            A new `Image` with ``WORKDIR`` applied.

        Examples:
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

    @_requires_image_instance
    def cmd(self, cmd: list[str]) -> "_Image":
        """Set the default command (`CMD`) to run when a container is started.

        Used with `modal.Sandbox`. Has no effect on `modal.Function`.

        Args:
            cmd: argv tokens for the default container command.

        Returns:
            A new `Image` with ``CMD`` applied.

        Examples:
            ```python
            image = (
                modal.Image.debian_slim().cmd(["python", "app.py"])
            )
            ```
        """

        if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
            raise InvalidError("Image CMD must be a list of strings.")

        cmd_args = _flatten_str_args("cmd", "cmd", cmd)
        cmd_str = '"' + '", "'.join(cmd_args) + '"' if cmd_args else ""
        dockerfile_cmd = f"CMD [{cmd_str}]"

        return self.dockerfile_commands(dockerfile_cmd)

    @_requires_image_instance
    def pipe(
        self,
        func: Callable[Concatenate["modal.Image", P], "modal.Image"],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> "_Image":
        """Apply a local function to expand the Image recipe.

        This method can be useful for defining reusable Image build
        recipes that compose well with the fluent Image builder interface.

        **Example**

        ```python
        def workspace_setup(image: modal.Image, repo: str) -> modal.Image:
            return image.run_commands(f"git clone {repo}").uv_pip_install(".")

        image = (
            modal.Image.debian_slim()
            .apt_install("git")
            .pipe(workspace_setup, "https://github.com/example/repo.git")
        )
        ```
        """
        # Typing here is complicated because user callables accept the public
        # Image type, but `self` is an instance of the internal _Image type.
        self_ = synchronizer._translate_out(self)
        res = func(self_, *args, **kwargs)  # type: ignore[reportUnknownReturnType]
        return typing.cast(_Image, synchronizer._translate_in(res))

    # Live handle methods

    @contextlib.contextmanager
    def imports(self):
        """Used to import packages in global scope that are only available when running remotely.

        By using this context manager you can avoid an `ImportError` due to not having certain
        packages installed locally.

        Returns:
            Context manager that records import failures until the image is hydrated in the remote environment.

        Examples:
            ```python notest
            with image.imports():
                import torch
            ```
        """
        env_image_id = config.get("image_id")
        try:
            yield
        except Exception as exc:
            if not self.is_hydrated:
                # Might be hydrated later (if it's the container's used image)
                self.inside_exceptions.append(exc)
            elif env_image_id == self.object_id:
                # Image is already hydrated (we can remove this case later
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
            image_id=self.object_id, timeout=55, last_entry_id=last_entry_id, include_logs_for_finished=True
        )
        async for response in self.client.stub.ImageJoinStreaming.unary_stream(request):
            if response.result.status:
                return
            if response.entry_id:
                last_entry_id = response.entry_id
            for task_log in response.task_logs:
                if task_log.data:
                    yield task_log.data

    @staticmethod
    def from_name(
        name: str,
        *,
        environment_name: str | None = None,
        client: _Client | None = None,
    ) -> "_Image":
        """Reference a named Image that was previously published with `.publish()`.

        Names can contain an optional `:tag` part - if no tag part is included `":latest"` is used,
        matching Docker conventions.

        ```python notest
        image = modal.Image.from_name("my-image")     # references my-image:latest
        image_v1 = modal.Image.from_name("my-image:v1")

        @app.function(image=image)
        def run():
            ...
        ```
        """
        namespace_prefix, tag = _parse_named_image_ref(name)

        if namespace_prefix and environment_name:
            raise InvalidError("Cannot specify 'environment_name' when the image name contains a '/'.")

        async def _load(self: _Image, resolver: Resolver, load_context: LoadContext, existing_object_id: str | None):
            req = api_pb2.ImageGetByTagRequest(
                tag=tag,
                environment_name="" if namespace_prefix else load_context.environment_name,
            )
            response = await load_context.client.stub.ImageGetByTag(req)
            self._hydrate(response.image_id, load_context.client, None)

        rep = _Image._repr(tag, environment_name)
        return _Image._from_loader(
            _load,
            rep,
            hydrate_lazily=True,
            skip_reload=True,
            load_context_overrides=LoadContext(environment_name=environment_name, client=client),
        )

    async def publish(
        self,
        name: str,
        *,
        environment_name: str | None = None,
        client: _Client | None = None,
    ) -> None:
        """Publish this image under the given name

        The Image must already be created (typically by calling `image.build()` or `sandbox.snapshot_filesystem()`).

        Image names can contain an explicit tag designation (using the `name:tag`). If no tag is included in the name,
        `":latest"` is used, matching Docker conventions. To publish multiple tags, call `.publish()` once per tag.

        ```python notest
        image = modal.Image.debian_slim().pip_install("numpy")
        image.build(app)
        image.publish("my-image-with-numpy")     # my-image-with-numpy:latest
        image.publish("my-image-with-numpy:v1")
        ```
        """
        namespace_prefix, tag = _parse_named_image_ref(name)

        if namespace_prefix:
            if environment_name:
                raise InvalidError("Cannot specify 'environment_name' when the image name contains a '/'.")
            resolved_env = ""
        else:
            resolved_env = environment_name or config.get("environment") or ""

        if self._object_id is None:
            raise InvalidError("Cannot publish an image that has not been created yet. Call `.build()` first.")

        _client = client or await _Client.from_env()

        await _client.stub.ImagePublish(
            api_pb2.ImagePublishRequest(
                image_id=self._object_id,
                environment_name=resolved_env,
                allow_public=False,
                tag=tag,
            )
        )

    async def hydrate(self, client: _Client | None = None) -> Self:
        """mdmd:hidden"""
        # Image inherits hydrate() from Object but can't be hydrated on demand
        # Overriding the method lets us hide it from the docs and raise a better error message
        if not self.is_hydrated:
            raise ExecutionError(
                "Images cannot currently be hydrated on demand; you can build an Image by running an App that uses it."
            )
        return self
