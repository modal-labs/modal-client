# Copyright Modal Labs 2025
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from modal_proto import api_pb2

from .._clustered_functions import ClusterInfo, get_cluster_info as _get_cluster_info
from .._functions import _Function
from .._object import _get_environment_name
from .._partial_function import _clustered
from .._runtime.container_io_manager import _ContainerIOManager
from .._utils.async_utils import synchronize_api, synchronizer
from .._utils.deprecation import deprecation_warning
from .._utils.grpc_utils import retry_transient_errors
from ..client import _Client
from ..cls import _Obj
from ..exception import InvalidError
from ..image import DockerfileSpec, ImageBuilderVersion, _Image, _ImageRegistryConfig
from ..secret import _Secret


def stop_fetching_inputs():
    """Don't fetch any more inputs from the server, after the current one.
    The container will exit gracefully after the current input is processed."""
    _ContainerIOManager.stop_fetching_inputs()


def get_local_input_concurrency():
    """Get the container's local input concurrency.
    If recently reduced to particular value, it can return a larger number than
    set due to in-progress inputs."""
    return _ContainerIOManager.get_input_concurrency()


def set_local_input_concurrency(concurrency: int):
    """Set the container's local input concurrency. Dynamic concurrency will be disabled.
    When setting to a smaller value, this method will not interrupt in-progress inputs.
    """
    _ContainerIOManager.set_input_concurrency(concurrency)


def get_cluster_info() -> ClusterInfo:
    return _get_cluster_info()


clustered = synchronize_api(_clustered, target_module=__name__)


@dataclass
class AppInfo:
    app_id: str
    name: str
    containers: int


@synchronizer.create_blocking
async def list_deployed_apps(environment_name: str = "", client: Optional[_Client] = None) -> list[AppInfo]:
    """List deployed Apps along with the number of containers currently running."""
    # This function exists to provide backwards compatibility for some users who had been
    # calling into the private function that previously backed the `modal app list` CLI command.
    # We plan to add more Python API for exposing this sort of information, but we haven't
    # settled on a design we're happy with yet. In the meantime, this function will continue
    # to support existing codebases. It's likely that the final API will be different
    # (e.g. more oriented around the App object). This function should be gracefully deprecated
    # one the new API is released.
    client = client or await _Client.from_env()

    resp: api_pb2.AppListResponse = await client.stub.AppList(
        api_pb2.AppListRequest(environment_name=_get_environment_name(environment_name))
    )

    app_infos = []
    for app_stats in resp.apps:
        if app_stats.state == api_pb2.APP_STATE_DEPLOYED:
            app_infos.append(
                AppInfo(
                    app_id=app_stats.app_id,
                    name=app_stats.description,
                    containers=app_stats.n_running_tasks,
                )
            )
    return app_infos


@synchronizer.create_blocking
async def raw_dockerfile_image(
    path: Union[str, Path],
    force_build: bool = False,
) -> _Image:
    """
    Build a Modal Image from a local Dockerfile recipe without any changes.

    Unlike for `modal.Image.from_dockerfile`, the provided recipe will not be embellished with
    steps to install dependencies for the Modal client package. As a consequence, the resulting
    Image cannot be used with a modal Function unless those dependencies are added in a subsequent
    layer. It _can_ be directly used with a modal Sandbox, which does not need the Modal client.

    We expect to support this experimental function until the `2025.04` Modal Image Builder is
    stable, at which point Modal Image recipes will no longer install the client dependencies
    by default. At that point, users can upgrade their Image Builder Version and migrate to
    `modal.Image.from_dockerfile` for usecases supported by this function.

    """

    def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
        with open(os.path.expanduser(path)) as f:
            commands = f.read().split("\n")
        return DockerfileSpec(commands=commands, context_files={})

    return _Image._from_args(
        dockerfile_function=build_dockerfile,
        force_build=force_build,
    )


@synchronizer.create_blocking
async def raw_registry_image(
    tag: str,
    registry_secret: Optional[_Secret] = None,
    credential_type: Literal["static", "aws", "gcp", None] = None,
    force_build: bool = False,
) -> _Image:
    """
    Build a Modal Image from a public or private image registry without any changes.

    Unlike for `modal.Image.from_registry`, the provided recipe will not be embellished with
    steps to install dependencies for the Modal client package. As a consequence, the resulting
    Image cannot be used with a modal Function unless those dependencies are added in a subsequent
    layer. It _can_ be directly used with a modal Sandbox, which does not need the Modal client.

    We expect to support this experimental function until the `2025.04` Modal Image Builder is
    stable, at which point Modal Image recipes will no longer install the client dependencies
    by default. At that point, users can upgrade their Image Builder Version and migrate to
    `modal.Image.from_registry` for usecases supported by this function.

    """

    def build_dockerfile(version: ImageBuilderVersion) -> DockerfileSpec:
        commands = [f"FROM {tag}"]
        return DockerfileSpec(commands=commands, context_files={})

    if registry_secret:
        if credential_type is None:
            raise InvalidError("credential_type must be provided when using a registry_secret")
        elif credential_type == "static":
            auth_type = api_pb2.REGISTRY_AUTH_TYPE_STATIC_CREDS
        elif credential_type == "aws":
            auth_type = api_pb2.REGISTRY_AUTH_TYPE_AWS
        elif credential_type == "gcp":
            auth_type = api_pb2.REGISTRY_AUTH_TYPE_GCP
        else:
            raise InvalidError(f"Invalid credential_type: {credential_type!r}")
        registry_config = _ImageRegistryConfig(auth_type, registry_secret)
    else:
        registry_config = None

    return _Image._from_args(
        dockerfile_function=build_dockerfile,
        image_registry_config=registry_config,
        force_build=force_build,
    )


@synchronizer.create_blocking
async def update_autoscaler(
    obj: Union[_Function, _Obj],
    *,
    min_containers: Optional[int] = None,
    max_containers: Optional[int] = None,
    buffer_containers: Optional[int] = None,
    scaledown_window: Optional[int] = None,
    client: Optional[_Client] = None,
) -> None:
    """Update the autoscaler settings for a Function or Obj (instance of a Cls).

    This is an experimental interface for a feature that we will be adding to
    replace the existing `.keep_warm()` method. The stable form of this interface
    may look different (i.e., it may be a standalone function or a method).

    """
    deprecation_warning(
        (2025, 5, 5),
        "The modal.experimental.update_autoscaler(...) function is now deprecated in favor of"
        " a stable `.update_autoscaler(...) method on the corresponding object.",
        show_source=True,
    )

    settings = api_pb2.AutoscalerSettings(
        min_containers=min_containers,
        max_containers=max_containers,
        buffer_containers=buffer_containers,
        scaledown_window=scaledown_window,
    )

    if client is None:
        client = await _Client.from_env()

    if isinstance(obj, _Function):
        f = obj
    else:
        assert obj._cls._class_service_function is not None
        await obj._cls._class_service_function.hydrate(client=client)
        f = obj._cached_service_function()
    await f.hydrate(client=client)

    request = api_pb2.FunctionUpdateSchedulingParamsRequest(function_id=f.object_id, settings=settings)
    await retry_transient_errors(client.stub.FunctionUpdateSchedulingParams, request)
