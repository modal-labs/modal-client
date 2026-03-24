# Copyright Modal Labs 2025
from dataclasses import dataclass
from typing import Optional, Union

from modal_proto import api_pb2

from .._clustered_functions import ClusterInfo, get_cluster_info as _get_cluster_info
from .._functions import _Function
from .._object import _get_environment_name
from .._partial_function import _clustered
from .._runtime.container_io_manager import _ContainerIOManager
from .._utils.async_utils import synchronize_api, synchronizer
from ..app import _App
from ..client import _Client
from ..cls import _Cls
from ..exception import InvalidError as InvalidError
from ..image import (
    DockerfileSpec as DockerfileSpec,
    ImageBuilderVersion as ImageBuilderVersion,
    _Image as _Image,
    _ImageRegistryConfig as _ImageRegistryConfig,
)
from ..secret import _Secret as _Secret
from .flash import (  # noqa: F401
    flash_forward,
    flash_get_containers,
    flash_prometheus_autoscaler,
    http_server,
)


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
async def stop_app(name: str, *, environment_name: Optional[str] = None, client: Optional[_Client] = None) -> None:
    """Stop a deployed App.

    This interface is experimental and may change in the future,
    although the functionality will continue to be supported.
    """
    client_ = client or await _Client.from_env()
    app = await _App.lookup(name, environment_name=environment_name, client=client_)
    req = api_pb2.AppStopRequest(app_id=app.app_id, source=api_pb2.APP_STOP_SOURCE_PYTHON_CLIENT)
    await client_.stub.AppStop(req)


@synchronizer.create_blocking
async def get_app_objects(
    app_name: str, *, environment_name: Optional[str] = None, client: Optional[_Client] = None
) -> dict[str, Union[_Function, _Cls]]:
    """Experimental interface for retrieving a dictionary of the Functions / Clses in an App.

    The return value is a dictionary mapping names to unhydrated Function or Cls objects.

    We plan to support this functionality through a stable API in the future. It's likely that
    the stable API will look different (it will probably be a method on the App object itself).

    """
    # This is implemented through a somewhat odd mixture of internal RPCs and public APIs.
    # While AppGetLayout provides the object ID and metadata for each object in the App, it's
    # currently somewhere between very awkward and impossible to hydrate a modal.Cls with just
    # that information, since the "class service function" needs to be loaded first
    # (and it's not always possible to do that without knowledge of the parameterization).
    # So instead we just use AppGetLayout to retrieve the names of the Functions / Clsices on
    # the App and then use the public .from_name constructors to return unhydrated handles.

    # Additionally, since we need to know the environment name to use `.from_name`, and the App's
    # environment name isn't stored anywhere on the App (and cannot be retrieved via an RPC), the
    # experimental function is parameterized by an App name while the stable API would instead
    # be a method on the App itself.

    if client is None:
        client = await _Client.from_env()

    app = await _App.lookup(app_name, environment_name=environment_name, client=client)
    req = api_pb2.AppGetLayoutRequest(app_id=app.app_id)
    app_layout_resp = await client.stub.AppGetLayout(req)

    app_objects: dict[str, Union[_Function, _Cls]] = {}

    for cls_name in app_layout_resp.app_layout.class_ids:
        app_objects[cls_name] = _Cls.from_name(app_name, cls_name, environment_name=environment_name)

    for func_name in app_layout_resp.app_layout.function_ids:
        if func_name.endswith(".*"):
            continue  # Only skip class service functions since classes are already registered above
        app_objects[func_name] = _Function.from_name(app_name, func_name, environment_name=environment_name)

    return app_objects


@synchronizer.create_blocking
async def image_delete(
    image_id: str,
    *,
    client: Optional[_Client] = None,
) -> None:
    """Delete an Image by its ID.

    Deletion is irreversible and will prevent Functions/Sandboxes from using
    the Image.

    This is an experimental interface for a feature that we will be adding to
    the main Image class. The stable form of this interface may look different.

    Note: When building an Image, each chained method call will create an
    intermediate Image layer, each with its own ID. Deleting an Image will not
    delete any of its intermediate layers, only the image identified by the
    provided ID.
    """
    if client is None:
        client = await _Client.from_env()

    req = api_pb2.ImageDeleteRequest(image_id=image_id)
    await client.stub.ImageDelete(req)
