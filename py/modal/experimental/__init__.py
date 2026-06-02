# Copyright Modal Labs 2025
from dataclasses import dataclass
from datetime import datetime, timezone

from modal_proto import api_pb2

from .._clustered_functions import ClusterInfo, get_cluster_info as _get_cluster_info
from .._functions import _Function
from .._image import (
    DockerfileSpec as DockerfileSpec,
    ImageBuilderVersion as ImageBuilderVersion,
    _Image as _Image,
    _ImageRegistryConfig as _ImageRegistryConfig,
)
from .._object import _get_environment_name
from .._partial_function import _clustered
from .._runtime.container_io_manager import _ContainerIOManager
from .._utils.async_utils import synchronize_api, synchronizer
from ..app import _App
from ..client import _Client
from ..cls import _Cls
from ..exception import InvalidError as InvalidError
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
async def list_deployed_apps(environment_name: str = "", client: _Client | None = None) -> list[AppInfo]:
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


@dataclass
class AppLifecycle:
    """Lifecycle information about an App."""

    created_at: datetime  # Time the App was initially created
    created_by: str  # User or service user name
    deployed_at: datetime | None  # Time of the most recent deployment event, if ever deployed
    deployed_by: str | None  # User or service user name, if ever deployed
    stopped_at: datetime | None  # None when the App is still running
    stopped_by: str | None  # User or service user name; None if still running or finished normally


def _timestamp_to_datetime(ts: float) -> datetime | None:
    # Unset timestamps come back from the server as 0.
    return datetime.fromtimestamp(ts, timezone.utc) if ts else None


@synchronizer.create_blocking
async def get_app_lifecycle(app_id: str, *, client: _Client | None = None) -> AppLifecycle:
    """Get lifecycle information about an App.

    This interface is experimental. This information will continue to be available in the future,
    but it may be accessed via a different interface, and the return value may have a different shape.
    """
    client = client or await _Client.from_env()

    resp: api_pb2.AppGetLifecycleResponse = await client.stub.AppGetLifecycle(
        api_pb2.AppGetLifecycleRequest(app_id=app_id)
    )
    lifecycle = resp.lifecycle
    return AppLifecycle(
        created_at=datetime.fromtimestamp(lifecycle.created_at, timezone.utc),
        created_by=lifecycle.created_by,
        deployed_at=_timestamp_to_datetime(lifecycle.deployed_at),
        deployed_by=lifecycle.deployed_by or None,
        stopped_at=_timestamp_to_datetime(lifecycle.stopped_at),
        stopped_by=lifecycle.stopped_by or None,
    )


@synchronizer.create_blocking
async def stop_app(name: str, *, environment_name: str | None = None, client: _Client | None = None) -> None:
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
    app_name: str, *, environment_name: str | None = None, client: _Client | None = None
) -> dict[str, _Function | _Cls]:
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

    app_objects: dict[str, _Function | _Cls] = {}

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
    client: _Client | None = None,
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
