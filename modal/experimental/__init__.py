# Copyright Modal Labs 2025
from dataclasses import dataclass
from typing import Optional

from modal_proto import api_pb2

from .._clustered_functions import ClusterInfo, get_cluster_info as _get_cluster_info
from .._object import _get_environment_name
from .._partial_function import _clustered
from .._runtime.container_io_manager import _ContainerIOManager
from .._utils.async_utils import synchronize_api, synchronizer
from ..client import _Client


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
