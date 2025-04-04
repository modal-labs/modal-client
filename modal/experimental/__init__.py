# Copyright Modal Labs 2025
from dataclasses import dataclass
from typing import Any, Callable, Optional

from modal_proto import api_pb2

from .._clustered_functions import ClusterInfo, get_cluster_info as _get_cluster_info
from .._functions import _Function
from .._object import _get_environment_name
from .._partial_function import _PartialFunction, _PartialFunctionFlags
from .._runtime.container_io_manager import _ContainerIOManager
from .._utils.async_utils import synchronizer
from ..client import _Client
from ..exception import InvalidError


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


def clustered(size: int, broadcast: bool = True):
    """Provision clusters of colocated and networked containers for the Function.

    Parameters:
    size: int
        Number of containers spun up to handle each input.
    broadcast: bool = True
        If True, inputs will be sent simultaneously to each container. Otherwise,
        inputs will be sent only to the rank-0 container, which is responsible for
        delegating to the workers.
    """

    assert broadcast, "broadcast=False has not been implemented yet!"

    if size <= 0:
        raise ValueError("cluster size must be greater than 0")

    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if isinstance(raw_f, _Function):
            raw_f = raw_f.get_raw_f()
            raise InvalidError(
                f"Applying decorators for {raw_f} in the wrong order!\nUsage:\n\n"
                "@app.function()\n@modal.clustered()\ndef clustered_function():\n    ..."
            )
        return _PartialFunction(
            raw_f, _PartialFunctionFlags.FUNCTION | _PartialFunctionFlags.CLUSTERED, cluster_size=size
        )

    return wrapper


def get_cluster_info() -> ClusterInfo:
    return _get_cluster_info()


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
