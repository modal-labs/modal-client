# Copyright Modal Labs 2022
from typing import (
    Any,
    Callable,
)

import modal._clustered_functions
from modal.functions import _Function

from ._container_io_manager import _ContainerIOManager
from .exception import (
    InvalidError,
)
from .partial_function import _PartialFunction, _PartialFunctionFlags


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


def get_cluster_info() -> modal._clustered_functions.ClusterInfo:
    return modal._clustered_functions.get_cluster_info()
