# Copyright Modal Labs 2022
from typing import Optional

from ._container_io_manager import _ContainerIOManager


def stop_fetching_inputs():
    """Don't fetch any more inputs from the server, after the current one.
    The container will exit gracefully after the current input is processed."""

    _ContainerIOManager.stop_fetching_inputs()


def set_local_concurrent_inputs(concurrent_inputs: int) -> None:
    """Set the number of concurrent inputs for the local container."""

    _ContainerIOManager._singleton.set_concurrent_inputs(concurrent_inputs)


def get_local_concurrent_inputs() -> Optional[int]:
    """Get the number of concurrent inputs for the local container."""

    return _ContainerIOManager._singleton.get_concurrent_inputs()
