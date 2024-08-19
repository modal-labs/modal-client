# Copyright Modal Labs 2022
from typing import Optional

from modal._utils.async_utils import synchronize_api

from ._container_io_manager import _ContainerIOManager


def stop_fetching_inputs():
    """Don't fetch any more inputs from the server, after the current one.
    The container will exit gracefully after the current input is processed."""

    _ContainerIOManager.stop_fetching_inputs()


async def _set_local_concurrent_inputs(concurrent_inputs: int) -> None:
    """Set the number of concurrent inputs for the local container."""

    return await _ContainerIOManager._singleton.set_concurrent_inputs(concurrent_inputs)


async def _get_local_concurrent_inputs() -> Optional[int]:
    """Get the number of concurrent inputs for the local container."""

    return await _ContainerIOManager._singleton.get_concurrent_inputs()


set_local_concurrent_inputs = synchronize_api(_set_local_concurrent_inputs)
get_local_concurrent_inputs = synchronize_api(_get_local_concurrent_inputs)
