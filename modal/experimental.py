# Copyright Modal Labs 2022

from modal._utils.async_utils import synchronize_api
from modal.functions import FunctionStats

from ._container_io_manager import _ContainerIOManager


def stop_fetching_inputs():
    """Don't fetch any more inputs from the server, after the current one.
    The container will exit gracefully after the current input is processed."""

    _ContainerIOManager.stop_fetching_inputs()


def set_local_concurrent_inputs(concurrent_inputs: int) -> None:
    """Set the number of concurrent inputs for the local container."""

    return _ContainerIOManager.set_input_concurrency(concurrent_inputs)


def get_local_concurrent_inputs() -> int:
    """Get the number of concurrent inputs for the local container. Returns 0 if not set."""

    return _ContainerIOManager.get_input_concurrency()


async def _get_current_function_stats() -> FunctionStats:
    """Return a `FunctionStats` object describing the current function's queue and runner counts."""

    resp = await _ContainerIOManager.get_current_function_stats()
    return FunctionStats(
        backlog=resp.backlog,
        num_total_runners=resp.num_total_tasks,
    )


get_current_function_stats = synchronize_api(_get_current_function_stats)
