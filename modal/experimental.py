# Copyright Modal Labs 2022
from ._container_io_manager import _ContainerIOManager


def stop_fetching_inputs():
    """Don't fetch any more inputs from the server, after the current one.
    The container will exit gracefully after the current input is processed."""

    _ContainerIOManager.stop_fetching_inputs()


def get_local_input_concurrency():
    """Return the container's local input concurrency."""

    return _ContainerIOManager.get_input_concurrency()
