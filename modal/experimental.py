# Copyright Modal Labs 2022
from modal._utils.async_utils import synchronize_api
from modal._utils.grpc_utils import retry_transient_errors
from modal.execution_context import current_input_id
from modal_proto import api_pb2

from ._container_io_manager import _ContainerIOManager


def stop_fetching_inputs():
    """Don't fetch any more inputs from the server, after the current one.
    The container will exit gracefully after the current input is processed."""

    _ContainerIOManager.stop_fetching_inputs()


def get_local_input_concurrency():
    """Get the container's local input concurrency. Return 0 if the container is not running."""

    return _ContainerIOManager.get_input_concurrency()

async def _logger(message: str) -> None:
    """Log a message to the Modal logs."""
    request = api_pb2.ContainerLogRequest(message=message, input_id=current_input_id())
    io_manager = _ContainerIOManager._singleton
    assert io_manager is not None
    await retry_transient_errors(io_manager._client.stub.ContainerLog, request)


logger = synchronize_api(_logger)
