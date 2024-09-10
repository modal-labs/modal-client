# Copyright Modal Labs 2022
from typing import Literal

from modal.execution_context import current_function_call_id, current_input_id
from modal_proto import api_pb2

from ._container_io_manager import _ContainerIOManager


def stop_fetching_inputs():
    """Don't fetch any more inputs from the server, after the current one.
    The container will exit gracefully after the current input is processed."""

    _ContainerIOManager.stop_fetching_inputs()


def get_local_input_concurrency():
    """Get the container's local input concurrency. Return 0 if the container is not running."""

    return _ContainerIOManager.get_input_concurrency()


def log(message: str, end: str = "\n", stream: Literal["stdout", "stderr"] = "stdout") -> None:
    """Add message to the current input's log for structured logging."""
    io_manager = _ContainerIOManager._singleton
    assert io_manager is not None
    fd = (
        api_pb2.FileDescriptor.FILE_DESCRIPTOR_STDOUT
        if stream == "stdout"
        else api_pb2.FileDescriptor.FILE_DESCRIPTOR_STDERR
    )
    io_manager.log_accumulator.add(
        input_id=current_input_id(),
        function_call_id=current_function_call_id(),
        fd=fd,
        data=message + end,
    )
