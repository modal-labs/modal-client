# Copyright Modal Labs 2022
import contextlib
import sys
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


class DemuxStream:
    """
    Demultiplexes a stream from multiple concurrent inputs.
    """

    def __init__(self, passthrough: any, fd: api_pb2.FileDescriptor, io_manager: "_ContainerIOManager"):
        self.passthrough = passthrough
        self.fd = fd
        self.io_manager = io_manager

    def write(self, data: str):
        input_id, function_call_id = current_input_id(), current_function_call_id()
        if input_id is None or function_call_id is None:
            self.passthrough.write(data + " (no tag)")
        else:
            self.io_manager.log_accumulator.add(
                input_id,
                function_call_id,
                self.fd,
                data,
            )

    def flush(self):
        pass

    def getvalue(self):
        return ""


@contextlib.contextmanager
def tagged_logs():
    io_manager = _ContainerIOManager._singleton
    assert io_manager is not None

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        sys.stdout = DemuxStream(original_stdout, api_pb2.FileDescriptor.FILE_DESCRIPTOR_STDOUT, io_manager)
        sys.stderr = DemuxStream(original_stderr, api_pb2.FileDescriptor.FILE_DESCRIPTOR_STDERR, io_manager)
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
