# Copyright Modal Labs 2022
from typing import (
    Any,
    Callable,
)

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
    """Run the function on a cluster of colocated and networked containers.

    Parameters:
    size: int
        Number of containers spun up to handle each input.
    broadcast: bool = True
        If True, inputs will be sent synchronously to each container. Otherwise,
        inputs will be sent only to the rank-0 container, which is responsible for
        delegating to the workers. Containers with input broadcasting enabled
        will shut down after processing a single input.
    """

    assert broadcast, "broadcast=False is not yet implemented"

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


# def _networked(func):
#     """This function handles i6pn address sharing between the main container and its peers.

#     It pops modal_rank, modal_size, and modal_queue from the kwargs of the function.
#     The container with a modal_rank of 0 is the main container and is responsible for sharing its i6pn address
#     through a modal queue.
#     All containers will block until they are able to acquire the i6pn address from the ephemeral modal queue.
#     This behavior ensures that on entry to the wrapped function, appropriate environment variables will have
#     been set for communication between containers.
#     """

#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         import os
#         import socket

#         def get_i6pn():
#             """Returns the ipv6 address assigned to this container."""
#             return socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)[0][4][0]

#         hostname = socket.gethostname()
#         addr_info = get_i6pn()
#         # nccl's default host ID is $(hostname)$(cat /proc/sys/kernel/random/boot_id).
#         # on runc, if two i6pn-linked containers get scheduled on the same worker,
#         # their boot ID and hostname will both be identical, causing nccl to break.
#         # As a workaround, we can explicitly specify a unique host ID here.
#         # See MOD-4067.
#         os.environ["NCCL_HOSTID"] = f"{hostname}{addr_info}"

#         rank = kwargs.pop("modal_rank", None)
#         size = kwargs.pop("modal_size", None)
#         q = kwargs.pop("modal_q", None)

#         if rank is None or size is None or q is None:
#             raise ValueError("Missing required arguments; `_networked` must be called using `grouped` decorator")
#         elif rank == 0:
#             q.put_many([addr_info for _ in range(size)])
#         main_ip = q.get()
#         assert main_ip is not None, "Failed to get main i6pn address"

#         os.environ["MODAL_MAIN_I6PN"] = f"{main_ip}"
#         os.environ["MODAL_WORLD_SIZE"] = f"{size}"
#         os.environ["MODAL_CONTAINER_RANK"] = f"{rank}"

#         return func(*args, **kwargs)

#     return wrapper
