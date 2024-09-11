# Copyright Modal Labs 2022
import typing
from typing import (
    Any,
    Callable,
    List,
)

import modal

from ._container_io_manager import _ContainerIOManager
from ._utils.async_utils import (
    synchronize_api,
)
from .exception import (
    InvalidError,
)
from .functions import FunctionCall, OriginalReturnType, P, ReturnType, _Function
from .object import _Object
from .partial_function import _PartialFunction, _PartialFunctionFlags


def stop_fetching_inputs():
    """Don't fetch any more inputs from the server, after the current one.
    The container will exit gracefully after the current input is processed."""

    _ContainerIOManager.stop_fetching_inputs()


def get_local_input_concurrency():
    """Get the container's local input concurrency. Return 0 if the container is not running."""

    return _ContainerIOManager.get_input_concurrency()


# START Experimental: Container Networking


class _GroupedFunction(typing.Generic[P, ReturnType, OriginalReturnType], _Object, type_prefix="gf"):
    def __init__(self, f: _Function, size: int):
        self.f = synchronize_api(f)
        self.size = size

    def remote(self, *args: P.args, **kwargs: P.kwargs) -> List[ReturnType]:
        """
        Calls the function remotely, executing it with the given arguments and returning the execution's result.
        """
        worker_handles: List[FunctionCall] = []
        with modal.Queue.ephemeral() as q:
            for i in range(self.size):
                handle = self.f.spawn(*args, **{**kwargs, "rank": i, "size": self.size, "q": q})
                worker_handles.append(handle)
        output: list[ReturnType] = []
        for i, handle in enumerate(worker_handles):
            output.append(handle.get())
        return output

    def local(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        raise NotImplementedError("Grouped function cannot be run locally")

    def spawn(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        raise NotImplementedError("Grouped function cannot be spawned")


def grouped(size=1):
    def wrapper(raw_f: Callable[..., Any]) -> _PartialFunction:
        if isinstance(raw_f, _Function):
            raw_f = raw_f.get_raw_f()
            raise InvalidError(
                f"Applying decorators for {raw_f} in the wrong order!\nUsage:\n\n"
                "@app.function()\n@modal.grouped()\ndef grouped_function():\n    ..."
            )
        raw_f = _networked(raw_f)
        return _PartialFunction(raw_f, _PartialFunctionFlags.FUNCTION | _PartialFunctionFlags.GROUPED, group_size=size)

    return wrapper


def _networked(func):
    def wrapper(*args, **kwargs):
        import os

        rank = kwargs.pop("rank", None)
        size = kwargs.pop("size", None)
        q = kwargs.pop("q", None)

        if rank is None or size is None or q is None:
            raise ValueError("This must be called using grouped decorator")

        if not rank:
            import socket

            addr_info = socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)
            # Extract IPv6 addresses from the results
            ipv6_addresses = [
                addr[4][0] for addr in addr_info if addr[1] == socket.SOCK_STREAM and "fdaa" in addr[4][0]
            ]
            main_ip = ipv6_addresses[0]
            q.put_many([main_ip for _ in range(size)])
        main_ip = q.get()

        os.environ["MODAL_MAIN_I6PN"] = f"{main_ip}"
        os.environ["MODAL_WORLD_SIZE"] = f"{size}"
        os.environ["MODAL_CONTAINER_RANK"] = f"{rank}"

        return func(*args, **kwargs)

    return wrapper


# END Experimental: Container Networking
