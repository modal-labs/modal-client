# Copyright Modal Labs 2022
import typing
from functools import wraps
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


class _GroupedFunctionCall:
    """Wrapper around _FunctionCall that allows for grouped functions to be spawned."""

    def __init__(self, handles: List[FunctionCall]):
        self.handles: List[FunctionCall] = handles

    def get(self) -> List[ReturnType]:
        """Get the result of a grouped function call."""
        output: List[ReturnType] = []
        for handle in self.handles:
            output.append(handle.get())
        return output

    def get_gen(self) -> ReturnType:
        raise NotImplementedError("Grouped functions cannot be generators")

    def get_call_graph(self) -> ReturnType:
        raise NotImplementedError("Grouped functions do not show call graph")

    def cancel(
        self,
        terminate_containers: bool = False,
    ):
        for handle in self.handles:
            handle.cancel(terminate_containers)


class _GroupedFunction(typing.Generic[P, ReturnType, OriginalReturnType], _Object, type_prefix="gf"):
    """Experimental wrapper around _Function that allows for containers to be spun up concurrently."""

    def __init__(self, f: _Function, size: int):
        self.raw_f = f.raw_f
        self.f = synchronize_api(f)
        self.size = size

    def remote(self, *args: P.args, **kwargs: P.kwargs) -> List[ReturnType]:
        """
        Calls the function remotely, executing it with the given arguments and returning the execution's result.
        """
        handler = self.spawn(*args, **kwargs)
        return handler.get()

    def local(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        raise NotImplementedError("Grouped function cannot be run locally")

    def spawn(self, *args: P.args, **kwargs: P.kwargs) -> _GroupedFunctionCall:
        worker_handles: List[FunctionCall] = []
        with modal.Queue.ephemeral() as q:
            for i in range(self.size):
                handle = self.f.spawn(*args, **kwargs, modal_rank=i, modal_size=self.size, modal_q=q)
                worker_handles.append(handle)
        handler = _GroupedFunctionCall(worker_handles)
        return handler

    def keep_warm(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        raise NotImplementedError("Grouped function cannot be kept warm")

    def from_name(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        raise NotImplementedError("Grouped function cannot be retrieved from name")

    def lookup(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        raise NotImplementedError("Grouped function cannot be looked up")

    def web_url(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        raise NotImplementedError("Grouped function does not have a web url")

    def remote_gen(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        raise NotImplementedError("Grouped function does not work with generators")

    def shell(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        raise NotImplementedError("Grouped function does not work with shell")

    def get_raw_f(self) -> Callable[..., Any]:
        return self.get_raw_f()

    def get_current_stats(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        raise NotImplementedError("Grouped function does not track queue and runner counts")

    def map(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        raise NotImplementedError("Grouped function does not work with map")

    def starmap(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        raise NotImplementedError("Grouped function does not work with star map")

    def for_each(self, *args: P.args, **kwargs: P.kwargs) -> ReturnType:
        raise NotImplementedError("Grouped function does not work with for each")


def grouped(size: int):
    """This wrapper defines the underlying raw function as a grouped function.

    The underlying function is wrapper with _network to ensure that container
    i6pn addresses are synchronized across all machines.
    When wrapped with app.function(), a _GroupedFunction will be produced instead of
    a Function. The call logic for a _GroupedFunction is defined above.
    """

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
    """This function handles i6pn address sharing between the main container and its peers.

    It pops modal_rank, modal_size, and modal_queue from the kwargs of the function.
    The container with a modal_rank of 0 is the main container and is responsible for sharing its i6pn address
    through a modal queue.
    All containers will block until they are able to acquire the i6pn address from the ephemeral modal queue.
    This behavior ensures that on entry to the wrapped function, appropriate environment variables will have
    been set for communication between containers.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        import os

        rank = kwargs.pop("modal_rank", None)
        size = kwargs.pop("modal_size", None)
        q = kwargs.pop("modal_q", None)

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
