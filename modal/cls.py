# Copyright Modal Labs 2022
import inspect
import pickle
from typing import Dict, Type, TypeVar

from modal_utils.async_utils import synchronize_api

from ._resolver import Resolver
from .client import _Client
from .functions import PartialFunction, _Function, _FunctionHandle, _PartialFunction

T = TypeVar("T")


class ClsMixin:
    @classmethod
    def remote(cls: Type[T], *args, **kwargs) -> T:
        ...

    @classmethod
    async def aio_remote(cls: Type[T], *args, **kwargs) -> T:
        ...


def make_remote_cls_constructors(
    user_cls: type,
    partial_functions: Dict[str, PartialFunction],
    functions: Dict[str, _Function],
):
    original_sig = inspect.signature(user_cls.__init__)  # type: ignore
    new_parameters = [param for name, param in original_sig.parameters.items() if name != "self"]
    sig = inspect.Signature(new_parameters)

    async def _remote(*args, **kwargs):
        params = sig.bind(*args, **kwargs)

        for name, param in params.arguments.items():
            try:
                pickle.dumps(param)
            except Exception:
                raise ValueError(
                    f"Only pickle-able types are allowed in remote class constructors. "
                    f"Found {name}={param} of type {type(param)}."
                )

        cls_dict = {}
        new_functions: Dict[str, _Function] = {}

        for k, v in partial_functions.items():
            base_function: _Function = functions[k]
            client: _Client = base_function._client
            new_function: _Function = _Function.from_parametrized(base_function, *params.args, **params.kwargs)
            resolver = Resolver(client)
            await resolver.load(new_function)
            new_functions[k] = new_function
            cls_dict[k] = v

        cls = type(f"Remote{user_cls.__name__}", (), cls_dict)
        _PartialFunction.initialize_cls(cls, new_functions)
        return cls()

    return synchronize_api(_remote)
