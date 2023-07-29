# Copyright Modal Labs 2022
import inspect
import pickle
from typing import Dict, Type, TypeVar

from modal_utils.async_utils import synchronize_api

from .functions import PartialFunction, _FunctionHandle, _PartialFunction

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
    function_handles: Dict[str, _FunctionHandle],
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
        new_function_handles: Dict[str, _FunctionHandle] = {}

        for k, v in partial_functions.items():
            new_function_handles[k] = await function_handles[k]._make_bound_function_handle(
                *params.args, **params.kwargs
            )
            cls_dict[k] = v

        cls = type(f"Remote{user_cls.__name__}", (), cls_dict)
        _PartialFunction.initialize_cls(cls, new_function_handles)
        return cls()

    return synchronize_api(_remote)
