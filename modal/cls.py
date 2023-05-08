# Copyright Modal Labs 2022
import inspect
from typing import Dict, Union
from modal_utils.async_utils import synchronize_apis
from .functions import _PartialFunction, PartialFunction, AioPartialFunction, _FunctionHandle


class ClsMixin:
    @staticmethod
    def remote(*args, **kwargs):
        ...

    @staticmethod
    async def aio_remote(*args, **kwargs):
        ...


def make_remote_cls_constructors(
    user_cls: type,
    partial_functions: Dict[str, Union[PartialFunction, AioPartialFunction]],
    function_handles: Dict[str, _FunctionHandle],
):
    original_sig = inspect.signature(user_cls.__init__)  # type: ignore
    new_parameters = [param for name, param in original_sig.parameters.items() if name != "self"]
    sig = inspect.Signature(new_parameters)
    # TODO: validate signature has only primitive types.

    async def _remote(*args, **kwargs):
        params = sig.bind(*args, **kwargs)

        cls_dict = {}
        new_function_handles: Dict[str, _FunctionHandle] = {}

        for k, v in partial_functions.items():
            new_function_handles[k] = await function_handles[k].make_bound_function_handle(params)
            cls_dict[k] = v

        cls = type(f"Remote{user_cls.__name__}", (), cls_dict)
        _PartialFunction.initialize_cls(cls, new_function_handles)
        return cls()

    return synchronize_apis(_remote)
