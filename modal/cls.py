# Copyright Modal Labs 2022
import pickle
from typing import Dict, Type, TypeVar

from modal_utils.async_utils import synchronize_api

from ._resolver import Resolver
from .functions import PartialFunction, _Function, _PartialFunction

T = TypeVar("T")


class ClsMixin:
    @classmethod
    def remote(cls: Type[T], *args, **kwargs) -> T:
        ...

    @classmethod
    async def aio_remote(cls: Type[T], *args, **kwargs) -> T:
        ...


def check_picklability(key, arg):
    try:
        pickle.dumps(arg)
    except Exception:
        raise ValueError(
            f"Only pickle-able types are allowed in remote class constructors: argument {key} of type {type(arg)}."
        )


def make_remote_cls_constructors(
    user_cls: type,
    partial_functions: Dict[str, PartialFunction],
    functions: Dict[str, _Function],
):
    cls = type(f"Remote{user_cls.__name__}", (), partial_functions)

    async def _remote(*args, **kwargs):
        for i, arg in enumerate(args):
            check_picklability(i + 1, arg)
        for key, kwarg in kwargs.items():
            check_picklability(key, kwarg)

        new_functions: Dict[str, _Function] = {}

        for k, v in partial_functions.items():
            base_function: _Function = functions[k]
            new_function: _Function = base_function.from_parametrized(args, kwargs)
            resolver = Resolver()
            await resolver.load(new_function)
            new_functions[k] = new_function

        obj = cls()
        _PartialFunction.initialize_obj(obj, new_functions)
        return obj

    return synchronize_api(_remote)
