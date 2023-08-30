# Copyright Modal Labs 2022
from datetime import date
import pickle
from typing import Dict, Type, TypeVar

from modal_utils.async_utils import synchronize_api

from .functions import PartialFunction, _Function, _PartialFunction
from .exception import deprecation_warning

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


def wrap_cls(
    user_cls: type,
    partial_functions: Dict[str, PartialFunction],
    functions: Dict[str, _Function],
):
    def _new(cls, *args, **kwargs):
        for i, arg in enumerate(args):
            check_picklability(i + 1, arg)
        for key, kwarg in kwargs.items():
            check_picklability(key, kwarg)

        new_functions: Dict[str, _Function] = {}

        obj = super(user_cls, user_cls).__new__(user_cls)
        obj.__init__(*args, **kwargs)

        for k, v in partial_functions.items():
            new_functions[k] = functions[k].from_parametrized(obj, args, kwargs)

        _PartialFunction.initialize_obj(obj, new_functions)
        return obj

    async def _remote(*args, **kwargs):
        deprecation_warning(
            date(2023, 8, 29),
            "`Cls.remote(...)` on classes is deprecated. Use the constructor: `Cls(...)`."
        )
        return _new(None, *args, **kwargs)

    user_cls.__new__ = _new
    user_cls.remote = synchronize_api(_remote)
