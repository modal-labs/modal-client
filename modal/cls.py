# Copyright Modal Labs 2022
import pickle
from datetime import date
from typing import Any, Callable, Dict, Type, TypeVar

from modal_utils.async_utils import synchronize_api

from .exception import deprecation_warning
from .functions import _Function

T = TypeVar("T")


class ClsMixin:
    def __init_subclass__(cls):
        deprecation_warning(date(2023, 9, 1), "`ClsMixin` is deprecated and can be safely removed.")

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


class _Obj:
    """An instance of a `Cls`, i.e. `Cls("foo", 42)` returns an `Obj`.

    All this class does is to return `Function` objects."""

    _functions: Dict[str, _Function]
    _has_local_obj: bool
    _local_obj: Any
    _local_obj_constr: Callable[[], Any]

    def __init__(self, user_cls: type, base_functions: Dict[str, _Function], args, kwargs):
        for i, arg in enumerate(args):
            check_picklability(i + 1, arg)
        for key, kwarg in kwargs.items():
            check_picklability(key, kwarg)

        self._functions = {}
        for k, fun in base_functions.items():
            self._functions[k] = fun.from_parametrized(self, args, kwargs)

        # Used for construction local object lazily
        self._has_local_obj = False
        self._local_obj = None
        self._local_obj_constr = lambda: user_cls(*args, **kwargs)

    def get_local_obj(self):
        # Construct local object lazily. Used for .local() calls
        if not self._has_local_obj:
            self._local_obj = self._local_obj_constr()
            setattr(self._local_obj, "_modal_functions", self._functions)  # Needed for PartialFunction.__get__
            self._has_local_obj = True

        return self._local_obj

    def __getattr__(self, k):
        return self._functions[k]


Obj = synchronize_api(_Obj)


class _Cls:
    # TODO(erikbern): Make this inherit from Object (needs backend support for this)
    _user_cls: type
    _functions: Dict[str, _Function]

    def __init__(self, user_cls, base_functions: Dict[str, _Function]):
        self._user_cls = user_cls
        self._base_functions = base_functions
        setattr(self._user_cls, "_modal_functions", base_functions)  # Needed for PartialFunction.__get__

    def get_user_cls(self):
        # Used by the container entrypoint
        return self._user_cls

    def get_base_function(self, k: str) -> _Function:
        return self._base_functions[k]

    def __call__(self, *args, **kwargs) -> _Obj:
        """This acts as the class constructor."""
        return _Obj(self._user_cls, self._base_functions, args, kwargs)

    async def remote(self, *args, **kwargs) -> _Obj:
        deprecation_warning(
            date(2023, 9, 1), "`Cls.remote(...)` on classes is deprecated. Use the constructor: `Cls(...)`."
        )
        return self(*args, **kwargs)

    def __getattr__(self, k):
        # Used by CLI and container entrypoint
        return self._base_functions[k]


Cls = synchronize_api(_Cls)
