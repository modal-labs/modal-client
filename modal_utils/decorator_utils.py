# Copyright Modal Labs 2022

import datetime
import functools
import inspect
import sys
from typing import Any, Callable, TypeVar

if sys.version_info >= (3, 10):
    from typing import Concatenate, ParamSpec
else:
    from typing_extensions import Concatenate, ParamSpec


def pretty_name(qualname):
    if "." in qualname:
        # convert _Stub.function to stub.function
        qualname = qualname.lstrip("_")
        qualname = qualname[0].lower() + qualname[1:]

    return qualname


def decorator_with_options(dec_fun):
    """Makes it possible for function fun to be used with and without arguments:

    @fun
    def f(x): ...

    @fun(foo=bar)
    def f(x): ...
    """

    @functools.wraps(dec_fun)
    def wrapper(*args, **kwargs):
        # Note: if the def_fun is a method, then args will contain the object the method is bound to.
        # TODO(erikbern): this could be solved using the descriptor protocol, but this would also require
        # synchronicity to implement the full descriptor protocol. This isn't trivial, but we should
        # do that some day!
        if len(args) >= 2 or (len(args) == 1 and inspect.isfunction(args[-1])):
            # The decorator is invoked with a function as its first argument
            # Call the decorator function directly
            name = pretty_name(dec_fun.__qualname__)
            from modal.exception import deprecation_warning

            deprecation_warning(
                datetime.date(2023, 4, 5),
                f"The decorator {name} without arguments will soon be deprecated. Add empty parens to it, e.g. @{name}() if there are no arguments",
            )
            return dec_fun(*args, **kwargs)
        else:
            # The function is called with arguments
            # bind those arguments to the function and decorate the next token
            # args is only nonempty if it's the object the method is bound to
            return functools.partial(dec_fun, *args, **kwargs)

    return wrapper


def decorator_with_options_unsupported(dec_fun):
    # Used when we are removing support for decorator_with_options
    @functools.wraps(dec_fun)
    def wrapper(*args, **kwargs):
        if len(args) >= 2 or (len(args) == 1 and inspect.isfunction(args[-1])):
            name = pretty_name(dec_fun.__qualname__)
            raise RuntimeError(
                f"The decorator {name} needs to be called before decorating a function. Add empty parens to it, e.g. @{name}() if there are no arguments"
            )
        else:
            return functools.partial(dec_fun, *args, **kwargs)

    return wrapper


P = ParamSpec("P")
R = TypeVar("R")


def decorator_simplify(
    dec_fun: Callable[Concatenate[Callable[..., Any], P], R]
) -> Callable[P, Callable[[Callable[..., Any]], R]]:
    """Compatible with decorator_with_options_unsupported but preserves type hints

    Usage:
    ```
    @decorator_simplify
    def decorator(raw_f: Callable[..., Any], xyz: int = 7) -> Whatever:
        return Whatever(raw_f, xyz)


    @decorator(xyz = 42)
    def f():
        return 77

    assert isinstance(f, Whatever)
    ```
    """

    @functools.wraps(dec_fun)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Callable[[Callable[..., Any]], R]:
        return functools.partial(dec_fun, *args, **kwargs)

    return wrapper
