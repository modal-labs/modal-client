# Copyright Modal Labs 2024
import functools
import sys
import warnings
from datetime import date
from typing import Any, Callable, TypeVar

from typing_extensions import ParamSpec  # Needed for Python 3.9

from ..exception import DeprecationError, PendingDeprecationError

# TODO(erikbern): we have something similready in function_utils.py
_INTERNAL_MODULES = ["modal", "synchronicity"]


def _is_internal_frame(frame):
    module = frame.f_globals["__name__"].split(".")[0]
    return module in _INTERNAL_MODULES


def deprecation_error(deprecated_on: tuple[int, int, int], msg: str):
    raise DeprecationError(f"Deprecated on {date(*deprecated_on)}: {msg}")


def deprecation_warning(
    deprecated_on: tuple[int, int, int], msg: str, *, pending: bool = False, show_source: bool = True
) -> None:
    """Utility for getting the proper stack entry.

    See the implementation of the built-in [warnings.warn](https://docs.python.org/3/library/warnings.html#available-functions).
    """
    filename, lineno = "<unknown>", 0
    if show_source:
        # Find the last non-Modal line that triggered the warning
        try:
            frame = sys._getframe()
            while frame is not None and _is_internal_frame(frame):
                frame = frame.f_back
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
        except ValueError:
            # Use the defaults from above
            pass

    warning_cls: type = PendingDeprecationError if pending else DeprecationError

    # This is a lower-level function that warnings.warn uses
    warnings.warn_explicit(f"{date(*deprecated_on)}: {msg}", warning_cls, filename, lineno)


P = ParamSpec("P")
R = TypeVar("R")


def renamed_parameter(
    date: tuple[int, int, int],
    old_name: str,
    new_name: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            mut_kwargs: dict[str, Any] = locals()["kwargs"]  # Avoid referencing kwargs directly due to bug in sigtools
            if old_name in mut_kwargs:
                mut_kwargs[new_name] = mut_kwargs.pop(old_name)
                func_name = func.__qualname__.removeprefix("_")  # Avoid confusion when synchronicity-wrapped
                message = (
                    f"The '{old_name}' parameter of `{func_name}` has been renamed to '{new_name}'."
                    "\nUsing the old name will become an error in a future release. Please update your code."
                )
                deprecation_warning(date, message, show_source=False)

            return func(*args, **kwargs)

        return wrapper

    return decorator
