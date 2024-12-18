# Copyright Modal Labs 2024
import functools
import sys
import warnings
from datetime import date
from typing import Any, Callable, TypeVar

from typing_extensions import ParamSpec  # Needed for Python 3.9

from ..exception import DeprecationError, PendingDeprecationError

_INTERNAL_MODULES = ["modal", "synchronicity"]


def _is_internal_frame(frame):
    module = frame.f_globals["__name__"].split(".")[0]
    return module in _INTERNAL_MODULES


def deprecation_error(deprecated_on: tuple[int, int, int], msg: str):
    raise DeprecationError(f"Deprecated on {date(*deprecated_on)}: {msg}")


def deprecation_warning(
    deprecated_on: tuple[int, int, int], msg: str, *, pending: bool = False, show_source: bool = True
) -> None:
    """Issue a Modal deprecation warning with source optionally attributed to user code.

    See the implementation of the built-in [warnings.warn](https://docs.python.org/3/library/warnings.html#available-functions).
    """
    filename, lineno = "<unknown>", 0
    if show_source:
        # Find the last non-Modal line that triggered the warning
        try:
            frame = sys._getframe()
            while frame is not None and _is_internal_frame(frame):
                frame = frame.f_back
            if frame is not None:
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
        except ValueError:
            # Use the defaults from above
            pass

    warning_cls = PendingDeprecationError if pending else DeprecationError

    # This is a lower-level function that warnings.warn uses
    warnings.warn_explicit(f"{date(*deprecated_on)}: {msg}", warning_cls, filename, lineno)


P = ParamSpec("P")
R = TypeVar("R")


def renamed_parameter(
    date: tuple[int, int, int],
    old_name: str,
    new_name: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for semi-gracefully changing a parameter name.

    Functions wrapped with this decorator can be defined using only the `new_name` of the parameter.
    If the function is invoked with the `old_name`, the wrapper will pass the value as a keyword
    argument for `new_name` and issue a Modal deprecation warning about the change.

    Note that this only prevents parameter renamings from breaking code at runtime.
    Type checking will fail when code uses `old_name`. To avoid this, the `old_name` can be
    preserved in the function signature with an `Annotated` type hint indicating the renaming.
    """

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
