# Copyright Modal Labs 2024
import sys
import warnings
from datetime import date

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
