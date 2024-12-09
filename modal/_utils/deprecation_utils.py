import functools
from typing import Callable, ParamSpec, TypeVar

from ..exception import deprecation_warning  # TODO move that helper into this module?

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
            if old_name in kwargs:
                kwargs[new_name] = kwargs.pop(old_name)
                func_name = func.__qualname__.removeprefix("_")  # Avoid confusion when synchronicity-wrapped
                message = (
                    f"The '{old_name}' parameter of `{func_name}` has been renamed to '{new_name}'."
                    "\nUsing the old name will become an error in a future release. Please update your code."
                )
                deprecation_warning(date, message, show_source=False)
            return func(*args, **kwargs)

        return wrapper

    return decorator
