# Copyright Modal Labs 2022
import functools
import inspect


def pretty_name(qualname):
    if "." in qualname:
        # convert _Stub.function to stub.function
        qualname = qualname.lstrip("_")
        qualname = qualname[0].lower() + qualname[1:]

    return qualname


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
