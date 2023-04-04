# Copyright Modal Labs 2022
import functools
import inspect


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
            return dec_fun(*args, **kwargs)
        else:
            # The function is called with arguments
            # bind those arguments to the function and decorate the next token
            # args is only nonempty if it's the object the method is bound to
            return functools.partial(dec_fun, *args, **kwargs)

    return wrapper


def decorator_with_options_deprecated(dec_fun):
    # Used when we are removing support for decorator_with_options
    @functools.wraps(dec_fun)
    def wrapper(*args, **kwargs):
        if len(args) >= 2 or (len(args) == 1 and inspect.isfunction(args[-1])):
            name = dec_fun.__name__
            raise RuntimeError(f"The function {name} needs to be used with arguments. Add () to it if there are none.")
        else:
            return functools.partial(dec_fun, *args, **kwargs)

    return wrapper
