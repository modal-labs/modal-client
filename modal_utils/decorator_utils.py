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
        if args and inspect.isfunction(args[-1]):
            # The decorator is invoked with a function as its first argument
            # Call the decorator function directly
            return dec_fun(*args, **kwargs)
        else:
            # The function is called with arguments
            # bind those arguments to the function and decorate the next token
            # args is only nonempty if it's the object the method is bound to
            return functools.partial(dec_fun, *args, **kwargs)

    return wrapper
