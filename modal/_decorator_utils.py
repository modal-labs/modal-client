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
        if args and not kwargs and inspect.isfunction(args[-1]):
            # The decorator is invoked without arguments
            # Return a closure that consumes the function
            return dec_fun(*args)
        else:
            # The function is called with arguments
            # bind those arguments to the function and decorate the next token
            return functools.partial(dec_fun, *args, **kwargs)

    return wrapper
