# Copyright Modal Labs 2023
"""This module does cool stuff"""

# global untyped objects are currently not documented
some_dict = {}  # type: ignore


class Foo:
    """A class that foos"""

    def bar(self):
        pass


def funky():
    """funks the baz

    **Usage**

    ```python
    import foo
    foo.funky()  # outputs something
    ```

    Enjoy!
    """
    pass


def hidden():
    """mdmd:hidden

    This is marked as hidden in docs and shouldn't be shown"""
