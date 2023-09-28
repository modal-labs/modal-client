# Copyright Modal Labs 2023
import importlib
import os
import pytest
import sys
from enum import IntEnum

from modal_docs.mdmd import mdmd

# Skipping a few tests on 3.7 - doesn't matter since we don't generate docs on 3.7
skip_37 = pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires Python 3.8")


def test_simple_function():
    def foo():
        pass

    assert (
        mdmd.function_str("bar", foo)
        == """```python
def bar():
```\n\n"""
    )


def test_simple_async_function():
    async def foo():
        pass

    assert (
        mdmd.function_str("bar", foo)
        == """```python
async def bar():
```\n\n"""
    )


def test_async_gen_function():
    async def foo():
        yield

    assert (
        mdmd.function_str("bar", foo)
        == """```python
async def bar():
```\n\n"""
    )


def test_complex_function_signature():
    def foo(a: str, *args, **kwargs):
        pass

    assert (
        mdmd.function_str("foo", foo)
        == """```python
def foo(a: str, *args, **kwargs):
```\n\n"""
    )


@skip_37
def test_function_has_docstring():
    def foo():
        """short description

        longer description"""

    assert (
        mdmd.function_str("foo", foo)
        == """```python
def foo():
```

short description

longer description
"""
    )


def test_simple_class_with_docstring():
    class Foo:
        """The all important Foo"""

        def bar(self, baz: str):
            """Bars the foo with the baz"""

    assert (
        mdmd.class_str("Foo", Foo)
        == """```python
class Foo(object)
```

The all important Foo

### bar

```python
def bar(self, baz: str):
```

Bars the foo with the baz
"""
    )


def test_enum():
    class Eee(IntEnum):
        FOO = 1
        BAR = 2
        XYZ = 3

    expected = """```python
class bar(enum.IntEnum)
```

An enumeration.

The possible values are:

* `FOO`
* `BAR`
* `XYZ`
"""

    assert mdmd.class_str("bar", Eee) == expected


def test_class_with_classmethod():
    class Foo:
        @classmethod
        def create_foo(cls, some_arg):
            pass

    assert (
        mdmd.class_str("Foo", Foo)
        == """```python
class Foo(object)
```

### create_foo

```python
@classmethod
def create_foo(cls, some_arg):
```

"""
    )


def test_class_with_baseclass_includes_base_methods():
    class Foo:
        def foo(self):
            pass

    class Bar(Foo):
        def bar(self):
            pass

    out = mdmd.class_str("Bar", Bar)
    assert "def foo(self):" in out


@skip_37
def test_module(monkeypatch):
    test_data_dir = os.path.join(os.path.dirname(__file__), "mdmd_data")
    monkeypatch.chdir(test_data_dir)
    monkeypatch.syspath_prepend(test_data_dir)
    test_module = importlib.import_module("foo")
    expected_output = open("./foo-expected.md").read()
    assert mdmd.module_str("foo", test_module) == expected_output


def test_docstring_format_reindents_code():
    assert (
        mdmd.format_docstring(
            """```python
        foo
            bar
        ```"""
        )
        == """```python
foo
    bar
```
"""
    )


def test_synchronicity_async_and_blocking_interfaces():
    from synchronicity import Synchronizer

    class Foo:
        """docky mcdocface"""

        async def foo(self):
            pass

        def bar(self):
            pass

    s = Synchronizer()
    AsyncFoo = s.create_async(Foo, "AsyncFoo")
    BlockingFoo = s.create_blocking(Foo, "BlockingFoo")

    assert (
        mdmd.class_str("AsyncFoo", AsyncFoo)
        == """```python
class AsyncFoo(object)
```

docky mcdocface

### foo

```python
async def foo(self):
```

### bar

```python
def bar(self):
```

"""
    )

    assert (
        mdmd.class_str("BlockingFoo", BlockingFoo)
        == """```python
class BlockingFoo(object)
```

docky mcdocface

### foo

```python
def foo(self):
```

### bar

```python
def bar(self):
```

"""
    )


def test_synchronicity_constructors():
    from synchronicity import Synchronizer

    class Foo:
        """docky mcdocface"""

        def __init__(self):
            """constructy mcconstructorface"""

    s = Synchronizer()
    AsyncFoo = s.create_async(Foo, "AsyncFoo")

    assert (
        mdmd.class_str("AsyncFoo", AsyncFoo)
        == """```python
class AsyncFoo(object)
```

docky mcdocface

```python
def __init__(self):
```

constructy mcconstructorface
"""
    )


def test_get_all_signature_comments():
    def foo(
        # prefix comment
        one,  # one comment
        two,  # two comment
        # postfix comment
    ) -> str:  # return value comment
        pass

    assert (
        mdmd.function_str("foo", foo)
        == """```python
def foo(
    # prefix comment
    one,  # one comment
    two,  # two comment
    # postfix comment
) -> str:  # return value comment
```

"""
    )


def test_get_decorators():
    BLA = 1

    def my_deco(arg):
        def wrapper(f):
            return f

        return wrapper

    @my_deco(BLA)
    def foo():
        pass

    assert (
        mdmd.function_str("foo", foo)
        == """```python
@my_deco(BLA)
def foo():
```

"""
    )
