# Copyright Modal Labs 2022
import inspect
import pytest

from modal._utils.deprecation import renamed_parameter
from modal.exception import DeprecationError

from .supports.functions import deprecated_function

# Not a pytest unit test, but an extra assertion that we catch issues in global scope too
# See #2228
exc = None
try:
    deprecated_function(42)
except Exception as e:
    exc = e
finally:
    assert isinstance(exc, DeprecationError)  # If you see this, try running `pytest client/client_test`


def test_deprecation():
    # See conftest.py in the root of the repo
    # All deprecation warnings in modal during tests will trigger exceptions
    with pytest.raises(DeprecationError):
        deprecated_function(42)

    # With this context manager, it doesn't raise an exception, but we record
    # the warning. This is the normal behavior outside of pytest.
    with pytest.warns(DeprecationError) as record:
        res = deprecated_function(42)
        assert res == 1764

    # Make sure it raises in the right file
    from .supports import functions

    assert record[0].filename == functions.__file__


@renamed_parameter((2024, 12, 1), "foo", "bar")
def my_func(bar: int) -> int:
    return bar**2


def test_renamed_parameter():
    message = "The 'foo' parameter .+ has been renamed to 'bar'"
    with pytest.warns(DeprecationError, match=message):
        res = my_func(foo=2)  # type: ignore
        assert res == 4
    assert my_func(bar=3) == 9
    assert my_func(4) == 16

    sig = inspect.signature(my_func)
    assert "bar" in sig.parameters
    assert "foo" not in sig.parameters
