# Copyright Modal Labs 2022
import pytest

from modal.exception import DeprecationError
from modal_test_support.functions import deprecated_function

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
    import modal_test_support.functions

    assert record[0].filename == modal_test_support.functions.__file__
