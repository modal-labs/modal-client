import pytest

from modal._test_support.functions import deprecated_function

# Not a pytest unit test, but an extra assertion that we catch issues in global scope too
# See #2228
exc = None
try:
    deprecated_function(42)
except Exception as e:
    exc = e
finally:
    assert isinstance(exc, DeprecationWarning)


def test_deprecation():
    # See conftest.py in the root of the repo
    # All deprecation warnings in modal during tests will trigger exceptions
    with pytest.raises(DeprecationWarning):
        deprecated_function(42)

    # With this context manager, it doesn't raise an exception, but we record
    # the warning. This is the normal behavior outside of pytest.
    with pytest.deprecated_call():
        res = deprecated_function(42)
        assert res == 1764
