import inspect
import pytest

from modal._utils.deprecation_utils import renamed_parameter
from modal.exception import DeprecationError


@renamed_parameter((2024, 12, 1), "bar", "foo")
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
