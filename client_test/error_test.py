from modal import Error
from modal.exception import NotFoundError


def test_modal_errors():
    assert issubclass(NotFoundError, Error)
