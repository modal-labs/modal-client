import pytest

from modal_utils.server_connection import GRPCConnectionFactory


def test_create_factory():
    cf = GRPCConnectionFactory("http://localhost:1234")
    assert cf.target == "localhost:1234"
    assert cf.credentials is None

    cf = GRPCConnectionFactory("http://localhost:1234", credentials=("foo", "bar"))
    assert cf.target == "localhost:1234"
    assert cf.credentials is not None

    with pytest.raises(AssertionError):
        # Non-localhost URLs must have TLS enabled.
        cf = GRPCConnectionFactory("http://not.a.real.domain:1234", credentials=("foo", "bar"))

    cf = GRPCConnectionFactory("https://api.modal.com", credentials=("foo", "bar"))
    assert cf.target == "api.modal.com"
    assert cf.credentials is not None
