from modal_utils.server_connection import GRPCConnectionFactory


def test_create_factory():
    cf = GRPCConnectionFactory("http://localhost:1234")
    assert cf.target == "localhost:1234"
    assert cf.credentials is None

    cf = GRPCConnectionFactory("http://localhost:1234", credentials=("foo", "bar"))
    assert cf.target == "localhost:1234"
    assert cf.credentials is not None

    cf = GRPCConnectionFactory("http://host.docker.internal:1234", credentials=("foo", "bar"))
    assert cf.target == "host.docker.internal:1234"
    assert cf.credentials is None  # This is a special case where we have to remove credentails

    cf = GRPCConnectionFactory("https://api.modal.com", credentials=("foo", "bar"))
    assert cf.target == "api.modal.com"
    assert cf.credentials is not None
