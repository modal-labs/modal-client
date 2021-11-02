from polyester.server_connection import GRPCConnectionFactory


async def test_create_factory():
    cf = GRPCConnectionFactory("http://localhost:1234")
    assert cf.target == "localhost:1234"
    assert cf.credentials is None

    cf = GRPCConnectionFactory("http://localhost:1234", credentials=("foo", "bar"))
    assert cf.target == "localhost:1234"
    assert cf.credentials is not None

    cf = GRPCConnectionFactory("http://host.docker.internal:1234", credentials=("foo", "bar"))
    assert cf.target == "host.docker.internal:1234"
    assert cf.credentials is None  # This is a special case where we have to remove credentails

    cf = GRPCConnectionFactory("https://api.polyester.cloud", credentials=("foo", "bar"))
    assert cf.target == "api.polyester.cloud"
    assert cf.credentials is not None
