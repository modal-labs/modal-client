# Copyright Modal Labs 2022
import os
import pytest
import tempfile

from modal import Secret, Stub
from modal.exception import InvalidError

from .supports.skip import skip_old_py


def test_secret_from_dict(servicer, client):
    stub = Stub()
    stub.secret = Secret.from_dict({"FOO": "hello, world"})
    with stub.run(client=client):
        assert stub.secret.object_id == "st-0"
        assert servicer.secrets["st-0"] == {"FOO": "hello, world"}


@skip_old_py("python-dotenv requires python3.8 or higher", (3, 8))
def test_secret_from_dotenv(servicer, client):
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(os.path.join(tmpdirname, ".env"), "w") as f:
            f.write("# My settings\nUSER=user\nPASSWORD=abc123\n")
        stub = Stub()
        stub.secret = Secret.from_dotenv(tmpdirname)
        with stub.run(client=client):
            assert stub.secret.object_id == "st-0"
            assert servicer.secrets["st-0"] == {"USER": "user", "PASSWORD": "abc123"}


def test_init_types():
    with pytest.raises(InvalidError):
        Secret.from_dict({"foo": 1.0})  # type: ignore


def test_secret_from_dict_none(servicer, client):
    stub = Stub()
    stub.secret = Secret.from_dict({"FOO": os.getenv("xyz"), "BAR": os.environ.get("abc"), "BAZ": "baz"})
    with stub.run(client=client):
        assert servicer.secrets["st-0"] == {"BAZ": "baz"}


def test_secret_from_name(servicer, client):
    # Deploy secret
    secret_id = Secret.create_deployed("my-secret", {"FOO": "123"}, client=client)

    # Look up secret
    secret = Secret.lookup("my-secret", client=client)
    assert secret.object_id == secret_id

    # Look up secret through app
    stub = Stub()
    stub.secret = Secret.from_name("my-secret")
    with stub.run(client=client):
        assert stub.secret.object_id == secret_id
