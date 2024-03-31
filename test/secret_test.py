# Copyright Modal Labs 2022
import os
import pytest
import tempfile
from unittest import mock

from modal import Secret, Stub
from modal.exception import InvalidError

from .supports.skip import skip_old_py


def dummy():
    ...


def test_secret_from_dict(servicer, client):
    stub = Stub()
    secret = Secret.from_dict({"FOO": "hello, world"})
    stub.function(secrets=[secret])(dummy)
    with stub.run(client=client):
        assert secret.object_id == "st-0"
        assert servicer.secrets["st-0"] == {"FOO": "hello, world"}


@skip_old_py("python-dotenv requires python3.8 or higher", (3, 8))
def test_secret_from_dotenv(servicer, client):
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(os.path.join(tmpdirname, ".env"), "w") as f:
            f.write("# My settings\nUSER=user\nPASSWORD=abc123\n")
        stub = Stub()
        secret = Secret.from_dotenv(tmpdirname)
        stub.function(secrets=[secret])(dummy)
        with stub.run(client=client):
            assert secret.object_id == "st-0"
            assert servicer.secrets["st-0"] == {"USER": "user", "PASSWORD": "abc123"}

@mock.patch.dict(os.environ, {"FOO": "easy", "BAR": "1234"})
def test_secret_from_local_environ(servicer, client):
    stub = Stub()
    secret = Secret.from_local_environ(["FOO", "BAR"])
    stub.function(secrets=[secret])(dummy)
    with stub.run(client=client):
        assert secret.object_id == "st-0"
        assert servicer.secrets["st-0"] == {"FOO": "easy", "BAR": "1234"}

    with pytest.raises(InvalidError, match="NOTFOUND"):
        Secret.from_local_environ(["FOO", "NOTFOUND"])



def test_init_types():
    with pytest.raises(InvalidError):
        Secret.from_dict({"foo": 1.0})  # type: ignore


def test_secret_from_dict_none(servicer, client):
    stub = Stub()
    secret = Secret.from_dict({"FOO": os.getenv("xyz"), "BAR": os.environ.get("abc"), "BAZ": "baz"})
    stub.function(secrets=[secret])(dummy)
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
    secret = Secret.from_name("my-secret")
    stub.function(secrets=[secret])(dummy)
    with stub.run(client=client):
        assert secret.object_id == secret_id
