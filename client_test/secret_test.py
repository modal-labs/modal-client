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
        assert servicer.secrets["st-0"].env_dict == {"FOO": "hello, world"}


@skip_old_py("python-dotenv requires python3.8 or higher", (3, 8))
def test_secret_from_dotenv(servicer, client):
    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(os.path.join(tmpdirname, ".env"), "w") as f:
            f.write("# My settings\nUSER=user\nPASSWORD=abc123\n")
        stub = Stub()
        stub.secret = Secret.from_dotenv(tmpdirname)
        with stub.run(client=client):
            assert stub.secret.object_id == "st-0"
            assert servicer.secrets["st-0"].env_dict == {"USER": "user", "PASSWORD": "abc123"}


def test_init_types():
    with pytest.raises(InvalidError):
        Secret.from_dict({"foo": None})  # type: ignore
