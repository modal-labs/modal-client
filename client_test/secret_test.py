# Copyright Modal Labs 2022
import pytest

from modal import Secret, Stub
from modal.exception import DeprecationError, InvalidError
from modal.secret import AioSecret

from .supports.skip import skip_old_py


def test_old_secret(servicer, client):
    stub = Stub()
    with pytest.warns(DeprecationError):
        stub.secret = Secret({"FOO": "BAR"})
    with stub.run(client=client) as running_app:
        assert running_app.secret.object_id == "st-123"
        # TODO(erikbern): check secret content


def test_secret_from_dict(servicer, client):
    stub = Stub()
    stub.secret = Secret.from_dict({"FOO": "BAR"})
    with stub.run(client=client) as running_app:
        assert running_app.secret.object_id == "st-123"
        # TODO(erikbern): check secret content


@skip_old_py("python-dotenv requires python3.8 or higher", (3, 8))
def test_secret_from_dotenv(servicer, client):
    stub = Stub()
    stub.secret = Secret.from_dotenv()
    with stub.run(client=client) as running_app:
        assert running_app.secret.object_id == "st-123"
        # TODO(erikbern): check secret content


def test_init_types():
    with pytest.raises(InvalidError):
        Secret.from_dict({"foo": None})  # type: ignore

    with pytest.raises(InvalidError):
        AioSecret.from_dict({"foo": None})  # type: ignore
