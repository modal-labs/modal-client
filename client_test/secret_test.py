# Copyright Modal Labs 2022
import pytest

from modal import Secret, Stub
from modal.exception import InvalidError
from modal.secret import AioSecret


def test_secret(servicer, client):
    stub = Stub()
    stub.secret = Secret({"FOO": "BAR"})
    with stub.run(client=client) as running_app:
        assert running_app.secret.object_id == "st-123"


def test_init_types():
    with pytest.raises(InvalidError):
        Secret({"foo": None})  # type: ignore

    with pytest.raises(InvalidError):
        AioSecret({"foo": None})  # type: ignore
