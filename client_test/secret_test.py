# Copyright Modal Labs 2022
import pytest
import typeguard

from modal import Secret, Stub
from modal.secret import AioSecret


def test_secret(servicer, client):
    stub = Stub()
    stub.secret = Secret({"FOO": "BAR"})
    with stub.run(client=client) as running_app:
        assert running_app.secret.object_id == "st-123"


def test_init_types():
    with pytest.raises(typeguard.TypeCheckError):
        Secret({"foo": None})

    with pytest.raises(typeguard.TypeCheckError):
        AioSecret({"foo": None})
