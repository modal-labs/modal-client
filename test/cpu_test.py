# Copyright Modal Labs 2023
import pytest

from modal import Stub
from modal.exception import InvalidError


def dummy():
    pass


def test_cpu_lower_bound(client, servicer):
    stub = Stub()

    stub.function(cpu=0.0)(dummy)

    with pytest.raises(InvalidError):
        with stub.run(client=client):
            pass

    stub.function(cpu=42)(dummy)
    with stub.run(client=client):
        pass
