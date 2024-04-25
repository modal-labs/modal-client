# Copyright Modal Labs 2023
import pytest

from modal import App
from modal.exception import InvalidError


def dummy():
    pass


def test_cpu_lower_bound(client, servicer):
    app = App()

    app.function(cpu=0.0)(dummy)

    with pytest.raises(InvalidError):
        with app.run(client=client):
            pass

    app.function(cpu=42)(dummy)
    with app.run(client=client):
        pass
