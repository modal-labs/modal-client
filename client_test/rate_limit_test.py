# Copyright Modal Labs 2022
import pytest

import modal
from modal.exception import InvalidError


def per_second_5():
    return 42


def per_minute_15():
    return 67


def dummy():
    pass


def test_rate_limit(client):
    stub = modal.Stub()

    per_second_5_modal = stub.function(per_second_5, rate_limit=modal.RateLimit(per_second=5))
    per_minute_15_modal = stub.function(per_minute_15, rate_limit=modal.RateLimit(per_minute=15))
    with pytest.raises(InvalidError):
        stub.function(dummy, rate_limit=modal.RateLimit(per_minute=15, per_second=5))

    with stub.run(client=client):
        per_second_5_modal.call()
        per_minute_15_modal.call()
