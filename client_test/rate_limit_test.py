import pytest

import modal
from modal.exception import InvalidError


def test_rate_limit(client):
    stub = modal.Stub()

    @stub.function(rate_limit=modal.RateLimit(per_second=5))
    def per_second_5():
        return 42

    @stub.function(rate_limit=modal.RateLimit(per_minute=15))
    def per_minute_15():
        return 67

    with pytest.raises(InvalidError):

        @stub.function(rate_limit=modal.RateLimit(per_minute=15, per_second=5))
        def two_params_limit():
            pass

    with stub.run(client=client):
        per_second_5()
        per_minute_15()
