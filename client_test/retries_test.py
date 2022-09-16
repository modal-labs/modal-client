import pytest

import modal
from modal.exception import InvalidError


def test_retries(client):
    stub = modal.Stub()

    @stub.function(retries=5)
    def default_retries_from_int():
        pass

    @stub.function(retries=modal.Retries(max_retries=5, backoff_coefficient=1.0))
    def fixed_delay_retries():
        pass

    @stub.function(retries=modal.Retries(max_retries=2, initial_delay=2.0, backoff_coefficient=2.0))
    def exponential_backoff():
        return 67

    @stub.function(retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, max_delay=30.0))
    def exponential_with_max_delay():
        return 67

    with pytest.raises(TypeError):

        @stub.function(retries=modal.Retries())
        def default_retries_from_class():
            # Reject no-args constructions, which is unreadable and harder to support long-term
            pass

    # Reject weird inputs:
    # Don't need server to detect and reject nonsensical input. Can do client-side.
    with pytest.raises(InvalidError):

        @stub.function(retries=modal.Retries(max_retries=-2))
        def negative_retries():
            pass

        @stub.function(retries=modal.Retries(max_retries=2, backoff_coefficient=0.0))
        def zero_backoff_coeff():
            pass

    with stub.run(client=client):
        default_retries_from_int()
        fixed_delay_retries()
        exponential_backoff()
        exponential_with_max_delay()
