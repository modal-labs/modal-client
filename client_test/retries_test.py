# Copyright Modal Labs 2022
import pytest

import modal
from modal.exception import InvalidError


def default_retries_from_int():
    pass


def fixed_delay_retries():
    pass


def exponential_backoff():
    return 67


def exponential_with_max_delay():
    return 67


def dummy():
    pass


def test_retries(client):
    stub = modal.Stub()

    default_retries_from_int_modal = stub.function(default_retries_from_int, retries=5)
    fixed_delay_retries_modal = stub.function(
        fixed_delay_retries, retries=modal.Retries(max_retries=5, backoff_coefficient=1.0)
    )
    exponential_backoff_modal = stub.function(
        exponential_backoff, retries=modal.Retries(max_retries=2, initial_delay=2.0, backoff_coefficient=2.0)
    )
    exponential_with_max_delay_modal = stub.function(
        exponential_with_max_delay, retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, max_delay=30.0)
    )

    with pytest.raises(TypeError):
        # Reject no-args constructions, which is unreadable and harder to support long-term
        stub.function(dummy, retries=modal.Retries())  # type: ignore

    # Reject weird inputs:
    # Don't need server to detect and reject nonsensical input. Can do client-side.
    with pytest.raises(InvalidError):
        stub.function(dummy, retries=modal.Retries(max_retries=-2))

    with pytest.raises(InvalidError):
        stub.function(dummy, retries=modal.Retries(max_retries=2, backoff_coefficient=0.0))

    with stub.run(client=client):
        default_retries_from_int_modal()
        fixed_delay_retries_modal()
        exponential_backoff_modal()
        exponential_with_max_delay_modal()
