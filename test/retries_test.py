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


def zero_retries():
    pass


def test_retries(client):
    stub = modal.Stub()

    default_retries_from_int_modal = stub.function(retries=5)(default_retries_from_int)
    fixed_delay_retries_modal = stub.function(retries=modal.Retries(max_retries=5, backoff_coefficient=1.0))(
        fixed_delay_retries
    )

    exponential_backoff_modal = stub.function(
        retries=modal.Retries(max_retries=2, initial_delay=2.0, backoff_coefficient=2.0)
    )(exponential_backoff)

    exponential_with_max_delay_modal = stub.function(
        retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, max_delay=30.0)
    )(exponential_with_max_delay)

    zero_retries_modal = stub.function(retries=0)(zero_retries)

    with pytest.raises(TypeError):
        # Reject no-args constructions, which is unreadable and harder to support long-term
        stub.function(retries=modal.Retries())(dummy)  # type: ignore

    # Reject weird inputs:
    # Don't need server to detect and reject nonsensical input. Can do client-side.
    with pytest.raises(InvalidError):
        stub.function(retries=modal.Retries(max_retries=-2))(dummy)

    with pytest.raises(InvalidError):
        stub.function(retries=modal.Retries(max_retries=2, backoff_coefficient=0.0))(dummy)

    with stub.run(client=client):
        default_retries_from_int_modal.remote()
        fixed_delay_retries_modal.remote()
        exponential_backoff_modal.remote()
        exponential_with_max_delay_modal.remote()
        zero_retries_modal.remote()
