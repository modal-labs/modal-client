# Copyright Modal Labs 2025
import pytest

from grpclib import GRPCError, Status

import modal
from modal._grpc_client import _STATUS_TO_EXCEPTION
from modal._utils.grpc_utils import RETRYABLE_GRPC_STATUS_CODES


def test_exception_map():
    assert {Status.OK, *_STATUS_TO_EXCEPTION} == set(Status)


@pytest.mark.parametrize("exception_type", _STATUS_TO_EXCEPTION.values())
def test_exception_inheritance(exception_type):
    assert issubclass(exception_type, modal.Error)
    assert issubclass(exception_type, GRPCError)


@pytest.mark.parametrize("exception_type", _STATUS_TO_EXCEPTION.values())
def test_exception_message(exception_type):
    exc = exception_type("oh no!")
    assert repr(exc) == f"{exception_type.__name__}('oh no!')"
    assert str(exc) == "oh no!"


def test_retryable_exceptions():
    retryable_exceptions = {
        exception_type
        for status_code, exception_type in _STATUS_TO_EXCEPTION.items()
        if status_code in RETRYABLE_GRPC_STATUS_CODES
    }
    assert retryable_exceptions == {modal.exception.ServiceError, modal.exception.InternalError}
