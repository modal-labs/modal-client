# Copyright Modal Labs 2025
import pytest

from grpclib import GRPCError, Status

import modal
from modal._grpc_client import _STATUS_TO_EXCEPTION


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
