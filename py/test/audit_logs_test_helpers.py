# Copyright Modal Labs 2026
import pytest
from typing import Any

from grpclib import GRPCError, Status

from modal._audit_logs import _rpcs
from modal._utils.grpc_utils import Retry
from modal_proto import api_pb2

RANGE: dict[str, Any] = {"q": "", "oldest_ms": 1000, "newest_ms": 9000}
RATE_LIMIT_ERROR = GRPCError(
    Status.RESOURCE_EXHAUSTED,
    "AuditLogCount rate limit exceeded (5/s sustained, burst 10). Please wait and retry.",
    details=[api_pb2.RPCRetryPolicy(retry_after_secs=0.1)],
)
DEADLINE_ERROR = GRPCError(Status.DEADLINE_EXCEEDED, "Deadline Exceeded")


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch):
    monkeypatch.setattr(_rpcs, "Retry", lambda **kwargs: Retry(**{**kwargs, "base_delay": 0, "max_delay": 0}))


def make_settings(**overrides: Any) -> api_pb2.AuditLogSettingsGetResponse:
    fields: dict[str, Any] = dict(
        bucket_size=5000,
        page_size=5000,
        max_concurrent_requests=5,
        max_buffered_buckets=10,
        request_timeout_secs=30.0,
        plan_timeout_secs=60.0,
    )
    return api_pb2.AuditLogSettingsGetResponse(**{**fields, **overrides})
