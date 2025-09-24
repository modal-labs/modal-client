# TODO "billing" vs "usage"
from datetime import datetime
from typing import Any, AsyncIterator, Literal, Optional

from modal._utils.async_utils import synchronizer
from modal.client import _Client
from modal.exception import InvalidError
from modal_proto import api_pb2


@synchronizer.create_blocking
async def generate_billing_report(
    *,
    # TODO defaults for start_date / resolution so you don't need to compute them?
    start_date: datetime,
    resolution: Literal["h", "d"],
    tag_names: Optional[list[str]] = None,
    client: Optional[_Client] = None,
) -> AsyncIterator[dict[str, Any]]:  # TODO can we improve return typing?
    # TODO client-side validation for start_date or only do that server-side?

    if resolution == "h":
        bucket_secs = 3600
    elif resolution == "d":
        bucket_secs = 86400
    else:
        raise InvalidError(f"Resolution must be either 'h' or 'd', not {resolution!r}")

    # TODO validate tag_names

    # TODO what parameterization for start / end?
    req = api_pb2.BillingReportRequest(
        bucket_secs=bucket_secs,
        tag_names=tag_names or [],
    )
    req.start_timestamp.FromDatetime(start_date)  # TODO handle timezone stuff?

    if client is None:
        client = await _Client.from_env()

    async for resp in client.stub.BillingReport.unary_stream(req):
        yield {
            "object_id": resp.object_id,
            "description": resp.description,
            "environment": resp.environment_name,  # TODO just "environment"?
            "timestamp": resp.timestamp.ToDatetime(),  # TODO timezone?  # TODO maybe name this "interval"?
            "cost": resp.cost,  # TODO as decimal? how to handle <$0.01 rounding?
            # TODO do we need to prevent tags from colliding with other keys
            # or alternatively should we "namespace" them (e.g. `tag:<name>`)
            **resp.tags,
        }
