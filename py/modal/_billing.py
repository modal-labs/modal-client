# Copyright Modal Labs 2025
from datetime import datetime
from decimal import Decimal
from typing import TypedDict

from ._utils.deprecation import deprecation_warning
from .client import _Client


class WorkspaceBillingReportItem(TypedDict):
    object_id: str
    description: str
    environment_name: str
    interval_start: datetime
    cost: Decimal
    tags: dict[str, str]


async def _workspace_billing_report(
    *,
    start: datetime,  # Start of the report, inclusive
    end: datetime | None = None,  # End of the report, exclusive
    resolution: str = "d",  # Resolution, e.g. "d" for daily or "h" for hourly
    tag_names: list[str] | None = None,  # Optional additional metadata to include
    client: _Client | None = None,
) -> list[WorkspaceBillingReportItem]:
    """Generate a tabular report of workspace usage by object and time.

    The result will be a list of dictionaries for each interval (determined by `resolution`)
    between the `start` and `end` limits. The dictionary represents a single Modal object
    that billing can be attributed to (e.g., an App) along with metadata (including user-defined
    tags) for identifying that object. The dictionary also contains a breakdown of the cost value
    attributed to individual resources (for an App, this can be CPU, Memory, specific GPU types,
    etc.). The specific resource types included in the breakdown are subject to change as
    Modal's billing model evolves.

    The `start` and `end` parameters are required to either have a UTC timezone or to be
    timezone-naive (which will be interpreted as UTC times). The timestamps in the result will
    be in UTC. Cost will be reported for full intervals, even if the provided `start` or `end`
    parameters are partial: `start` will be rounded to the beginning of its interval, while
    partial `end` intervals will be excluded.

    Additional user-provided metadata can be included in the report if the objects have tags
    and `tag_names` (i.e., keys) are specified in the request. Alternatively, pass `tag_names=["*"]`
    to include all tags in the report. Note that tags will be attributed to the entire interval even
    if they were added or removed at some point within it. If the tag name was not in use during an
    interval, it will be absent from the tags dictionary in that output row.

    It's also possible to generate reports using the
    [`modal billing report`](https://modal.com/docs/cli/latest/billing) CLI command. The CLI
    has a few convenience features for generating reports across relative time ranges.

    """

    deprecation_warning(
        (2026, 6, 18),
        "The `modal.billing.workspace_billing_report()` function is deprecated. "
        "Use the `workspace.billing.report()` method on the `modal.Workspace` object instead.",
    )

    from ._workspace import _Workspace

    data = await _Workspace.from_context(client=client).billing.report(
        start=start,
        end=end,
        resolution=resolution,
        tag_names=tag_names,
    )

    res: list[WorkspaceBillingReportItem] = []
    for datum in data:
        item: WorkspaceBillingReportItem = {
            "object_id": datum.object_id,
            "description": datum.description,
            "environment_name": datum.environment_name,
            "interval_start": datum.interval_start,
            "cost": Decimal(datum.cost),
            "tags": dict(datum.tags),
        }

        res.append(item)

    return res
