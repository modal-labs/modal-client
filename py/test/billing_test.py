# Copyright Modal Labs 2025
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import modal.billing
from modal.exception import InvalidError


def test_workspace_billing_report(servicer, client):
    before_request = datetime.now(timezone.utc)
    with servicer.intercept() as ctx:
        report = modal.billing.workspace_billing_report(
            start=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            tag_names=["team", "project"],
            client=client,
        )
        assert len(ctx.get_requests("WorkspaceBillingReport")) == 1

    request = ctx.pop_request("WorkspaceBillingReport")
    assert request.end_timestamp.ToDatetime().replace(tzinfo=timezone.utc) >= before_request

    item = report[0]
    assert isinstance(item["tags"], dict)
    assert isinstance(item["interval_start"], datetime)
    assert item["interval_start"].tzinfo == timezone.utc
    assert isinstance(item["cost"], Decimal)

    with pytest.raises(InvalidError, match="'start' parameter must be in UTC"):
        modal.billing.workspace_billing_report(
            start=datetime(2025, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            client=client,
        )

    with pytest.raises(InvalidError, match="'end' parameter must be in UTC"):
        modal.billing.workspace_billing_report(
            start=datetime(2025, 1, 1, 0, 0, 0),
            end=datetime(2025, 1, 2, 0, 0, 0, tzinfo=ZoneInfo("America/New_York")),
            client=client,
        )
