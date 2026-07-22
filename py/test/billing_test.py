# Copyright Modal Labs 2025
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import modal.billing
from modal.environments import Environment
from modal.exception import DeprecationError, InvalidError
from modal.workspace import Workspace
from modal_proto import api_pb2


def test_workspace_billing_report(servicer, client):
    before_request = datetime.now(timezone.utc)
    with servicer.intercept() as ctx:
        with pytest.raises(DeprecationError):
            modal.billing.workspace_billing_report(
                start=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                tag_names=["team", "project"],
                client=client,
            )

        ws = Workspace.from_context(client=client)

        report = ws.billing.report(
            start=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            tag_names=["team", "project"],
        )

        assert len(ctx.get_requests("WorkspaceBillingReport")) == 1

    request: api_pb2.WorkspaceBillingReportRequest = ctx.pop_request("WorkspaceBillingReport")
    assert request.end_timestamp.ToDatetime().replace(tzinfo=timezone.utc) >= before_request

    for item in report:
        assert isinstance(item["tags"], dict)
        assert isinstance(item["interval_start"], datetime)
        assert item["interval_start"].tzinfo == timezone.utc
        assert isinstance(item["cost"], Decimal)

    with pytest.raises(InvalidError, match="'start' parameter must be in UTC"):
        ws.billing.report(
            start=datetime(2025, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        )

    with pytest.raises(InvalidError, match="'end' parameter must be in UTC"):
        ws.billing.report(
            start=datetime(2025, 1, 1, 0, 0, 0),
            end=datetime(2025, 1, 2, 0, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        )


def test_workspace_object_billing_report(servicer, client):
    before_request = datetime.now(timezone.utc)
    with servicer.intercept() as ctx:
        ws = Workspace.from_context(client=client)
        report = ws.billing.report(
            start=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            tag_names=["team", "project"],
        )

        assert len(ctx.get_requests("WorkspaceBillingReport")) == 1

    request: api_pb2.WorkspaceBillingReportRequest = ctx.pop_request("WorkspaceBillingReport")
    assert request.end_timestamp.ToDatetime().replace(tzinfo=timezone.utc) >= before_request

    assert len(report) == 2
    for item in report:
        assert isinstance(item.tags, dict)
        assert isinstance(item.interval_start, datetime)
        assert item.interval_start.tzinfo == timezone.utc
        assert isinstance(item.cost, Decimal)
        assert isinstance(item.cost_by_resource, dict)


def test_workspace_object_billing_summary(servicer, client):
    with servicer.intercept() as ctx:
        ws = Workspace.from_context(client=client)
        summary = ws.billing.summary(cycle=datetime(2025, 1, 1, tzinfo=timezone.utc))

        assert len(ctx.get_requests("WorkspaceBillingSummary")) == 1

    request: api_pb2.WorkspaceBillingSummaryRequest = ctx.pop_request("WorkspaceBillingSummary")
    start = request.start_timestamp.ToDatetime(timezone.utc)

    assert start.day == 1
    assert start.hour == 0
    assert start.minute == 0
    assert start.second == 0
    assert start.microsecond == 0

    assert isinstance(summary.metered_cost, Decimal)
    assert isinstance(summary.billed_cost, Decimal)
    assert isinstance(summary.adjustments, dict)
    assert isinstance(summary.metered_cost_breakdown, dict)


def test_environment_object_billing_report(servicer, client):
    before_request = datetime.now(timezone.utc)
    with servicer.intercept() as ctx:
        env = Environment.from_name("main", client=client)
        report = env.billing.report(
            start=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            tag_names=["team", "project"],
        )
        assert len(ctx.get_requests("WorkspaceBillingReport")) == 1

        # need to hydrate to resolve id
        assert len(ctx.get_requests("EnvironmentGetOrCreate")) == 1

    request: api_pb2.WorkspaceBillingReportRequest = ctx.pop_request("WorkspaceBillingReport")
    assert request.end_timestamp.ToDatetime().replace(tzinfo=timezone.utc) >= before_request

    assert len(report) == 1

    item = report[0]
    assert isinstance(item.tags, dict)
    assert isinstance(item.interval_start, datetime)
    assert item.interval_start.tzinfo == timezone.utc
    assert isinstance(item.cost, Decimal)
    assert isinstance(item.cost_by_resource, dict)

    assert item.environment_name == "main"


def test_environment_object_billing_summary(servicer, client):
    with servicer.intercept() as ctx:
        env = Environment.from_name("main", client=client)
        summary = env.billing.summary(cycle=datetime(2025, 1, 1, tzinfo=timezone.utc))

        assert len(ctx.get_requests("EnvironmentBillingSummary")) == 1

        # need to hydrate to resolve id
        assert len(ctx.get_requests("EnvironmentGetOrCreate")) == 1

    request: api_pb2.EnvironmentBillingSummaryRequest = ctx.pop_request("EnvironmentBillingSummary")
    start = request.start_timestamp.ToDatetime(timezone.utc)

    assert start.day == 1
    assert start.hour == 0
    assert start.minute == 0
    assert start.second == 0
    assert start.microsecond == 0

    assert isinstance(summary.metered_cost, Decimal)
    assert isinstance(summary.metered_cost_breakdown, dict)
