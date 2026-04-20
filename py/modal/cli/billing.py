# Copyright Modal Labs 2025
from datetime import datetime
from json import dumps
from typing import Optional

import click

from modal._billing import _workspace_billing_report
from modal._utils.async_utils import synchronizer
from modal._utils.time_utils import parse_date, parse_date_range, resolve_timezone

from ._help import ModalGroup
from .utils import display_table

billing_cli = ModalGroup(name="billing", help="View workspace billing information.")


DATE_HELP = "Date (in UTC by default): ISO format (2025-01-01) or relative (yesterday, 3 days ago, etc.)."


@billing_cli.command("report", no_args_is_help=True)
@click.option("--start", default=None, help=f"Start date. {DATE_HELP}")
@click.option("--end", default=None, help=f"End date. {DATE_HELP} Defaults to now.")
@click.option(
    "--for",
    "for_",
    default=None,
    help="Convenience range: today, yesterday, this week, last week, this month, last month.",
)
@click.option("-r", "--resolution", default="d", help="Time resolution: 'd' (daily) or 'h' (hourly).")
@click.option(
    "--tz",
    default=None,
    help="Timezone for date interpretation: 'local', offset (5, -4, +05:30), or IANA name. Requires hourly resolution.",
)
@click.option("-t", "--tag-names", default=None, help="Comma-separated list of tag names to include.")
@click.option("--json", "json", is_flag=True, default=False, help="Output as JSON.")
@click.option("--csv", "csv", is_flag=True, default=False, help="Output as CSV.")
@synchronizer.create_blocking
async def report(
    start: Optional[str],
    end: Optional[str],
    for_: Optional[str],
    resolution: str,
    tz: Optional[str],
    tag_names: Optional[str],
    json: bool,
    csv: bool,
):
    """Generate a billing report for the workspace.

    The report range can be provided by setting `--start` / `--end` dates (`--end` defaults to 'now')
    or by requesting a date range using `--for` (e.g., `--for today`, `--for 'last month'`).

    This command provides a CLI frontend for the
    [`modal.billing.workspace_billing_report`](https://modal.com/docs/reference/modal.billing) API.

    Note that, as with the API, the start date is inclusive and the end date is exclusive.
    Data will be reported for full intervals only. Using `--for` is a convenient way to define a
    complete interval.

    Examples:

    ```bash
    modal billing report --start 2025-12-01 --end 2026-01-01

    modal billing report --for "last month" --tag-names team,project

    modal billing report --for today --resolution h

    modal billing report --for yesterday -r h --tz local

    modal billing report --for "last month" --csv > report.csv

    modal billing report --start 2025-12-01 --json > report.json
    ```

    """
    # Validate mutually exclusive output formats
    if json and csv:
        raise click.UsageError("--json and --csv are mutually exclusive")

    # Resolve timezone if provided
    resolved_tz = None
    if tz is not None:
        try:
            resolved_tz = resolve_timezone(tz)
        except ValueError as exc:
            raise click.UsageError(str(exc))

    # Validate --for vs --start/--end
    if for_ is not None:
        if start is not None or end is not None:
            raise click.UsageError("--for is mutually exclusive with --start and --end")
        try:
            start_dt, end_dt = parse_date_range(for_, tz=resolved_tz)
        except ValueError as exc:
            raise click.UsageError(str(exc))
    elif start is not None:
        try:
            start_dt = parse_date(start, tz=resolved_tz)
            end_dt = parse_date(end, tz=resolved_tz) if end else None
        except (ValueError, OverflowError) as exc:
            raise click.UsageError(str(exc))
    else:
        raise click.UsageError("Either --for or --start is required")

    # Parse tag names
    tags = [t.strip() for t in tag_names.split(",")] if tag_names else None

    # Validate resolution
    if resolution not in ("d", "h"):
        raise click.UsageError("Resolution must be 'd' (daily) or 'h' (hourly)")

    if resolved_tz is not None and resolution != "h":
        raise click.UsageError(
            "--tz requires hourly resolution (--resolution h / -r h). "
            "Daily intervals are UTC-aligned and cannot be shifted to a custom timezone."
        )

    # Fetch data
    rows_data = await _workspace_billing_report(
        start=start_dt,
        end=end_dt,
        resolution=resolution,
        tag_names=tags,
    )

    # Build output columns and rows
    columns = ["Object ID", "Description", "Environment", "Interval Start", "Cost"]
    if tags:
        columns.append("Tags")

    # Format interval_start based on resolution and output format
    def format_interval(dt: datetime) -> str:
        if resolved_tz is not None:
            dt = dt.astimezone(resolved_tz)
        else:
            # Strip UTC tzinfo so isoformat() doesn't append +00:00
            dt = dt.replace(tzinfo=None)
        if json or csv:
            # Full ISO format for machine-readable output
            return dt.isoformat()
        elif resolution == "d":
            # Just the date for daily resolution
            return dt.strftime("%Y-%m-%d")
        else:
            # Date and time without seconds for hourly resolution
            return dt.strftime("%Y-%m-%dT%H:%M")

    rows = []
    for item in rows_data:
        row = [
            item["object_id"],
            item["description"],
            item["environment_name"],
            format_interval(item["interval_start"]),
            str(item["cost"]),
        ]
        if tags:
            row.append(dumps(item["tags"]))
        rows.append(row)

    display_table(columns, rows, json=json, csv=csv)
