# Copyright Modal Labs 2025
from dataclasses import dataclass
from datetime import datetime, tzinfo
from json import dumps

import click
from rich.box import SIMPLE_HEAD
from rich.table import Table
from rich.text import Text

from modal._utils.async_utils import synchronizer
from modal._utils.display_utils import pretty_decimal
from modal._utils.time_utils import (
    format_interval,
    parse_date,
    parse_date_range,
    resolve_timezone,
)
from modal._workspace import _Workspace
from modal.exception import InvalidError
from modal.output import OutputManager

from ._help import ModalGroup
from .utils import _col_name_to_json_key, display_table

billing_cli = ModalGroup(name="billing", help="View workspace billing information.")


DATE_HELP = "Date (in UTC by default): ISO format (2025-01-01) or relative (yesterday, 3 days ago, etc.)."
ISO_DATE_HELP = "Date (in UTC by default) in ISO format (2025-01-01)."


@dataclass(slots=True)
class _ParsedInterval:
    start: datetime
    end: datetime | None
    tz: tzinfo | None


def _validate_and_parse_interval(
    *,
    start: str | None = None,
    end: str | None = None,
    for_: str | None = None,
    tz: str | None = None,
    resolution: str | None = None,
) -> _ParsedInterval:
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
            start_dt, end_dt = parse_date_range(
                for_,
                tz=resolved_tz,
            )
        except ValueError as exc:
            raise click.UsageError(str(exc))
    elif start is not None:
        try:
            start_dt = parse_date(start, tz=resolved_tz)
            end_dt = parse_date(end, tz=resolved_tz) if end else None
        except (ValueError, OverflowError) as exc:
            raise click.UsageError(str(exc))
    elif end is not None:
        raise click.UsageError("Must provide --start if providing --end")
    else:
        raise click.UsageError("Either --for or --start is required")

    # Validate resolution
    if resolution is not None:
        if resolution not in ("d", "h"):
            raise click.UsageError("Resolution must be 'd' (daily) or 'h' (hourly)")

        if resolved_tz is not None and resolution != "h":
            raise click.UsageError(
                "--tz requires hourly resolution (--resolution h / -r h). "
                "Daily intervals are UTC-aligned and cannot be shifted to a custom timezone."
            )

    return _ParsedInterval(start_dt, end_dt, resolved_tz)


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
@click.option("--show-resources", is_flag=True, default=False, help="Further break down usage by resource type.")
@click.option("--json", "json", is_flag=True, default=False, help="Output as JSON.")
@click.option("--csv", "csv", is_flag=True, default=False, help="Output as CSV.")
@synchronizer.create_blocking
async def report(
    start: str | None,
    end: str | None,
    for_: str | None,
    resolution: str,
    tz: str | None,
    tag_names: str | None,
    show_resources: bool,
    json: bool,
    csv: bool,
):
    """Generate a billing report for the workspace.

    The report range can be provided by setting `--start` / `--end` dates (`--end` defaults to 'now')
    or by requesting a date range using `--for` (e.g., `--for today`, `--for 'last month'`).

    This command provides a CLI frontend for the
    [`Workspace.billing.report`](https://modal.com/docs/sdk/py/latest/Workspace#billingreport) API.

    Note that, as with the API, the start date is inclusive and the end date is exclusive.
    Data will be reported for full intervals only. Using `--for` is a convenient way to define a
    complete interval.

    In addition, the `--show-resources` option further breaks the cost in each bucket by the resource
    that generated it (CPU, Memory, specific GPU types, etc.). Note that the specific resource types
    included in the report are subject to change as Modal's billing model evolves.

    Examples:

    ```bash
    modal billing report --start 2025-12-01 --end 2026-01-01

    modal billing report --for "last month" --tag-names team,project

    modal billing report --for today --resolution h

    modal billing report --for "this month" --show-resources

    modal billing report --for yesterday -r h --tz local

    modal billing report --for "last month" --csv > report.csv

    modal billing report --start 2025-12-01 --json > report.json
    ```

    """
    # Validate mutually exclusive output formats
    if json and csv:
        raise click.UsageError("--json and --csv are mutually exclusive")

    # Parse tag names
    tags = [t.strip() for t in tag_names.split(",")] if tag_names else None

    interval = _validate_and_parse_interval(start=start, end=end, for_=for_, tz=tz, resolution=resolution)

    # Fetch data
    rows_data = await _Workspace.from_context().billing.report(
        start=interval.start,
        end=interval.end,
        resolution=resolution,
        tag_names=tags,
    )

    # Build output columns and rows
    columns = [
        "Object ID",
        "Description",
        "Environment",
        "Interval Start",
        *(["Resource"] if show_resources else []),
        "Cost",
        *(["Tags"] if tags else []),
    ]

    rows = []
    for item in rows_data:
        row_base = [
            item.object_id,
            item.description,
            item.environment_name,
            format_interval(
                item.interval_start,
                tz=interval.tz,
                isoformat=json or csv,
                show_date_only=resolution == "d",
            ),
        ]

        row_set: list[list[str]] = []
        if show_resources:
            for resource, cost in item.cost_by_resource.items():
                row_set.append([*row_base, resource, str(cost)])
        else:
            row_set.append([*row_base, str(item.cost)])

        if tags:
            for row in row_set:
                row.append(dumps(item.tags))

        rows.extend(row_set)

    display_table(columns, rows, json=json, csv=csv)


# todo(ayush): mirror this in `modal workspace` and then deprecate this one
@billing_cli.command("summary")
@click.option(
    "--for",
    "for_",
    default=None,
    type=str,
    help=('What cycle to show a summary for. Accepts: "this month", "last month", and ISO 8601 months ("YYYY-MM").'),
)
@click.option("--json", "json", is_flag=True, default=False, help="Output as JSON.")
@synchronizer.create_blocking
async def summary(for_: str | None, json: bool):
    """Generate a billing summary for the workspace.

    The summary range can be provided by setting `--for` (e.g `--for 'last month'`). If not
    provided, `--for` defaults to "this month".

    Summaries are provided for single month intervals (aligned to the month boundary) only. To see
    summaries for longer intervals, call `summary` for each month in the interval.

    This command provides a CLI frontend for the
    [`Workspace.billing.summary`](https://modal.com/docs/sdk/py/latest/Workspace#billingsummary) API.

    Examples:

    ```bash
    modal billing summary # defaults to --for "this month"

    modal billing summary --for "last month"

    modal billing summary --for 2026-01
    ```

    """
    # If called with no arguments, we default to the current billing cycle
    if for_ is None:
        for_ = "this month"

    try:
        summary = await _Workspace.from_context().billing.summary(cycle=for_)
    except (ValueError, InvalidError) as exc:
        raise click.UsageError(str(exc))

    output = OutputManager.get()
    if json:
        output.print_json(
            dumps(
                {
                    "metered_cost": str(summary.metered_cost),
                    "billed_cost": str(summary.billed_cost),
                    "adjustments": {_col_name_to_json_key(k): str(v) for k, v in summary.adjustments.items()},
                    "metered_cost_breakdown": {
                        _col_name_to_json_key(k): str(v) for k, v in summary.metered_cost_breakdown.items()
                    },
                }
            )
        )

        return

    t = Table(show_header=False, box=SIMPLE_HEAD)

    t.add_row(Text("Metered Cost:"), Text(pretty_decimal(summary.metered_cost), justify="right"))
    for k, v in sorted(summary.metered_cost_breakdown.items(), key=lambda pair: -pair[1]):
        if v == 0:
            continue

        t.add_row(Text(f"  {k}:", style="dim"), Text(pretty_decimal(v), style="dim", justify="right"))

    for k, v in summary.adjustments.items():
        if v == 0:
            continue

        # note: discounts are sent by the server as negative values, additional costs are sent as
        # positive values
        t.add_row(Text(f"{k}:"), Text(pretty_decimal(v), justify="right"))

    t.add_row(
        Text("Billed Cost:", style="bold"),
        Text(f"${pretty_decimal(summary.billed_cost)}", justify="right", style="bold"),
    )

    output.print(t)
