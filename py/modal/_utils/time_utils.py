# Copyright Modal Labs 2025
import re
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Optional, Union
from zoneinfo import ZoneInfo


def resolve_timezone(s: str) -> tzinfo:
    """Resolve a timezone string to a tzinfo object.

    Accepted values:
    - "local": system timezone via locale_tz()
    - Integer string (e.g. "5", "-4"): UTC offset in whole hours
    - Offset string (e.g. "+05:30", "-03:00"): UTC offset with minutes
    - IANA name (e.g. "America/New_York"): resolved via ZoneInfo

    Raises ValueError for unrecognized input.
    """
    s = s.strip()
    if s.lower() == "local":
        return locale_tz()

    # Integer offset (whole hours)
    if re.match(r"^-?\d+$", s):
        return timezone(timedelta(hours=int(s)))

    # Offset string like +05:30 or -03:00
    m = re.match(r"^([+-])(\d{2}):(\d{2})$", s)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        hours, minutes = int(m.group(2)), int(m.group(3))
        return timezone(timedelta(hours=sign * hours, minutes=sign * minutes))

    # IANA timezone name
    try:
        return ZoneInfo(s)
    except Exception:
        raise ValueError(
            f"Unknown timezone: '{s}'. Use 'local', an integer offset (e.g. 5, -4), "
            f"an offset string (e.g. +05:30), or an IANA name (e.g. America/New_York)."
        )


def parse_date(s: str, tz: Optional[tzinfo] = None) -> datetime:
    """Parse a date string, supporting both ISO format and relative dates.

    Supported formats:
    - ISO format: 2025-01-01, 2025-01-01T00:00:00
    - Relative: now, today, yesterday, N days ago, N weeks ago, N months ago, N hours ago

    When `tz` is provided, date-like values (today, yesterday, ISO dates) are
    interpreted as midnight in that timezone, then converted to UTC.
    Relative offsets (N days/hours ago, now) remain UTC-relative.

    Returns a datetime in UTC.
    Raises ValueError if the format is not recognized.
    """
    s_orig = s.strip()
    s = s_orig.lower()
    now = datetime.now(timezone.utc)

    if s == "now":
        return now

    if s == "today":
        if tz is not None:
            today_local = now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
            return today_local.astimezone(timezone.utc)
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    if s == "yesterday":
        if tz is not None:
            today_local = now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_local = today_local - timedelta(days=1)
            return yesterday_local.astimezone(timezone.utc)
        return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    # "N days/weeks/months/hours ago"
    match = re.fullmatch(r"(\d+)\s+(day|week|month|hour)s?\s+ago", s)
    if match:
        n, unit = int(match.group(1)), match.group(2)
        if unit == "day":
            return now - timedelta(days=n)
        elif unit == "week":
            return now - timedelta(weeks=n)
        elif unit == "hour":
            return now - timedelta(hours=n)
        elif unit == "month":
            # Approximate months as 30 days
            return now - timedelta(days=n * 30)

    # Fall back to ISO parsing
    try:
        dt = datetime.fromisoformat(s_orig)
    except ValueError:
        raise ValueError(
            f"Invalid date format: '{s}'. Use ISO format (2025-01-01) or relative (yesterday, 7 days ago)."
        )

    if dt.tzinfo is not None:
        # Already has timezone info — convert to UTC
        return dt.astimezone(timezone.utc)
    if tz is not None:
        # Interpret as local time in the given timezone, then convert to UTC
        return dt.replace(tzinfo=tz).astimezone(timezone.utc)
    return dt.replace(tzinfo=timezone.utc)


def parse_date_range(s: str, tz: Optional[tzinfo] = None) -> tuple[datetime, datetime]:
    """Parse a convenience range string into a (start, end) pair of UTC datetimes.

    Accepted values:
        today, yesterday, this week, last week, this month, last month

    When `tz` is provided, boundaries are computed in that timezone then
    converted to UTC. Otherwise, boundaries are midnight UTC.

    Weeks start on Monday (ISO 8601).
    Raises ValueError for unrecognized input.
    """
    s = s.strip().lower()
    now = datetime.now(timezone.utc)

    if tz is not None:
        today = now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def _to_utc(dt: datetime) -> datetime:
        return dt.astimezone(timezone.utc)

    if s == "today":
        return _to_utc(today), _to_utc(today + timedelta(days=1))

    if s == "yesterday":
        yesterday = today - timedelta(days=1)
        return _to_utc(yesterday), _to_utc(today)

    if s == "this week":
        monday = today - timedelta(days=today.weekday())
        return _to_utc(monday), _to_utc(monday + timedelta(weeks=1))

    if s == "last week":
        monday = today - timedelta(days=today.weekday())
        return _to_utc(monday - timedelta(weeks=1)), _to_utc(monday)

    if s == "this month":
        first = today.replace(day=1)
        if today.month == 12:
            next_first = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_first = today.replace(month=today.month + 1, day=1)
        return _to_utc(first), _to_utc(next_first)

    if s == "last month":
        this_first = today.replace(day=1)
        if today.month == 1:
            last_first = today.replace(year=today.year - 1, month=12, day=1)
        else:
            last_first = today.replace(month=today.month - 1, day=1)
        return _to_utc(last_first), _to_utc(this_first)

    accepted = "today, yesterday, this week, last week, this month, last month"
    raise ValueError(f"Unrecognized range: '{s}'. Accepted values: {accepted}")


def relative_timestamp(dt: datetime) -> str:
    """Convert a tz-aware datetime to a human-readable relative time string.

    Examples: "just now", "30 seconds ago", "5 minutes ago", "2 hours ago",
    "yesterday", "3 days ago", "2 weeks ago", "3 months ago", "1 year ago".

    Raises ValueError if the datetime is naive (no tzinfo).
    """
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")

    now = datetime.now(timezone.utc)
    delta = now - dt
    total_seconds = int(delta.total_seconds())

    if total_seconds < 0:
        return "just now"

    if total_seconds < 10:
        return "just now"
    if total_seconds < 60:
        return f"{total_seconds} seconds ago"
    if total_seconds < 120:
        return "1 minute ago"

    minutes = total_seconds // 60
    if minutes < 60:
        return f"{minutes} minutes ago"
    if minutes < 120:
        return "1 hour ago"

    hours = minutes // 60
    if hours < 24:
        return f"{hours} hours ago"
    if hours < 48:
        return "yesterday"

    days = hours // 24
    if days < 14:
        return f"{days} days ago"

    weeks = days // 7
    if days < 60:
        return f"{weeks} weeks ago"

    months = days // 30
    if days < 365:
        return f"{months} months ago"

    years = days // 365
    if years == 1:
        return "1 year ago"
    return f"{years} years ago"


def locale_tz() -> tzinfo:
    return datetime.now().astimezone().tzinfo


def as_timestamp(arg: Optional[Union[datetime, str]]) -> float:
    """Coerce a user-provided argument to a timestamp.

    An argument provided without timezone information will be treated as local time.

    When the argument is null, returns the current time.
    """
    if arg is None:
        dt = datetime.now().astimezone()
    elif isinstance(arg, str):
        dt = datetime.fromisoformat(arg)
    elif isinstance(arg, datetime):
        dt = arg
    else:
        raise TypeError(f"Invalid argument: {arg}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=locale_tz())
    return dt.timestamp()


def timestamp_to_localized_dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=locale_tz())


def timestamp_to_localized_str(ts: float, isotz: bool = True) -> Optional[str]:
    if ts > 0:
        dt = timestamp_to_localized_dt(ts)
        if isotz:
            return dt.isoformat(sep=" ", timespec="seconds")
        else:
            return f"{dt:%Y-%m-%d %H:%M %Z}"
    else:
        return None
