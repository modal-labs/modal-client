# Copyright Modal Labs 2025
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

from modal._utils.time_utils import parse_date, parse_date_range, resolve_timezone

FIXED_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_now():
    with patch("modal._utils.time_utils.datetime") as mock_datetime:
        mock_datetime.now.return_value = FIXED_NOW
        mock_datetime.fromisoformat = datetime.fromisoformat
        yield


@pytest.mark.parametrize(
    "input,expected",
    [
        ("today", FIXED_NOW.replace(hour=0, minute=0, second=0, microsecond=0)),
        ("yesterday", (FIXED_NOW - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)),
        ("7 days ago", FIXED_NOW - timedelta(days=7)),
        ("1 day ago", FIXED_NOW - timedelta(days=1)),
        ("2 weeks ago", FIXED_NOW - timedelta(weeks=2)),
        ("3 hours ago", FIXED_NOW - timedelta(hours=3)),
        ("1 month ago", FIXED_NOW - timedelta(days=30)),
        ("  TODAY  ", FIXED_NOW.replace(hour=0, minute=0, second=0, microsecond=0)),  # whitespace + case
    ],
)
def test_parse_date_relative(mock_now, input, expected):
    assert parse_date(input) == expected


def test_parse_date_now(mock_now):
    # "now" returns the current time, not midnight
    assert parse_date("now") == FIXED_NOW


@pytest.mark.parametrize(
    "input,expected",
    [
        ("2025-01-15", datetime(2025, 1, 15, tzinfo=timezone.utc)),
        ("2025-01-15T14:30:00", datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)),
    ],
)
def test_parse_date_iso(input, expected):
    assert parse_date(input) == expected


@pytest.mark.parametrize("input", ["not-a-date", "sometime ago"])
def test_parse_date_invalid(input):
    with pytest.raises(ValueError, match="Invalid date format"):
        parse_date(input)


# FIXED_NOW = 2025-01-15 12:00 UTC (Wednesday, weekday=2)
_UTC = timezone.utc


def _utc(y, m, d):
    return datetime(y, m, d, tzinfo=_UTC)


@pytest.mark.parametrize(
    "input,expected_start,expected_end",
    [
        ("today", _utc(2025, 1, 15), _utc(2025, 1, 16)),
        ("yesterday", _utc(2025, 1, 14), _utc(2025, 1, 15)),
        ("this week", _utc(2025, 1, 13), _utc(2025, 1, 20)),
        ("last week", _utc(2025, 1, 6), _utc(2025, 1, 13)),
        ("this month", _utc(2025, 1, 1), _utc(2025, 2, 1)),
        ("last month", _utc(2024, 12, 1), _utc(2025, 1, 1)),
    ],
)
def test_parse_date_range(mock_now, input, expected_start, expected_end):
    start, end = parse_date_range(input)
    assert start == expected_start
    assert end == expected_end


def test_parse_date_range_invalid(mock_now):
    with pytest.raises(ValueError, match="Unrecognized range"):
        parse_date_range("next year")


@pytest.mark.parametrize("input", ["  TODAY  ", "Today", " THIS WEEK ", "LAST month"])
def test_parse_date_range_case_insensitive(mock_now, input):
    start, end = parse_date_range(input)
    assert start < end


# --- resolve_timezone tests ---


def test_resolve_timezone_local():
    tz = resolve_timezone("local")
    # Should return a tzinfo from the system; just check it's usable
    assert tz is not None
    datetime.now(tz)  # should not raise


def test_resolve_timezone_integer_offset():
    tz = resolve_timezone("5")
    assert tz == timezone(timedelta(hours=5))

    tz = resolve_timezone("-4")
    assert tz == timezone(timedelta(hours=-4))

    tz = resolve_timezone("0")
    assert tz == timezone.utc


def test_resolve_timezone_offset_string():
    tz = resolve_timezone("+05:30")
    assert tz == timezone(timedelta(hours=5, minutes=30))

    tz = resolve_timezone("-03:00")
    assert tz == timezone(timedelta(hours=-3))


def test_resolve_timezone_iana():
    tz = resolve_timezone("America/New_York")
    assert isinstance(tz, ZoneInfo)
    assert str(tz) == "America/New_York"


def test_resolve_timezone_invalid():
    with pytest.raises(ValueError, match="Unknown timezone"):
        resolve_timezone("Not/A/Timezone")


# --- parse_date with tz tests ---

# Eastern is UTC-5, so midnight Eastern = 05:00 UTC
_EASTERN = ZoneInfo("America/New_York")


def test_parse_date_iso_with_tz():
    # 2025-01-01 midnight in Eastern (UTC-5) = 2025-01-01 05:00 UTC
    result = parse_date("2025-01-01", tz=_EASTERN)
    assert result == datetime(2025, 1, 1, 5, 0, 0, tzinfo=timezone.utc)


def test_parse_date_today_with_tz(mock_now):
    # FIXED_NOW = 2025-01-15 12:00 UTC
    # In UTC-5, that's 2025-01-15 07:00 Eastern
    # Midnight Eastern on that day = 2025-01-15 05:00 UTC
    result = parse_date("today", tz=_EASTERN)
    assert result == datetime(2025, 1, 15, 5, 0, 0, tzinfo=timezone.utc)


def test_parse_date_yesterday_with_tz(mock_now):
    # FIXED_NOW = 2025-01-15 12:00 UTC -> 07:00 Eastern
    # Yesterday midnight Eastern = 2025-01-14 05:00 UTC
    result = parse_date("yesterday", tz=_EASTERN)
    assert result == datetime(2025, 1, 14, 5, 0, 0, tzinfo=timezone.utc)


def test_parse_date_now_ignores_tz(mock_now):
    # "now" should always return current UTC time regardless of tz
    result = parse_date("now", tz=_EASTERN)
    assert result == FIXED_NOW


def test_parse_date_relative_ignores_tz(mock_now):
    # Relative offsets remain UTC-relative
    result = parse_date("7 days ago", tz=_EASTERN)
    assert result == FIXED_NOW - timedelta(days=7)


def test_parse_date_iso_with_explicit_tz_in_string():
    # If the ISO string already has timezone info, it should be converted to UTC (not replaced)
    # +05:00 means midnight at UTC+5 = 19:00 previous day in UTC
    result = parse_date("2025-01-01T00:00:00+05:00", tz=_EASTERN)
    assert result == datetime(2024, 12, 31, 19, 0, 0, tzinfo=timezone.utc)

    # +00:00 should remain unchanged
    result = parse_date("2025-01-01T00:00:00+00:00", tz=_EASTERN)
    assert result == datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Without tz param, explicit offset should still be converted
    result = parse_date("2025-01-01T00:00:00+05:00")
    assert result == datetime(2024, 12, 31, 19, 0, 0, tzinfo=timezone.utc)


# --- parse_date_range with tz tests ---


def test_parse_date_range_this_month_with_tz(mock_now):
    # FIXED_NOW = 2025-01-15 12:00 UTC -> 07:00 Eastern on Jan 15
    # "this month" in Eastern: Jan 1 00:00 Eastern -> Feb 1 00:00 Eastern
    # = Jan 1 05:00 UTC -> Feb 1 05:00 UTC
    start, end = parse_date_range("this month", tz=_EASTERN)
    assert start == datetime(2025, 1, 1, 5, 0, 0, tzinfo=timezone.utc)
    assert end == datetime(2025, 2, 1, 5, 0, 0, tzinfo=timezone.utc)


def test_parse_date_range_today_with_tz(mock_now):
    start, end = parse_date_range("today", tz=_EASTERN)
    assert start == datetime(2025, 1, 15, 5, 0, 0, tzinfo=timezone.utc)
    assert end == datetime(2025, 1, 16, 5, 0, 0, tzinfo=timezone.utc)


def test_parse_date_range_without_tz_unchanged(mock_now):
    # Verify default behavior is still UTC
    start, end = parse_date_range("today")
    assert start == _utc(2025, 1, 15)
    assert end == _utc(2025, 1, 16)


def test_parse_date_range_with_fixed_offset(mock_now):
    # UTC+5:30 (India): midnight IST = 18:30 UTC previous day
    ist = timezone(timedelta(hours=5, minutes=30))
    # FIXED_NOW = 2025-01-15 12:00 UTC = 2025-01-15 17:30 IST
    # "today" in IST: Jan 15 00:00 IST = Jan 14 18:30 UTC
    start, end = parse_date_range("today", tz=ist)
    assert start == datetime(2025, 1, 14, 18, 30, 0, tzinfo=timezone.utc)
    assert end == datetime(2025, 1, 15, 18, 30, 0, tzinfo=timezone.utc)
