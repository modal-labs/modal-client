# Copyright Modal Labs 2025
from datetime import datetime
from typing import Optional


def timestamp_to_localized_dt(ts: float) -> datetime:
    locale_tz = datetime.now().astimezone().tzinfo
    return datetime.fromtimestamp(ts, tz=locale_tz)


def timestamp_to_localized_str(ts: float, isotz: bool = True) -> Optional[str]:
    if ts > 0:
        dt = timestamp_to_localized_dt(ts)
        if isotz:
            return dt.isoformat(sep=" ", timespec="seconds")
        else:
            return f"{dt:%Y-%m-%d %H:%M %Z}"
    else:
        return None
