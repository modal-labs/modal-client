from datetime import datetime


def timestamp_to_local(ts: float) -> str:
    locale_tz = datetime.now().astimezone().tzinfo
    dt = datetime.fromtimestamp(ts, tz=locale_tz)
    return dt.isoformat(sep=" ", timespec="seconds")
