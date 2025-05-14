from datetime import datetime


def timestamp_to_local(ts: float, isotz: bool = True) -> str:
    if ts > 0:
        locale_tz = datetime.now().astimezone().tzinfo
        dt = datetime.fromtimestamp(ts, tz=locale_tz)
        if isotz:
            return dt.isoformat(sep=" ", timespec="seconds")
        else:
            return f"{datetime.strftime(dt, '%Y-%m-%d %H:%M')} {locale_tz.tzname(dt)}"
    else:
        return None
