from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class Schedule:
    """Schedules represent a time frame to repeatedly run a Modal function.

    There are two types of schedules, _cron_ and _period_. Cron jobs are specified
    using a syntax similar to [Unix cron tabs](https://crontab.guru/), while functions
    that have a period schedule are run according to a fixed time interval.
    """

    _cron_string: Optional[str] = None
    _period: Optional[float] = None

    @classmethod
    def cron(cls, cron_string: str):
        """Create a schedule with a specified cron string.

        # Usage

        ```python
        import modal

        modal.Schedule.cron("* * * * *")  # runs every minute
        modal.Schedule.cron("5 4 * * *")  # runs at 4:05
        ```
        """
        return Schedule(_cron_string=cron_string)

    @classmethod
    def period(cls, period: Union[float, str]):
        """Create a schedule that runs every given time interval.

        # Usage

        ```python
        import modal

        modal.Schedule.period("1d")   # runs every day
        modal.Schedule.period("4h")   # runs every 4 hours
        modal.Schedule.period("15m")  # runs every 15 minutes
        modal.Schedule.period("30s")  # runs every 30 seconds

        modal.Schedule.period(0.456)  # runs every 456 ms
        ```
        """
        if isinstance(period, str):
            try:
                if period[-1] == "d":
                    period = int(period[:-1]) * 24 * 3600
                elif period[-1] == "h":
                    period = int(period[:-1]) * 3600
                elif period[-1] == "m":
                    period = int(period[:-1]) * 60
                elif period[-1] == "s":
                    period = int(period[:-1])
                else:
                    raise
            except Exception:
                raise ValueError(f"Failed to parse period while creating Schedule: {period}")
        return Schedule(_period=period)
