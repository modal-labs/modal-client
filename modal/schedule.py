from .proto import api_pb2

from modal.exception import InvalidError


class Schedule:
    """Schedules represent a time frame to repeatedly run a Modal function."""

    def __init__(self, proto_message):
        self.proto_message = proto_message


class Cron(Schedule):
    """Cron jobs are specified using the standard
    [Unix cron tabs](https://crontab.guru/)

    # Usage

    ```python
    import modal

    @modal.function(schedule=modal.Cron("* * * * *"))
    def f():
        print("This function will run every minute")

    modal.Cron("5 4 * * *")  # run at 4:05am every night
    modal.Cron("0 9 * * 4")  # runs every Thursday 9am
    ```
    """

    def __init__(self, cron_string: str):
        cron = api_pb2.Schedule.Cron(cron_string=cron_string)
        super().__init__(api_pb2.Schedule(cron=cron))


class Period(Schedule):
    """Create a schedule that runs every given time interval.

    This behaves similar to the
    [dateutil](https://dateutil.readthedocs.io/en/latest/relativedelta.html)
    package.

    # Usage

    ```python
    import modal

    @modal.function(schedule=modal.Period(days=1))
    def f():
        print("This function will run every day")

    modal.Period(hours=4)    # runs every 4 hours
    modal.Period(minutes=15) # runs every 15 minutes
    modal.Period(seconds=30) # runs every 30 seconds
    ```
    """

    def __init__(self, years=0, months=0, weeks=0, days=0, hours=0, minutes=0, seconds=0, microseconds=0):
        period = api_pb2.Schedule.Period(
            years=years,
            months=months,
            weeks=weeks,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds,
        )
        super().__init__(api_pb2.Schedule(period=period))
