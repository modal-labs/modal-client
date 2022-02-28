import abc

import dateutil.relativedelta

from modal.exception import InvalidError


class Schedule:
    """Schedules represent a time frame to repeatedly run a Modal function."""

    @abc.abstractmethod
    def protobuf_representation(self):
        pass


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
        self._cron_string = cron_string


class Period(Schedule):
    """Create a schedule that runs every given time interval.

    This is based on the
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

    # TODO: figure out if we can copy the argument list from relativedelta
    def __init__(self, **kwargs):
        self._delta = dateutil.relativedelta.relativedelta(**kwargs)
