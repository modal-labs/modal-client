import enum

from .proto import api_pb2


class RateLimitInterval(enum.Enum):
    SECOND = "second"
    MINUTE = "minute"


class RateLimit:
    """Add a rate limit to a modal function.

    # Usage

    ```python
    import modal

    # runs at most twice a second.
    @modal.function(rate_limit=modal.RateLimit(2, modal.RateLimitInterval.SECOND))
    def f():
        pass

    # runs at most once a minute.
    @modal.function(rate_limit=modal.RateLimit(1, modal.RateLimitInterval.MINUTE))
    def f():
        pass
    ```
    """

    def __init__(self, limit: int, interval: RateLimitInterval):
        self.limit = limit
        self.interval = interval

    def to_proto(self):
        if self.interval == RateLimitInterval.SECOND:
            return api_pb2.RateLimit(limit=self.limit, interval=api_pb2.RATE_LIMIT_INTERVAL_SECOND)
        elif self.interval == RateLimitInterval.MINUTE:
            return api_pb2.RateLimit(limit=self.limit, interval=api_pb2.RATE_LIMIT_INTERVAL_MINUTE)
