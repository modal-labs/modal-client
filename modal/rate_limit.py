from typing import Optional

from .exception import InvalidError
from .proto import api_pb2


class RateLimit:
    """Add a rate limit to a modal function.

    # Usage

    ```python
    import modal

    # runs at most twice a second.
    @modal.function(rate_limit=modal.RateLimit(per_second=2))
    def f():
        pass

    # runs at most once a minute.
    @modal.function(rate_limit=modal.RateLimit(per_minute=1))
    def f():
        pass
    ```
    """

    def __init__(self, *, per_second: Optional[int] = None, per_minute: Optional[int] = None):
        if per_second is not None and per_minute is not None:
            raise InvalidError("Must specify excatly one of per_second and per_minute.")

        if per_second is None and per_minute is None:
            raise InvalidError("Must specify exactly one of per_second and per_minute.")

        self.per_second = per_second
        self.per_minute = per_minute

    def to_proto(self):
        if self.per_second:
            return api_pb2.RateLimit(limit=self.per_second, interval=api_pb2.RATE_LIMIT_INTERVAL_SECOND)
        else:
            return api_pb2.RateLimit(limit=self.per_minute, interval=api_pb2.RATE_LIMIT_INTERVAL_MINUTE)
