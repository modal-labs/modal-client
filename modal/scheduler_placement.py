# Copyright Modal Labs 2024
from typing import Optional

from modal_proto import api_pb2


class SchedulerPlacement:
    """mdmd:hidden This is an experimental feature."""

    proto: api_pb2.SchedulerPlacement

    def __init__(
        self,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        spot: Optional[bool] = None,
    ):
        """mdmd:hidden"""
        _lifecycle: Optional[str] = None
        if spot is not None:
            _lifecycle = "spot" if spot else "on-demand"

        self.proto = api_pb2.SchedulerPlacement(
            _region=region,
            _zone=zone,
            _lifecycle=_lifecycle,
        )
