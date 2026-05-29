# Copyright Modal Labs 2024
from collections.abc import Sequence

from modal_proto import api_pb2


class SchedulerPlacement:
    """mdmd:hidden This is an experimental feature."""

    proto: api_pb2.SchedulerPlacement

    def __init__(
        self,
        region: str | Sequence[str] | None = None,
        zone: str | None = None,
        spot: bool | None = None,
        instance_type: str | Sequence[str] | None = None,
    ):
        """mdmd:hidden"""
        _lifecycle: str | None = None
        if spot is not None:
            _lifecycle = "spot" if spot else "on-demand"

        regions = []
        if region:
            if isinstance(region, str):
                regions = [region]
            else:
                regions = list(region)

        instance_types = []
        if instance_type:
            if isinstance(instance_type, str):
                instance_types = [instance_type]
            else:
                instance_types = list(instance_type)

        self.proto = api_pb2.SchedulerPlacement(
            regions=regions,
            _zone=zone,
            _lifecycle=_lifecycle,
            _instance_types=instance_types,
        )
