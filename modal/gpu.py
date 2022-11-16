# Copyright Modal Labs 2022
from dataclasses import dataclass

from modal_proto import api_pb2


@dataclass
class _GPUConfig:
    type: "api_pb2.GPUType.V"
    count: int

    def _to_proto(self) -> api_pb2.GPUConfig:
        """Convert this GPU config to an internal protobuf representation."""
        return api_pb2.GPUConfig(
            type=self.type,
            count=self.count,
        )


class T4(_GPUConfig):
    def __init__(self):
        super().__init__(api_pb2.GPU_TYPE_T4, 1)


class A100(_GPUConfig):
    def __init__(self, *, count: int = 1):
        super().__init__(api_pb2.GPU_TYPE_A100, count)
