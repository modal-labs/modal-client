# Copyright Modal Labs 2022
from dataclasses import dataclass
from datetime import date
from typing import Union

from modal_proto import api_pb2

from .exception import InvalidError, deprecation_warning


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


class A10G(_GPUConfig):
    def __init__(self, *, count: int = 1):
        super().__init__(api_pb2.GPU_TYPE_A10G, count)


class Any(_GPUConfig):
    def __init__(self, *, count: int = 1):
        super().__init__(api_pb2.GPU_TYPE_ANY, count)


STRING_TO_GPU_CONFIG = {"t4": T4(), "a100": A100(), "a10g": A10G(), "any": Any()}

# bool will be deprecated in future versions.
GPU_T = Union[None, bool, str, _GPUConfig]


def parse_gpu_config(value: GPU_T) -> api_pb2.GPUConfig:
    if isinstance(value, _GPUConfig):
        return value._to_proto()
    elif isinstance(value, str):
        if value.lower() not in STRING_TO_GPU_CONFIG:
            raise InvalidError(
                f"Invalid GPU type: {value}. Value must be one of {list(STRING_TO_GPU_CONFIG.keys())} (case-insensitive)."
            )
        return STRING_TO_GPU_CONFIG[value.lower()]._to_proto()
    elif value is True:
        deprecation_warning(
            date(2022, 12, 19), 'Setting gpu=True is deprecated. Use `gpu="any"` or `gpu=modal.gpu.Any()` instead.'
        )
        return Any()._to_proto()
    elif value is None or value is False:
        return api_pb2.GPUConfig()
    else:
        raise InvalidError(f"Invalid GPU config: {value}. Value must be a string, a GPUConfig object or `None`.")
