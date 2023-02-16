# Copyright Modal Labs 2022
from dataclasses import dataclass
from datetime import date
from typing import Union, Optional

from modal_proto import api_pb2

from .exception import InvalidError, deprecation_warning


@dataclass
class _GPUConfig:
    type: "api_pb2.GPUType.V"
    count: int
    memory: int

    def _to_proto(self) -> api_pb2.GPUConfig:
        """Convert this GPU config to an internal protobuf representation."""
        return api_pb2.GPUConfig(
            type=self.type,
            count=self.count,
            memory=self.memory,
        )


class T4(_GPUConfig):
    def __init__(self):
        super().__init__(api_pb2.GPU_TYPE_T4, 1, 0)

    def __repr__(self):
        return f"GPU(T4, count={self.count})"


class A100(_GPUConfig):
    def __init__(self, *, count: int = 1, memory: int = 0):
        allowed_memory_values = {0, 20, 40}
        if memory not in allowed_memory_values:
            raise ValueError(f"A100s can only have memory values of {allowed_memory_values} => memory={memory}")
        if memory == 20:
            super().__init__(api_pb2.GPU_TYPE_A100_20G, count, memory)
        else:
            super().__init__(api_pb2.GPU_TYPE_A100, count, memory)

    def __repr__(self):
        if self.memory == 20:
            return f"GPU(A100-20G, count={self.count})"
        else:
            return f"GPU(A100-40G, count={self.count})"


class A10G(_GPUConfig):
    def __init__(self, *, count: int = 1, memory: int = 0):
        super().__init__(api_pb2.GPU_TYPE_A10G, count, memory)

    def __repr__(self):
        return f"GPU(A10G, count={self.count})"


class Any(_GPUConfig):
    def __init__(self, *, count: int = 1, memory: int = 0):
        super().__init__(api_pb2.GPU_TYPE_ANY, count, memory)

    def __repr__(self):
        return f"GPU(Any, count={self.count})"


STRING_TO_GPU_CONFIG = {"t4": T4(), "a100": A100(), "a100-20g": A100(memory=20), "a10g": A10G(), "any": Any()}

# bool will be deprecated in future versions.
GPU_T = Union[None, bool, str, _GPUConfig]


def _parse_gpu_config(value: GPU_T, warn_on_true: bool = True) -> Optional[_GPUConfig]:
    if isinstance(value, _GPUConfig):
        return value
    elif isinstance(value, str):
        if value.lower() not in STRING_TO_GPU_CONFIG:
            raise InvalidError(
                f"Invalid GPU type: {value}. Value must be one of {list(STRING_TO_GPU_CONFIG.keys())} (case-insensitive)."
            )
        return STRING_TO_GPU_CONFIG[value.lower()]
    elif value is True:
        if warn_on_true:
            deprecation_warning(
                date(2022, 12, 19), 'Setting gpu=True is deprecated. Use `gpu="any"` or `gpu=modal.gpu.Any()` instead.'
            )
        return Any()
    elif value is None or value is False:
        return None
    else:
        raise InvalidError(f"Invalid GPU config: {value}. Value must be a string, a GPUConfig object or `None`.")


def parse_gpu_config(value: GPU_T, warn_on_true: bool = True) -> api_pb2.GPUConfig:
    gpu_config = _parse_gpu_config(value, warn_on_true)
    if gpu_config is None:
        return api_pb2.GPUConfig()
    return gpu_config._to_proto()


def display_gpu_config(value: GPU_T) -> str:
    return repr(_parse_gpu_config(value, False))
