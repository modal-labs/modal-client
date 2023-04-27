# Copyright Modal Labs 2022
from dataclasses import dataclass
from datetime import date
from typing import Union, Optional

from modal_proto import api_pb2

from .exception import InvalidError, deprecation_error


@dataclass
class _GPUConfig:
    type: "api_pb2.GPUType.V"
    count: int
    memory: int = 0

    def _to_proto(self) -> api_pb2.GPUConfig:
        """Convert this GPU config to an internal protobuf representation."""
        return api_pb2.GPUConfig(
            type=self.type,
            count=self.count,
            memory=self.memory,
        )


class T4(_GPUConfig):
    """
    [NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) GPU class.

    Low-cost GPU option, providing 16GiB of GPU memory.
    """

    def __init__(self, count: int = 1):
        super().__init__(api_pb2.GPU_TYPE_T4, count, 0)

    def __repr__(self):
        return f"GPU(T4, count={self.count})"


class A100(_GPUConfig):
    """
    [NVIDIA A100 Tensor Core](https://www.nvidia.com/en-us/data-center/a100/) GPU class.

    The most powerful GPU available in the cloud. Available in 20GiB and 40GiB GPU memory configurations.
    """

    def __init__(self, *, count: int = 1, memory: int = 0):
        allowed_memory_values = {0, 20, 40}
        if memory not in allowed_memory_values:
            raise ValueError(f"A100s can only have memory values of {allowed_memory_values} => memory={memory}")

        # Multi-GPU workloads require a different GPU type.
        gpu_type = api_pb2.GPU_TYPE_A100
        if count > 1:
            gpu_type = api_pb2.GPU_TYPE_A100_40GB_MANY

        if memory == 20:
            if count != 1:
                raise ValueError(f"Cannot request more than 1 A100 20GB unit. Requested {count}")
            super().__init__(api_pb2.GPU_TYPE_A100_20G, count, memory)
        else:
            super().__init__(gpu_type, count, memory)

    def __repr__(self):
        if self.memory == 20:
            return f"GPU(A100-20G, count={self.count})"
        else:
            return f"GPU(A100-40G, count={self.count})"


class A10G(_GPUConfig):
    """
    [NVIDIA A10G Tensor Core](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) GPU class.

    A10G GPUs deliver up to 3.3x better ML training performance, 3x better ML inference performance,
    and 3x better graphics performance, in comparison to NVIDIA T4 GPUs.
    """

    def __init__(self, *, count: int = 1):
        super().__init__(api_pb2.GPU_TYPE_A10G, count)

    def __repr__(self):
        return f"GPU(A10G, count={self.count})"


class Inferentia2(_GPUConfig):
    """mdmd:hidden"""

    def __init__(self, *, count: int = 1):
        super().__init__(api_pb2.GPU_TYPE_INFERENTIA2, count)

    def __repr__(self):
        return f"GPU(INFERENTIA2, count={self.count})"


class Any(_GPUConfig):
    """Selects any one of the GPU classes available within Modal, according to availability."""

    def __init__(self, *, count: int = 1):
        super().__init__(api_pb2.GPU_TYPE_ANY, count)

    def __repr__(self):
        return f"GPU(Any, count={self.count})"


STRING_TO_GPU_CONFIG = {
    "t4": T4(),
    "a100": A100(),
    "a100-20g": A100(memory=20),
    "a10g": A10G(),
    "inf2": Inferentia2(),
    "any": Any(),
}
display_string_to_config = "\n".join(f'- "{key}" → `{value}`' for key, value in STRING_TO_GPU_CONFIG.items())
__doc__ = f"""
**GPU configuration shortcodes**

The following are the valid `str` values for the `gpu` parameter of [`@stub.function`](/docs/reference/modal.Stub#function).

{display_string_to_config}

Other configurations can be created using the constructors documented below.
"""

# bool will be deprecated in future versions.
GPU_T = Union[None, bool, str, _GPUConfig]


def _parse_gpu_config(value: GPU_T, raise_on_true: bool = True) -> Optional[_GPUConfig]:
    if isinstance(value, _GPUConfig):
        return value
    elif isinstance(value, str):
        if value.lower() not in STRING_TO_GPU_CONFIG:
            raise InvalidError(
                f"Invalid GPU type: {value}. Value must be one of {list(STRING_TO_GPU_CONFIG.keys())} (case-insensitive)."
            )
        return STRING_TO_GPU_CONFIG[value.lower()]
    elif value is True:
        if raise_on_true:
            deprecation_error(
                date(2022, 12, 19), 'Setting gpu=True is deprecated. Use `gpu="any"` or `gpu=modal.gpu.Any()` instead.'
            )
        return Any()
    elif value is None or value is False:
        return None
    else:
        raise InvalidError(f"Invalid GPU config: {value}. Value must be a string, a GPUConfig object or `None`.")


def parse_gpu_config(value: GPU_T, raise_on_true: bool = True) -> api_pb2.GPUConfig:
    """mdmd:hidden"""
    gpu_config = _parse_gpu_config(value, raise_on_true)
    if gpu_config is None:
        return api_pb2.GPUConfig()
    return gpu_config._to_proto()


def display_gpu_config(value: GPU_T) -> str:
    """mdmd:hidden"""
    return repr(_parse_gpu_config(value, False))
