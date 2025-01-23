# Copyright Modal Labs 2022
from dataclasses import dataclass
from typing import Callable, Optional, Union

from modal_proto import api_pb2

from .exception import InvalidError


@dataclass(frozen=True)
class _GPUConfig:
    type: "api_pb2.GPUType.V"  # Deprecated, at some point
    count: int
    gpu_type: str
    memory: int = 0

    def _to_proto(self) -> api_pb2.GPUConfig:
        """Convert this GPU config to an internal protobuf representation."""
        return api_pb2.GPUConfig(
            type=self.type,
            count=self.count,
            memory=self.memory,
            gpu_type=self.gpu_type,
        )


class T4(_GPUConfig):
    """
    [NVIDIA T4 Tensor Core](https://www.nvidia.com/en-us/data-center/tesla-t4/) GPU class.

    A low-cost data center GPU based on the Turing architecture, providing 16GB of GPU memory.
    """

    def __init__(
        self,
        count: int = 1,  # Number of GPUs per container. Defaults to 1.
    ):
        super().__init__(api_pb2.GPU_TYPE_T4, count, "T4")

    def __repr__(self):
        return f"GPU(T4, count={self.count})"


class L4(_GPUConfig):
    """
    [NVIDIA L4 Tensor Core](https://www.nvidia.com/en-us/data-center/l4/) GPU class.

    A mid-tier data center GPU based on the Ada Lovelace architecture, providing 24GB of GPU memory.
    Includes RTX (ray tracing) support.
    """

    def __init__(
        self,
        count: int = 1,  # Number of GPUs per container. Defaults to 1.
    ):
        super().__init__(api_pb2.GPU_TYPE_L4, count, "L4")

    def __repr__(self):
        return f"GPU(L4, count={self.count})"


class A100(_GPUConfig):
    """
    [NVIDIA A100 Tensor Core](https://www.nvidia.com/en-us/data-center/a100/) GPU class.

    The flagship data center GPU of the Ampere architecture. Available in 40GB and 80GB GPU memory configurations.
    """

    def __init__(
        self,
        *,
        count: int = 1,  # Number of GPUs per container. Defaults to 1.
        size: Union[str, None] = None,  # Select GB configuration of GPU device: "40GB" or "80GB". Defaults to "40GB".
    ):
        if size == "40GB" or not size:
            super().__init__(api_pb2.GPU_TYPE_A100, count, "A100-40GB", 40)
        elif size == "80GB":
            super().__init__(api_pb2.GPU_TYPE_A100_80GB, count, "A100-80GB", 80)
        else:
            raise ValueError(f"size='{size}' is invalid. A100s can only have memory values of 40GB or 80GB.")

    def __repr__(self):
        if self.memory == 80:
            return f"GPU(A100-80GB, count={self.count})"
        else:
            return f"GPU(A100-40GB, count={self.count})"


class A10G(_GPUConfig):
    """
    [NVIDIA A10G Tensor Core](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) GPU class.

    A mid-tier data center GPU based on the Ampere architecture, providing 24 GB of memory.
    A10G GPUs deliver up to 3.3x better ML training performance, 3x better ML inference performance,
    and 3x better graphics performance, in comparison to NVIDIA T4 GPUs.
    """

    def __init__(
        self,
        *,
        # Number of GPUs per container. Defaults to 1.
        # Useful if you have very large models that don't fit on a single GPU.
        count: int = 1,
    ):
        super().__init__(api_pb2.GPU_TYPE_A10G, count, "A10G")

    def __repr__(self):
        return f"GPU(A10G, count={self.count})"


class H100(_GPUConfig):
    """
    [NVIDIA H100 Tensor Core](https://www.nvidia.com/en-us/data-center/h100/) GPU class.

    The flagship data center GPU of the Hopper architecture.
    Enhanced support for FP8 precision and a Transformer Engine that provides up to 4X faster training
    over the prior generation for GPT-3 (175B) models.
    """

    def __init__(
        self,
        *,
        # Number of GPUs per container. Defaults to 1.
        # Useful if you have very large models that don't fit on a single GPU.
        count: int = 1,
    ):
        super().__init__(api_pb2.GPU_TYPE_H100, count, "H100")

    def __repr__(self):
        return f"GPU(H100, count={self.count})"


class L40S(_GPUConfig):
    """
    [NVIDIA L40S](https://www.nvidia.com/en-us/data-center/l40s/) GPU class.

    The L40S is a data center GPU for the Ada Lovelace architecture. It has 48 GB of on-chip
    GDDR6 RAM and enhanced support for FP8 precision.
    """

    def __init__(
        self,
        *,
        # Number of GPUs per container. Defaults to 1.
        # Useful if you have very large models that don't fit on a single GPU.
        count: int = 1,
    ):
        super().__init__(api_pb2.GPU_TYPE_L40S, count, "L40S")

    def __repr__(self):
        return f"GPU(L40S, count={self.count})"


class Any(_GPUConfig):
    """Selects any one of the GPU classes available within Modal, according to availability."""

    def __init__(self, *, count: int = 1):
        super().__init__(api_pb2.GPU_TYPE_ANY, count, "ANY")

    def __repr__(self):
        return f"GPU(Any, count={self.count})"


STRING_TO_GPU_CONFIG: dict[str, Callable] = {
    "t4": T4,
    "l4": L4,
    "a100": A100,
    "a100-80gb": lambda: A100(size="80GB"),
    "h100": H100,
    "a10g": A10G,
    "l40s": L40S,
    "any": Any,
}
display_string_to_config = "\n".join(f'- "{key}" → `{c()}`' for key, c in STRING_TO_GPU_CONFIG.items() if key != "inf2")
__doc__ = f"""
**GPU configuration shortcodes**

The following are the valid `str` values for the `gpu` parameter of
[`@app.function`](/docs/reference/modal.App#function).

{display_string_to_config}

The shortcodes also support specifying count by suffixing `:N` to acquire `N` GPUs.
For example, `a10g:4` will provision 4 A10G GPUs.

Other configurations can be created using the constructors documented below.
"""

# bool will be deprecated in future versions.
GPU_T = Union[None, bool, str, _GPUConfig]


def _parse_gpu_config(value: GPU_T) -> Optional[_GPUConfig]:
    if isinstance(value, _GPUConfig):
        return value
    elif isinstance(value, str):
        count = 1
        if ":" in value:
            value, count_str = value.split(":", 1)
            try:
                count = int(count_str)
            except ValueError:
                raise InvalidError(f"Invalid GPU count: {count_str}. Value must be an integer.")

        if value.lower() == "a100-20g":
            return A100(size="20GB", count=count)  # Triggers unsupported error underneath.
        elif value.lower() == "a100-40gb":
            return A100(size="40GB", count=count)
        elif value.lower() == "a100-80gb":
            return A100(size="80GB", count=count)
        elif value.lower() not in STRING_TO_GPU_CONFIG:
            raise InvalidError(
                f"Invalid GPU type: {value}. "
                f"Value must be one of {list(STRING_TO_GPU_CONFIG.keys())} (case-insensitive)."
            )
        else:
            return STRING_TO_GPU_CONFIG[value.lower()](count=count)
    elif value is None or value is False:
        return None
    else:
        raise InvalidError(f"Invalid GPU config: {value}. Value must be a string, a `GPUConfig` object, or `None`.")


def parse_gpu_config(value: GPU_T) -> api_pb2.GPUConfig:
    """mdmd:hidden"""
    gpu_config = _parse_gpu_config(value)
    if gpu_config is None:
        return api_pb2.GPUConfig()
    return gpu_config._to_proto()


def display_gpu_config(value: GPU_T) -> str:
    """mdmd:hidden"""
    return repr(_parse_gpu_config(value))
