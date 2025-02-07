# Copyright Modal Labs 2022
from dataclasses import dataclass
from typing import Union

from modal_proto import api_pb2

from .exception import InvalidError


@dataclass(frozen=True)
class _GPUConfig:
    gpu_type: str
    count: int

    def _to_proto(self) -> api_pb2.GPUConfig:
        """Convert this GPU config to an internal protobuf representation."""
        return api_pb2.GPUConfig(
            gpu_type=self.gpu_type,
            count=self.count,
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
        super().__init__("T4", count)

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
        super().__init__("L4", count)

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
            super().__init__("A100-40GB", count)
        elif size == "80GB":
            super().__init__("A100-80GB", count)
        else:
            raise ValueError(f"size='{size}' is invalid. A100s can only have memory values of 40GB or 80GB.")

    def __repr__(self):
        return f"GPU({self.gpu_type}, count={self.count})"


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
        super().__init__("A10G", count)

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
        super().__init__("H100", count)

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
        super().__init__("L40S", count)

    def __repr__(self):
        return f"GPU(L40S, count={self.count})"


class Any(_GPUConfig):
    """Selects any one of the GPU classes available within Modal, according to availability."""

    def __init__(self, *, count: int = 1):
        super().__init__("ANY", count)

    def __repr__(self):
        return f"GPU(Any, count={self.count})"


STRING_TO_GPU_TYPE: dict[str, str] = {
    # TODO(erikbern): we will move this table to the server soon,
    # and let clients just pass any gpu type string through
    "t4": "T4",
    "l4": "L4",
    "a100": "A100-40GB",
    "a100-80gb": "A100-80GB",
    "h100": "H100",
    "a10g": "A10G",
    "l40s": "L40S",
    "any": "ANY",
}
display_string_to_config = "\n".join(f'- "{key}"' for key in STRING_TO_GPU_TYPE.keys())
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


def parse_gpu_config(value: GPU_T) -> api_pb2.GPUConfig:
    if isinstance(value, _GPUConfig):
        return value._to_proto()
    elif isinstance(value, str):
        count = 1
        if ":" in value:
            value, count_str = value.split(":", 1)
            try:
                count = int(count_str)
            except ValueError:
                raise InvalidError(f"Invalid GPU count: {count_str}. Value must be an integer.")

        if value.lower() == "a100-40gb":
            gpu_type = "A100-40GB"
        elif value.lower() not in STRING_TO_GPU_TYPE:
            raise InvalidError(
                f"Invalid GPU type: {value}. "
                f"Value must be one of {list(STRING_TO_GPU_TYPE.keys())} (case-insensitive)."
            )
        else:
            gpu_type = STRING_TO_GPU_TYPE[value.lower()]

        return api_pb2.GPUConfig(
            gpu_type=gpu_type,
            count=count,
        )
    elif value is None or value is False:
        return api_pb2.GPUConfig()
    else:
        raise InvalidError(f"Invalid GPU config: {value}. Value must be a string, a `GPUConfig` object, or `None`.")
