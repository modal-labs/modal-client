# Copyright Modal Labs 2022
from dataclasses import dataclass
from typing import Optional, Union

from modal_proto import api_pb2

from .exception import InvalidError, deprecation_error, deprecation_warning


@dataclass(frozen=True)
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
    [NVIDIA T4 Tensor Core](https://www.nvidia.com/en-us/data-center/tesla-t4/) GPU class.

    A low-cost data center GPU based on the Turing architecture, providing 16GiB of GPU memory.
    """

    def __init__(
        self,
        count: int = 1,  # Number of GPUs per container. Defaults to 1.
    ):
        super().__init__(api_pb2.GPU_TYPE_T4, count, 0)

    def __repr__(self):
        return f"GPU(T4, count={self.count})"


class L4(_GPUConfig):
    """
    [NVIDIA L4 Tensor Core](https://www.nvidia.com/en-us/data-center/l4/) GPU class.

    A mid-tier data center GPU based on the Ada Lovelace architecture, providing 24GiB of GPU memory.
    Includes RTX (ray tracing) support.
    """

    def __init__(
        self,
        count: int = 1,  # Number of GPUs per container. Defaults to 1.
    ):
        super().__init__(api_pb2.GPU_TYPE_L4, count, 0)

    def __repr__(self):
        return f"GPU(L4, count={self.count})"


class A100(_GPUConfig):
    """
    [NVIDIA A100 Tensor Core](https://www.nvidia.com/en-us/data-center/a100/) GPU class.

    The flagship data center GPU of the Ampere architecture. Available in 40GiB and 80GiB GPU memory configurations.
    """

    def __init__(
        self,
        *,
        count: int = 1,  # Number of GPUs per container. Defaults to 1.
        memory: Optional[int] = None,  # Deprecated. Use `size` instead.
        size: Union[str, None] = None,  # Select GiB configuration of GPU device: "40GB" or "80GB". Defaults to "40GB".
    ):
        allowed_memory_values = {40, 80}
        allowed_size_values = {"40GB", "80GB"}

        if memory is not None:
            deprecation_warning(
                (2024, 5, 16),
                "The `memory` parameter is deprecated. Use the `size='80GB'` parameter instead.",
            )

        if memory == 20:
            raise ValueError(
                "A100 20GB is unsupported, consider `modal.A10G`, `modal.A100(memory_gb='40')`, or `modal.H100` instead"
            )
        elif memory and size:
            raise ValueError("Cannot specify both `memory` and `size`. Just specify `size`.")
        elif memory:
            if memory not in allowed_memory_values:
                raise ValueError(f"A100s can only have memory values of {allowed_memory_values} => memory={memory}")
        elif size:
            if size not in allowed_size_values:
                raise ValueError(
                    f"size='{size}' is invalid. A100s can only have memory values of {allowed_size_values}."
                )
            memory = int(size.replace("GB", ""))
        else:
            memory = 40

        if memory == 80:
            super().__init__(api_pb2.GPU_TYPE_A100_80GB, count, memory)
        else:
            super().__init__(api_pb2.GPU_TYPE_A100, count, memory)

    def __repr__(self):
        if self.memory == 80:
            return f"GPU(A100-80GB, count={self.count})"
        else:
            return f"GPU(A100-40GB, count={self.count})"


class A10G(_GPUConfig):
    """
    [NVIDIA A10G Tensor Core](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) GPU class.

    A mid-tier data center GPU based on the Ampere architecture, providing 24 GiB of memory.
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
        super().__init__(api_pb2.GPU_TYPE_A10G, count)

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
        super().__init__(api_pb2.GPU_TYPE_H100, count)

    def __repr__(self):
        return f"GPU(H100, count={self.count})"


class Any(_GPUConfig):
    """Selects any one of the GPU classes available within Modal, according to availability."""

    def __init__(self, *, count: int = 1):
        super().__init__(api_pb2.GPU_TYPE_ANY, count)

    def __repr__(self):
        return f"GPU(Any, count={self.count})"


STRING_TO_GPU_CONFIG = {
    "t4": T4,
    "l4": L4,
    "a100": A100,
    "h100": H100,
    "a10g": A10G,
    "any": Any,
}
display_string_to_config = "\n".join(
    f'- "{key}" → `{cls()}`' for key, cls in STRING_TO_GPU_CONFIG.items() if key != "inf2"
)
__doc__ = f"""
**GPU configuration shortcodes**

The following are the valid `str` values for the `gpu` parameter of
[`@app.function`](/docs/reference/modal.Stub#function).

{display_string_to_config}

Other configurations can be created using the constructors documented below.
"""

# bool will be deprecated in future versions.
GPU_T = Union[None, bool, str, _GPUConfig]


def _parse_gpu_config(value: GPU_T, raise_on_true: bool = True) -> Optional[_GPUConfig]:
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
            return A100(memory=20, count=count)  # Triggers unsupported error underneath.
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
    elif value is True:
        if raise_on_true:
            deprecation_error(
                (2022, 12, 19), 'Setting gpu=True is deprecated. Use `gpu="any"` or `gpu=modal.gpu.Any()` instead.'
            )
        else:
            # We didn't support targeting a GPU type for run_function until 2023-12-12
            deprecation_warning(
                (2023, 12, 13), 'Setting gpu=True is deprecated. Use `gpu="any"` or `gpu=modal.gpu.Any()` instead.'
            )
        return Any()
    elif value is None or value is False:
        return None
    else:
        raise InvalidError(f"Invalid GPU config: {value}. Value must be a string, a `GPUConfig` object, or `None`.")


def parse_gpu_config(value: GPU_T, raise_on_true: bool = True) -> api_pb2.GPUConfig:
    """mdmd:hidden"""
    gpu_config = _parse_gpu_config(value, raise_on_true)
    if gpu_config is None:
        return api_pb2.GPUConfig()
    return gpu_config._to_proto()


def display_gpu_config(value: GPU_T) -> str:
    """mdmd:hidden"""
    return repr(_parse_gpu_config(value, False))
