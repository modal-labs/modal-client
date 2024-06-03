# Copyright Modal Labs 2024
from typing import Optional, Tuple, Union

from modal_proto import api_pb2

from .exception import InvalidError
from .gpu import GPU_T, parse_gpu_config


def convert_fn_config_to_resources_config(
    *,
    cpu: Optional[float],
    memory: Optional[Union[int, Tuple[int, int]]],
    gpu: GPU_T,
    ephemeral_disk: Optional[int],
) -> api_pb2.Resources:
    if cpu is not None and cpu < 0.1:
        raise InvalidError(f"Invalid fractional CPU value {cpu}. Cannot have less than 0.10 CPU resources.")
    gpu_config = parse_gpu_config(gpu)
    milli_cpu = int(1000 * cpu) if cpu is not None else None
    if memory and isinstance(memory, int):
        memory_mb = memory
        memory_mb_max = 0  # no limit
    elif memory and isinstance(memory, tuple):
        memory_mb, memory_mb_max = memory
        if memory_mb_max < memory_mb:
            raise InvalidError(f"Cannot specify a memory limit lower than request: {memory_mb_max} < {memory_mb}")
    else:
        memory_mb = 0
        memory_mb_max = 0
    return api_pb2.Resources(
        milli_cpu=milli_cpu,
        gpu_config=gpu_config,
        memory_mb=memory_mb,
        memory_mb_max=memory_mb_max,
        ephemeral_disk_mb=ephemeral_disk,
    )
