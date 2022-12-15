# Copyright Modal Labs 2022
import inspect

from modal._function_utils import FunctionInfo
from modal_utils.async_utils import synchronize_apis


class _Entrypoint:
    def __init__(self, f_info: FunctionInfo):
        self.f_info = f_info

    async def __call__(self, *args, **kwargs):
        result = self.f_info.raw_f(*args, **kwargs)
        if inspect.iscoroutine(result):
            result = await result
        return result


Entrypoint, AioEntrypoint = synchronize_apis(_Entrypoint)
