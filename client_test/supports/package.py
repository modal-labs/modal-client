import asyncio

from modal._function_utils import FunctionInfo
from modal.mount import _get_files

from .a import *  # noqa
from .b.c import *  # noqa


def f():
    pass


async def get_files():
    fn_info = FunctionInfo(f)

    async for file_info in _get_files(fn_info.package_path, fn_info.condition, fn_info.recursive):
        print(file_info[0])


if __name__ == "__main__":
    asyncio.run(get_files())
