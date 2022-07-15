import asyncio

import pkg_b.f  # noqa
import pkg_b.g.h  # noqa

import modal
from modal._function_utils import FunctionInfo

from .a import *  # noqa
from .b.c import *  # noqa


def f():
    pass


async def get_files():
    fn_info = FunctionInfo(f)

    for _, mount in fn_info.get_mounts().items():
        async for file_info in mount._get_files():
            if not file_info.filename.startswith(modal.__path__[0]):
                print(file_info.rel_filename)


if __name__ == "__main__":
    asyncio.run(get_files())
