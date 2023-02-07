# Copyright Modal Labs 2022
import asyncio

import pkg_b.f  # noqa
import pkg_b.g.h  # noqa

import modal  # noqa
from modal._function_utils import FunctionInfo

from .a import *  # noqa
from .b.c import *  # noqa


def f():
    pass


async def get_files():
    fn_info = FunctionInfo(f)

    for _, mount in fn_info.get_mounts().items():
        async for filename in mount._get_remote_files():
            print(filename)


if __name__ == "__main__":
    asyncio.run(get_files())
