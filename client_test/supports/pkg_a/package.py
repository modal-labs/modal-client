# Copyright Modal Labs 2022
import asyncio

import pkg_b.f  # noqa
import pkg_b.g.h  # noqa

import modal  # noqa
from modal._function_utils import FunctionInfo
from modal.functions import _get_function_mounts

from .a import *  # noqa
from .b.c import *  # noqa


def f():
    pass


async def get_files():
    fn_info = FunctionInfo(f)

    for mount in _get_function_mounts(fn_info, include_client_mount=False):
        async for file_info in mount._get_files(mount.entries):
            print(file_info.mount_filename)


if __name__ == "__main__":
    asyncio.run(get_files())
