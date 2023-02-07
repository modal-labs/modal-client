# Copyright Modal Labs 2022
import asyncio

import a  # noqa
import b  # noqa
import b.c  # noqa
import pkg_b  # noqa
import six  # noqa

import modal  # noqa
from modal._function_utils import FunctionInfo


def f():
    pass


async def get_files():
    fn_info = FunctionInfo(f, serialized=True)

    for _, mount in fn_info.get_mounts().items():
        async for filename in mount._get_remote_files():
            print(filename)


if __name__ == "__main__":
    asyncio.run(get_files())
