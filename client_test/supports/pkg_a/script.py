import asyncio

import a  # noqa
import b  # noqa
import b.c  # noqa
import pkg_b  # noqa
import six  # noqa

from modal._function_utils import FunctionInfo


def f():
    pass


async def get_files():
    fn_info = FunctionInfo(f)
    import modal

    for _, mount in fn_info.get_mounts().items():
        async for file_info in mount._get_files():
            if not file_info.filename.startswith(modal.__path__[0]):
                print(file_info.rel_filename)


if __name__ == "__main__":
    asyncio.run(get_files())
