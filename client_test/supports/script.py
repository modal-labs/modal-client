import asyncio

import b  # noqa
import b.c  # noqa
import six  # noqa

import a  # noqa
from modal._app_state import AppState
from modal._function_utils import FunctionInfo
from modal.app import _App


def f():
    pass


async def get_files():
    fn_info = FunctionInfo(f)

    app = _App()
    app.state = AppState.RUNNING

    for _, mount in fn_info.get_mounts().items():
        async for file_info in mount._get_files():
            print(file_info.rel_filename)


if __name__ == "__main__":
    asyncio.run(get_files())
