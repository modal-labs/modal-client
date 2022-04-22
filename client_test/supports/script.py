import asyncio

import a  # noqa
import b  # noqa
import b.c  # noqa
import six  # noqa

from modal._app_state import AppState
from modal._function_utils import FunctionInfo
from modal.app import _App


def f():
    pass


async def get_files():
    fn_info = FunctionInfo(f)

    app = _App()
    app.state = AppState.RUNNING

    for mount in fn_info.create_mounts(app):
        async for file_info in mount._get_files():
            print(file_info[0])


if __name__ == "__main__":
    asyncio.run(get_files())
