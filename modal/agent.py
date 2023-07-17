from .object import _Handle

from modal_utils.async_utils import synchronize_api


class FileHandleReader:
    async def read(self):
        pass


class _AgentHandle(_Handle, type_prefix="ag"):
    """mdmd:hidden"""

    async def wait(self):
        pass


AgentHandle = synchronize_api(_AgentHandle)
