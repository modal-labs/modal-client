# Copyright Modal Labs 2024
# Note: this is a temporary module until we've (1) deleted the current app.py (3) renamed stub.py to app.py
from typing import List, Optional

from modal_proto import api_pb2

from ._utils.async_utils import synchronize_api
from .client import _Client
from .object import _get_environment_name


async def _list_apps(env: Optional[str] = None, client: Optional[_Client] = None) -> List[api_pb2.AppStats]:
    """List apps in a given Modal environment."""
    if client is None:
        client = await _Client.from_env()
    resp: api_pb2.AppListResponse = await client.stub.AppList(
        api_pb2.AppListRequest(environment_name=_get_environment_name(env))
    )
    return list(resp.apps)


list_apps = synchronize_api(_list_apps)
