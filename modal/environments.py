# Copyright Modal Labs 2023
from typing import Optional, List

from google.protobuf.empty_pb2 import Empty
from google.protobuf.wrappers_pb2 import StringValue

from modal.client import _Client
from modal.config import config
from modal_proto import api_pb2
from modal_utils.async_utils import synchronizer


@synchronizer.create_blocking
async def delete_environment(name: str):
    client = await _Client.from_env()
    stub = client.stub
    await stub.EnvironmentDelete(api_pb2.EnvironmentDeleteRequest(name=name))


@synchronizer.create_blocking
async def update_environment(
    current_name: str, *, new_name: Optional[str] = None, new_web_suffix: Optional[str] = None
):
    if new_name is not None:
        if len(new_name) < 1:
            raise ValueError("The new environment name cannot be empty")

        new_name = StringValue(value=new_name)
    if new_web_suffix is not None:
        new_web_suffix = StringValue(value=new_web_suffix)

    update_payload = api_pb2.EnvironmentUpdateRequest(
        current_name=current_name, name=new_name, web_suffix=new_web_suffix
    )
    client = await _Client.from_env()
    stub = client.stub
    await stub.EnvironmentUpdate(update_payload)


@synchronizer.create_blocking
async def create_environment(name: str):
    client = await _Client.from_env()
    stub = client.stub
    await stub.EnvironmentCreate(api_pb2.EnvironmentCreateRequest(name=name))


@synchronizer.create_blocking
async def list_environments() -> List[api_pb2.EnvironmentListItem]:
    client = await _Client.from_env()
    stub = client.stub
    resp = await stub.EnvironmentList(Empty())
    return list(resp.items)


def ensure_env(environment_name: Optional[str] = None) -> str:
    """Override config environment with environment from environment_name

    This is necessary since a cli command that runs Modal code, e.g. `modal.lookup()` without
    explicit environment specification wouldn't pick up the environment specified in a command
    line flag otherwise, e.g. when doing `modal run --env=foo`
    """
    if environment_name is not None:
        config.override_locally("environment", environment_name)

    return config.get("environment")
