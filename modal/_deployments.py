# Copyright Modal Labs 2022
from typing import TYPE_CHECKING, Optional, TypeVar

from grpclib import GRPCError, Status

from modal_proto import api_pb2
from modal_utils.grpc_utils import retry_transient_errors

from ._resolver import Resolver
from .client import _Client
from .exception import InvalidError

if TYPE_CHECKING:
    from .object import _Object

else:
    _Object = TypeVar("_Object")


async def deploy_single_object(
    obj: _Object, type_prefix: str, client: _Client, label: str, namespace, environment_name: Optional[str]
):
    """mdmd:hidden"""
    existing_object_id: Optional[str]
    existing_app_id: Optional[str]

    # Look up existing app+object
    request = api_pb2.AppLookupObjectRequest(
        app_name=label,
        namespace=namespace,
        object_entity=type_prefix,
        environment_name=environment_name,
    )
    try:
        response = await retry_transient_errors(client.stub.AppLookupObject, request)
        existing_object_id = response.object.object_id
        existing_app_id = response.app_id
        assert existing_object_id
        assert existing_app_id
    except GRPCError as exc:
        if exc.status == Status.NOT_FOUND:
            existing_object_id = None
            existing_app_id = None
        else:
            raise

    if existing_app_id is None:
        # Create new app if it doesn't exist (duplicated from `_LocalApp._init_new` temporarily)
        app_req = api_pb2.AppCreateRequest(
            description=label,
            environment_name=environment_name,
            app_state=api_pb2.APP_STATE_INITIALIZING,
        )
        app_resp = await retry_transient_errors(client.stub.AppCreate, app_req)
        existing_app_id = app_resp.app_id

    # Create object
    resolver = Resolver(client, environment_name=environment_name, app_id=existing_app_id)
    await resolver.load(obj, existing_object_id)
    if existing_object_id is not None:
        assert obj.object_id == existing_object_id
    assert len(resolver.objects()) == 1
    req_set = api_pb2.AppSetObjectsRequest(
        app_id=existing_app_id,
        single_object_id=obj.object_id,
    )
    await retry_transient_errors(client.stub.AppSetObjects, req_set)

    # Deploy app (duplicated from `_LocalApp.deploy` temporarily)
    deploy_req = api_pb2.AppDeployRequest(
        app_id=existing_app_id,
        name=label,
        namespace=namespace,
        object_entity=type_prefix,
    )
    try:
        await retry_transient_errors(client.stub.AppDeploy, deploy_req)
    except GRPCError as exc:
        if exc.status == Status.INVALID_ARGUMENT:
            raise InvalidError(exc.message)
        if exc.status == Status.FAILED_PRECONDITION:
            raise InvalidError(exc.message)
        raise
