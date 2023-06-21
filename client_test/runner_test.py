# Copyright Modal Labs 2023
import typing
from typing import List, Any, Tuple

import pytest

import modal
from modal.runner import run_stub
from modal_proto import api_pb2

T = typing.TypeVar("T")


def pop_message(calls: List[Tuple[str, Any]], message_type: typing.Type[T]) -> Tuple[T, List[Tuple[str, Any]]]:
    for i, (_, msg) in enumerate(calls):
        if isinstance(msg, message_type):
            return msg, calls[i + 1 :]

    raise Exception("No message of that type in call list")


def test_run_stub(servicer, client):
    dummy_stub = modal.Stub()
    with servicer.intercept() as ctx:
        with run_stub(dummy_stub, client=client):
            pass

    _, remaining_calls = pop_message(ctx.calls, api_pb2.AppCreateRequest)
    _, remaining_calls = pop_message(remaining_calls, api_pb2.AppSetObjectsRequest)
    _, remaining_calls = pop_message(remaining_calls, api_pb2.AppClientDisconnectRequest)


def test_run_stub_profile_env_with_refs(servicer, client, monkeypatch):
    monkeypatch.setenv("MODAL_ENVIRONMENT", "profile_env")
    with servicer.intercept() as ctx:
        dummy_stub = modal.Stub()
        dummy_stub.ref = modal.Secret.from_name("some_secret")
    assert ctx.calls == []  # all calls should be deferred

    with servicer.intercept() as ctx:
        ctx.add_response("AppLookupObject", [api_pb2.AppLookupObjectResponse(object_id="st-123")])
        with run_stub(dummy_stub, client=client):
            pass

    with pytest.raises(Exception):
        pop_message(ctx.calls, api_pb2.SecretCreateRequest)  # should not create a new secret...

    app_create, remaining_calls = pop_message(ctx.calls, api_pb2.AppCreateRequest)
    assert app_create.environment_name == "profile_env"

    app_lookup_object, remaining_calls = pop_message(remaining_calls, api_pb2.AppLookupObjectRequest)
    assert app_lookup_object.environment_name == "profile_env"


def test_run_stub_custom_env_with_refs(servicer, client, monkeypatch):
    monkeypatch.setenv("MODAL_ENVIRONMENT", "profile_env")
    dummy_stub = modal.Stub()
    dummy_stub.own_env_secret = modal.Secret.from_name("own_env_secret")
    dummy_stub.other_env_secret = modal.Secret.from_name(
        "other_env_secret", environment_name="third"
    )  # explicit lookup

    with servicer.intercept() as ctx:
        ctx.add_response("AppLookupObject", [api_pb2.AppLookupObjectResponse(object_id="st-123")])
        ctx.add_response("AppLookupObject", [api_pb2.AppLookupObjectResponse(object_id="st-456")])
        with run_stub(dummy_stub, client=client, environment_name="custom"):
            pass

    with pytest.raises(Exception):
        pop_message(ctx.calls, api_pb2.SecretCreateRequest)

    app_create, remaining_calls = pop_message(ctx.calls, api_pb2.AppCreateRequest)
    assert app_create.environment_name == "custom"

    app_lookup_object, remaining_calls = pop_message(remaining_calls, api_pb2.AppLookupObjectRequest)
    assert app_lookup_object.environment_name == "custom"

    app_lookup_object, remaining_calls = pop_message(remaining_calls, api_pb2.AppLookupObjectRequest)
    assert app_lookup_object.environment_name == "third"
