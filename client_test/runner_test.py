# Copyright Modal Labs 2023
import pytest
import typing

import modal
from modal.runner import run_stub
from modal_proto import api_pb2

T = typing.TypeVar("T")


def test_run_stub(servicer, client):
    dummy_stub = modal.Stub()
    with servicer.intercept() as ctx:
        with run_stub(dummy_stub, client=client):
            pass

    ctx.pop_request("AppCreate")
    ctx.pop_request("AppSetObjects")
    ctx.pop_request("AppClientDisconnect")


def test_run_stub_profile_env_with_refs(servicer, client, monkeypatch):
    monkeypatch.setenv("MODAL_ENVIRONMENT", "profile_env")
    with servicer.intercept() as ctx:
        dummy_stub = modal.Stub()
        dummy_stub.ref = modal.Secret.from_name("some_secret")
    assert ctx.calls == []  # all calls should be deferred

    with servicer.intercept() as ctx:
        ctx.add_response("AppLookupObject", api_pb2.AppLookupObjectResponse(object_id="st-123"))
        with run_stub(dummy_stub, client=client):
            pass

    with pytest.raises(Exception):
        ctx.pop_request("SecretCreate")  # should not create a new secret...

    app_create = ctx.pop_request("AppCreate")
    assert app_create.environment_name == "profile_env"

    app_lookup_object = ctx.pop_request("AppLookupObject")
    assert app_lookup_object.environment_name == "profile_env"


def test_run_stub_custom_env_with_refs(servicer, client, monkeypatch):
    monkeypatch.setenv("MODAL_ENVIRONMENT", "profile_env")
    dummy_stub = modal.Stub()
    dummy_stub.own_env_secret = modal.Secret.from_name("own_env_secret")
    dummy_stub.other_env_secret = modal.Secret.from_name(
        "other_env_secret", environment_name="third"
    )  # explicit lookup

    with servicer.intercept() as ctx:
        ctx.add_response("AppLookupObject", api_pb2.AppLookupObjectResponse(object_id="st-123"))
        ctx.add_response("AppLookupObject", api_pb2.AppLookupObjectResponse(object_id="st-456"))
        with run_stub(dummy_stub, client=client, environment_name="custom"):
            pass

    with pytest.raises(Exception):
        ctx.pop_request("SecretCreate")

    app_create = ctx.pop_request("AppCreate")
    assert app_create.environment_name == "custom"

    app_lookup_object = ctx.pop_request("AppLookupObject")
    assert app_lookup_object.environment_name == "custom"

    app_lookup_object2 = ctx.pop_request("AppLookupObject")
    assert app_lookup_object2.environment_name == "third"
