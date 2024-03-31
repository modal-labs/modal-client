# Copyright Modal Labs 2023
import pytest
import typing

import modal
from modal.client import Client
from modal.exception import ExecutionError
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


def test_run_stub_unauthenticated(servicer):
    dummy_stub = modal.Stub()
    with Client.anonymous(servicer.remote_addr) as client:
        with pytest.raises(ExecutionError, match=".+unauthenticated client"):
            with run_stub(dummy_stub, client=client):
                pass


def dummy():
    ...


def test_run_stub_profile_env_with_refs(servicer, client, monkeypatch):
    monkeypatch.setenv("MODAL_ENVIRONMENT", "profile_env")
    with servicer.intercept() as ctx:
        dummy_stub = modal.Stub()
        ref = modal.Secret.from_name("some_secret")
        dummy_stub.function(secrets=[ref])(dummy)

    assert ctx.calls == []  # all calls should be deferred

    with servicer.intercept() as ctx:
        ctx.add_response("SecretGetOrCreate", api_pb2.SecretGetOrCreateResponse(secret_id="st-123"))
        with run_stub(dummy_stub, client=client):
            pass

    with pytest.raises(Exception):
        ctx.pop_request("SecretCreate")  # should not create a new secret...

    app_create = ctx.pop_request("AppCreate")
    assert app_create.environment_name == "profile_env"

    secret_get_or_create = ctx.pop_request("SecretGetOrCreate")
    assert secret_get_or_create.environment_name == "profile_env"


def test_run_stub_custom_env_with_refs(servicer, client, monkeypatch):
    monkeypatch.setenv("MODAL_ENVIRONMENT", "profile_env")
    dummy_stub = modal.Stub()
    own_env_secret = modal.Secret.from_name("own_env_secret")
    other_env_secret = modal.Secret.from_name("other_env_secret", environment_name="third")  # explicit lookup

    dummy_stub.function(secrets=[own_env_secret, other_env_secret])(dummy)

    with servicer.intercept() as ctx:
        ctx.add_response("SecretGetOrCreate", api_pb2.SecretGetOrCreateResponse(secret_id="st-123"))
        ctx.add_response("SecretGetOrCreate", api_pb2.SecretGetOrCreateResponse(secret_id="st-456"))
        with run_stub(dummy_stub, client=client, environment_name="custom"):
            pass

    with pytest.raises(Exception):
        ctx.pop_request("SecretCreate")

    app_create = ctx.pop_request("AppCreate")
    assert app_create.environment_name == "custom"

    secret_get_or_create = ctx.pop_request("SecretGetOrCreate")
    assert secret_get_or_create.environment_name == "custom"

    secret_get_or_create_2 = ctx.pop_request("SecretGetOrCreate")
    assert secret_get_or_create_2.environment_name == "third"
