# Copyright Modal Labs 2025
import pytest
from datetime import datetime, timezone

import modal
import modal.experimental
from modal.exception import NotFoundError
from modal_proto import api_pb2

app = modal.App(include_source=False)


@app.function()
def f():
    pass


@app.cls()
class C:
    @modal.method()
    def method(self):
        pass


@app._experimental_server(port=8000, routing_region="us-east", serialized=True)
class SimpleServer:
    @modal.enter()
    def start(self):
        pass


def test_app_get_objects(client, servicer):
    app.deploy(name="test", environment_name="dev", client=client)
    res = modal.experimental.get_app_objects("test", environment_name="dev", client=client)
    assert len(res) == 3
    assert res.keys() == {"C", "SimpleServer", "f"}
    assert isinstance(res["C"], modal.Cls)
    assert isinstance(res["f"], modal.Function)
    assert isinstance(res["SimpleServer"], modal.Function)


def test_image_delete(client, servicer):
    with app.run(client=client):
        image = modal.Image.debian_slim().build(app)

    assert image.object_id in servicer.images
    modal.experimental.image_delete(image.object_id, client=client)
    assert image.object_id not in servicer.images

    with pytest.raises(NotFoundError):
        modal.experimental.image_delete("im-nonexistent", client=client)


def test_stop_app(client, servicer):
    app_name, environment_name = "test", "dev"
    app.deploy(name=app_name, environment_name=environment_name, client=client)
    modal.experimental.stop_app(app_name, environment_name=environment_name, client=client)
    with pytest.raises(NotFoundError):
        modal.App.lookup(app_name, environment_name=environment_name, client=client)


def test_get_app_lifecycle(client, servicer):
    with servicer.intercept() as ctx:
        ctx.add_response(
            "AppGetLifecycle",
            api_pb2.AppGetLifecycleResponse(
                lifecycle=api_pb2.AppLifecycle(
                    created_at=1000.0,
                    created_by="alice",
                    deployed_at=2000.0,
                    deployed_by="bob",
                    version=3,
                    stopped_at=3000.0,
                    stopped_by="carol",
                )
            ),
        )
        lifecycle = modal.experimental.get_app_lifecycle("ap-123", client=client)

    assert ctx.pop_request("AppGetLifecycle").app_id == "ap-123"
    assert lifecycle.created_at == datetime.fromtimestamp(1000.0, timezone.utc)
    assert lifecycle.created_by == "alice"
    assert lifecycle.deployed_at == datetime.fromtimestamp(2000.0, timezone.utc)
    assert lifecycle.deployed_by == "bob"
    assert lifecycle.stopped_at == datetime.fromtimestamp(3000.0, timezone.utc)
    assert lifecycle.stopped_by == "carol"


def test_get_app_lifecycle_running(client, servicer):
    # A running App has no stop event, and an ephemeral App was never deployed; the server
    # returns 0 timestamps and empty strings, which should surface as None.
    with servicer.intercept() as ctx:
        ctx.add_response(
            "AppGetLifecycle",
            api_pb2.AppGetLifecycleResponse(lifecycle=api_pb2.AppLifecycle(created_at=1000.0, created_by="alice")),
        )
        lifecycle = modal.experimental.get_app_lifecycle("ap-123", client=client)

    assert lifecycle.created_at == datetime.fromtimestamp(1000.0, timezone.utc)
    assert lifecycle.deployed_at is None
    assert lifecycle.deployed_by is None
    assert lifecycle.stopped_at is None
    assert lifecycle.stopped_by is None
