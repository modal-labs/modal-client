# Copyright Modal Labs 2025
import pytest

import modal
import modal.experimental
from modal.exception import NotFoundError

app = modal.App(include_source=False)


@app.function()
def f():
    pass


@app.cls()
class C:
    @modal.method()
    def method(self):
        pass


@app.server(port=8000, proxy_regions=["us-east"], serialized=True)
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
