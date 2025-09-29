# Copyright Modal Labs 2025
import pytest
from typing import Union, cast

import modal
import modal.experimental
from modal.exception import DeprecationError, NotFoundError

app = modal.App(include_source=False)


@app.function()
def f():
    pass


@app.cls()
class C:
    @modal.method()
    def method(self):
        pass


@pytest.mark.parametrize("which", ["function", "cls"])
def test_update_autoscaler(client, servicer, which):
    overrides = {
        "min_containers": 1,
        "max_containers": 5,
        "buffer_containers": 2,
        "scaledown_window": 10,
    }

    with app.run(client=client):
        obj: Union[modal.Function, modal.cls.Obj]
        # Hardcode the object ID based on what we expect from the mock servicer
        # which is pretty janky, but avoids hydrating the object in the test so that
        # we can properly assert that update_autoscaler handles unhydrated objects correctly.
        if which == "function":
            obj, obj_id = f, "fu-1"
        else:
            obj, obj_id = cast(modal.cls.Obj, C()), "fu-2"

        with pytest.warns(DeprecationError):
            modal.experimental.update_autoscaler(obj, client=client, **overrides)  # type: ignore

        settings = servicer.app_functions[obj_id].autoscaler_settings  # type: ignore
        assert settings.min_containers == overrides["min_containers"]
        assert settings.max_containers == overrides["max_containers"]
        assert settings.buffer_containers == overrides["buffer_containers"]
        assert settings.scaledown_window == overrides["scaledown_window"]


@pytest.mark.parametrize("which", ["function", "cls"])
def test_update_autoscaler_after_lookup(client, servicer, which):
    app.deploy(name="test", client=client)

    overrides = {
        "min_containers": 1,
        "max_containers": 5,
        "buffer_containers": 2,
        "scaledown_window": 10,
    }

    # See above for why we're hardcoding the object IDs.
    # Would be much nicer to be able to look up the internal definition by the *name*...
    obj: Union[modal.Function, modal.cls.Obj]
    if which == "function":
        obj = modal.Function.from_name("test", "f")
        obj_id = "fu-1"
    else:
        C = modal.Cls.from_name("test", "C")
        obj = C()
        obj_id = "fu-2"

    with pytest.warns(DeprecationError):
        modal.experimental.update_autoscaler(obj, client=client, **overrides)  # type: ignore

    settings = servicer.app_functions[obj_id].autoscaler_settings  # type: ignore
    assert settings.min_containers == overrides["min_containers"]
    assert settings.max_containers == overrides["max_containers"]
    assert settings.buffer_containers == overrides["buffer_containers"]
    assert settings.scaledown_window == overrides["scaledown_window"]


def test_app_get_objects(client, servicer):
    app.deploy(name="test", environment_name="dev", client=client)
    res = modal.experimental.get_app_objects("test", environment_name="dev", client=client)
    assert res.keys() == {"C", "f"}
    assert isinstance(res["C"], modal.Cls)
    assert isinstance(res["f"], modal.Function)


def test_image_delete(client, servicer):
    with app.run(client=client):
        image = modal.Image.debian_slim().build(app)

    assert image.object_id in servicer.images
    modal.experimental.image_delete(image.object_id, client=client)
    assert image.object_id not in servicer.images

    with pytest.raises(NotFoundError):
        modal.experimental.image_delete("im-nonexistent", client=client)
