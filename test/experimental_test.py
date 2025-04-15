# Copyright Modal Labs 2025
import pytest
from typing import Union, cast

import modal
import modal.experimental

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
        if which == "function":
            obj = f
            obj_id = obj.object_id
        else:
            obj = cast(modal.cls.Obj, C())
            # This is ugly
            obj_id = obj._cached_service_function().object_id  # type: ignore

        modal.experimental.update_autoscaler(obj, client=client, **overrides)  # type: ignore

        settings = servicer.app_functions[obj_id].autoscaler_settings  # type: ignore
        assert settings.min_containers == overrides["min_containers"]
        assert settings.max_containers == overrides["max_containers"]
        assert settings.buffer_containers == overrides["buffer_containers"]
        assert settings.scaledown_window == overrides["scaledown_window"]
