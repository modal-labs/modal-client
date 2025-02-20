# Copyright Modal Labs 2024
from modal import App

app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


@app.function(
    buffer_containers=10,
)
def f1():
    pass


def test_fn_container_buffer(servicer, client):
    with app.run(client=client):
        assert len(servicer.app_functions) == 1
        fn1 = servicer.app_functions["fu-1"]  # f1
        # Test forward / backward compatibility
        assert fn1._experimental_buffer_containers == fn1.autoscaler_settings.buffer_containers == 10
