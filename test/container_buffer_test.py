# Copyright Modal Labs 2024
from modal import App

app = App()


@app.function(
    _experimental_buffer_containers=10,
)
def f1():
    pass


def test_fn_container_buffer(servicer, client):
    with app.run(client=client):
        assert len(servicer.app_functions) == 1
        fn1 = servicer.app_functions["fu-1"]  # f1
        assert fn1._experimental_buffer_containers == 10
