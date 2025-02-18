# Copyright Modal Labs 2024
import modal
import modal.experimental
from modal import App

app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


@app.function()
@modal.experimental.clustered(size=2)
def f1():
    pass


@app.function()
def f2():
    pass


@app.function(i6pn=True)
def f3():
    pass


def test_experimental_cluster(servicer, client):
    with app.run(client=client):
        assert len(servicer.app_functions) == 3

        fn1 = servicer.app_functions["fu-1"]  # f1
        assert fn1._experimental_group_size == 2
        assert fn1.i6pn_enabled is True

        fn2 = servicer.app_functions["fu-2"]  # f2
        assert not fn2._experimental_group_size
        assert fn2.i6pn_enabled is False

        fn3 = servicer.app_functions["fu-3"]  # f3
        assert not fn3._experimental_group_size
        assert fn3.i6pn_enabled is True


def test_run_experimental_cluster(client, servicer, monkeypatch):
    with app.run(client=client):
        # The servicer returns the sum of the squares of all arguments
        assert f1.remote(2, 4) == 2**2 + 4**2
