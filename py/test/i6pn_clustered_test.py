# Copyright Modal Labs 2024
import pytest

import modal.experimental
from modal import App
from modal.exception import InvalidError

app = App(include_source=False)


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


@pytest.mark.parametrize("size", [0, -1, 1.5])
def test_cluster_size_validation(size):
    with pytest.raises(InvalidError, match="positive integer"):
        modal.experimental.clustered(size=size)


@pytest.mark.parametrize("cluster_first", [True, False])
def test_cluster_batching_rejected(cluster_first):
    def batched_fn(xs: list[int]) -> list[int]:
        return xs

    decorators = [modal.experimental.clustered(size=2), modal.batched(max_batch_size=2, wait_ms=10)]
    if not cluster_first:
        decorators.reverse()
    wrapped = decorators[0](decorators[1](batched_fn))
    with pytest.raises(InvalidError, match="dynamic batching"):
        App(include_source=False).function(serialized=True)(wrapped)
