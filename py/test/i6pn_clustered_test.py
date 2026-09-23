# Copyright Modal Labs 2024
import pytest

import modal.experimental
from modal import App
from modal.exception import DeprecationError, InvalidError

app = App(include_source=False)


@app.function()
@modal.clustered(size=2)
def f1(x: int, y: int):
    pass


@app.function()
def f2():
    pass


@app.function(i6pn=True)
def f3():
    pass


def test_cluster(servicer, client):
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


def test_run_cluster(client, servicer, monkeypatch):
    with app.run(client=client):
        # The servicer returns the sum of the squares of all arguments
        assert f1.remote(2, 4) == 2**2 + 4**2


def test_experimental_clustered_compatibility():
    with pytest.warns(DeprecationError, match="Use `modal.clustered"):
        wrapper = modal.experimental.clustered(2, True, True)
    test_app = App(include_source=False)
    fn = test_app.function(serialized=True)(wrapper(lambda: None))
    assert fn is not None
    with pytest.warns(DeprecationError), pytest.raises(AssertionError, match="broadcast=False"):
        modal.experimental.clustered(2, False)


@pytest.mark.parametrize("size", [0, -1, 1.5])
def test_cluster_size_validation(size):
    with pytest.raises(InvalidError, match="positive integer"):
        modal.clustered(size=size)


@pytest.mark.parametrize("fabric_size", [0, -1, True, 1.5, 3])
def test_cluster_fabric_validation(fabric_size):
    with pytest.raises(InvalidError, match="fabric_size"):
        App(include_source=False).function(serialized=True, experimental_options={"fabric_size": fabric_size})(
            modal.clustered(size=4)(lambda: None)
        )


@pytest.mark.parametrize("kind", ["function", "cls", "server", "legacy"])
def test_cluster_fabric_option(client, servicer, kind):
    test_app = App(include_source=False)
    options = {"fabric_size": 2, "other_option": "value"}
    if kind == "legacy":
        with pytest.warns(DeprecationError):
            decorator = modal.experimental.clustered(4, True, False, 2)
        test_app.function(serialized=True)(decorator(lambda: None))
    elif kind == "function":
        test_app.function(serialized=True, experimental_options=options)(modal.clustered(size=4)(lambda: None))
    elif kind == "cls":

        @test_app.cls(serialized=True, experimental_options=options)
        @modal.clustered(size=4)
        class Service:
            @modal.method()
            def run(self):
                pass
    else:

        @test_app.server(serialized=True, experimental_options=options)
        @modal.clustered(size=4)
        class Server:
            @modal.enter()
            def start(self):
                pass

    with test_app.run(client=client):
        definitions = list(servicer.app_functions.values())
        assert definitions
        for definition in definitions:
            assert definition._experimental_group_size == 4
            assert definition._experimental_fabric_size == 2
            assert "fabric_size" not in definition.experimental_options
            if kind != "legacy":
                assert definition.experimental_options["other_option"] == "value"
    assert options == {"fabric_size": 2, "other_option": "value"}


def test_cluster_fabric_requires_cluster():
    with pytest.raises(InvalidError, match="requires @modal.clustered"):
        App(include_source=False).function(serialized=True, experimental_options={"fabric_size": 2})(lambda: None)


def test_cluster_fabric_duplicate():
    with pytest.warns(DeprecationError):
        wrapped = modal.experimental.clustered(4, fabric_size=2)(lambda: None)
    with pytest.raises(InvalidError, match="only once"):
        App(include_source=False).function(serialized=True, experimental_options={"fabric_size": 2})(wrapped)


def test_cluster_public_signature():
    with pytest.raises(TypeError):
        modal.clustered(size=4, fabric_size=2)  # type: ignore
    with pytest.raises(TypeError):
        modal.clustered(2)  # type: ignore
    with pytest.raises(TypeError):
        modal.clustered(size=2, broadcast=True)  # type: ignore


@pytest.mark.parametrize("cluster_first", [True, False])
def test_cluster_batching_rejected(cluster_first):
    def batched_fn(xs: list[int]) -> list[int]:
        return xs

    clustered = modal.clustered(size=2)
    batched = modal.batched(max_batch_size=2, wait_ms=10)
    if cluster_first:
        wrapped = clustered(batched(batched_fn))
    else:
        wrapped = batched(clustered(batched_fn))
    with pytest.raises(InvalidError, match="dynamic batching"):
        App(include_source=False).function(serialized=True)(wrapped)
