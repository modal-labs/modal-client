# Copyright Modal Labs 2022
import pytest

from modal import App
from modal.exception import InvalidError
from modal_proto import api_pb2


def dummy():
    pass  # not actually used in test (servicer returns sum of square of all args)


def test_gpu_any_function(client, servicer):
    app = App(include_source=False)

    app.function(gpu="any")(dummy)
    with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu_config.count == 1


@pytest.mark.parametrize(
    "gpu_arg,gpu_type,count",
    [
        ("A100-40GB", "A100-40GB", 1),
        ("a100-40gb", "A100-40GB", 1),
        ("a10g", "A10G", 1),
        ("t4:7", "T4", 7),
        ("a100-80GB:5", "A100-80GB", 5),
        ("l40s:2", "L40S", 2),
    ],
)
def test_gpu_string_config(client, servicer, gpu_arg, gpu_type, count):
    app = App(include_source=False)

    app.function(gpu=gpu_arg)(dummy)
    with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu_config.gpu_type == gpu_type
    assert func_def.resources.gpu_config.count == count


@pytest.mark.parametrize("gpu_arg", ["foo", "a10g:hello", "nonexistent:2"])
def test_invalid_gpu_string_config(client, servicer, gpu_arg):
    app = App(include_source=False)

    # Invalid enum value.
    with pytest.raises(InvalidError):
        app.function(gpu=gpu_arg)(dummy)
        with app.run(client=client):
            pass


def test_cloud_provider_selection(client, servicer):
    app = App(include_source=False)

    app.function(gpu="A100", cloud="gcp")(dummy)
    with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.cloud_provider == api_pb2.CLOUD_PROVIDER_UNSPECIFIED  # No longer set
    assert func_def.cloud_provider_str == "gcp"

    assert func_def.resources.gpu_config.gpu_type == "A100"
    assert func_def.resources.gpu_config.count == 1


def test_invalid_cloud_provider_selection(client, servicer):
    app = App(include_source=False)

    # Invalid enum value.
    with pytest.raises(InvalidError):
        app.function(cloud="foo")(dummy)
        with app.run(client=client):
            pass


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_gpu_type_selection_from_count(client, servicer, count):
    app = App(include_source=False)

    # Task type does not change when user asks more than 1 GPU on an A100.
    app.function(gpu=f"A100:{count}")(dummy)
    with app.run(client=client):
        pass

    func_def = next(iter(servicer.app_functions.values()))

    assert func_def.resources.gpu_config.count == count
