# Copyright Modal Labs 2022
import pytest

import modal.gpu
from modal import App
from modal.exception import InvalidError
from modal_proto import api_pb2


def dummy():
    pass  # not actually used in test (servicer returns sum of square of all args)


def test_gpu_any_function(client, servicer):
    app = App()

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
    app = App()

    app.function(gpu=gpu_arg)(dummy)
    with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu_config.gpu_type == gpu_type
    assert func_def.resources.gpu_config.count == count


@pytest.mark.parametrize("gpu_arg", ["foo", "a10g:hello", "nonexistent:2"])
def test_invalid_gpu_string_config(client, servicer, gpu_arg):
    app = App()

    # Invalid enum value.
    with pytest.raises(InvalidError):
        app.function(gpu=gpu_arg)(dummy)
        with app.run(client=client):
            pass


def test_gpu_config_function(client, servicer):
    app = App()

    with pytest.warns(match='gpu="A100-40GB"'):
        app.function(gpu=modal.gpu.A100())(dummy)
    with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu_config.count == 1


def test_gpu_config_function_more(client, servicer):
    # Make sure some other GPU types also throw warnings
    with pytest.warns(match='gpu="A100-80GB"'):
        modal.gpu.A100(size="80GB")
    with pytest.warns(match='gpu="T4:7"'):
        modal.gpu.T4(count=7)


def test_cloud_provider_selection(client, servicer):
    app = App()

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
    app = App()

    # Invalid enum value.
    with pytest.raises(InvalidError):
        app.function(cloud="foo")(dummy)
        with app.run(client=client):
            pass


@pytest.mark.parametrize(
    "memory_arg,gpu_type",
    [
        ("40GB", "A100-40GB"),
        ("80GB", "A100-80GB"),
    ],
)
def test_memory_selection_gpu_variant(client, servicer, memory_arg, gpu_type):
    app = App()
    with pytest.warns(match='gpu="A100'):
        app.function(gpu=modal.gpu.A100(size=memory_arg))(dummy)

    with app.run(client=client):
        pass

    func_def = next(iter(servicer.app_functions.values()))

    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.gpu_type == gpu_type


def test_gpu_unsupported_config():
    app = App()

    with pytest.raises(ValueError, match="size='20GB' is invalid"):
        app.function(gpu=modal.gpu.A100(size="20GB"))(dummy)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_gpu_type_selection_from_count(client, servicer, count):
    app = App()

    # Task type does not change when user asks more than 1 GPU on an A100.
    app.function(gpu=f"A100:{count}")(dummy)
    with app.run(client=client):
        pass

    func_def = next(iter(servicer.app_functions.values()))

    assert func_def.resources.gpu_config.count == count
