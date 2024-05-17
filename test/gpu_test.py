# Copyright Modal Labs 2022
import pytest

from modal import App
from modal.exception import DeprecationError, InvalidError
from modal_proto import api_pb2


def dummy():
    pass  # not actually used in test (servicer returns sum of square of all args)


def test_gpu_true_function(client, servicer):
    app = App()

    with pytest.raises(DeprecationError):
        app.function(gpu=True)(dummy)


def test_gpu_any_function(client, servicer):
    app = App()

    app.function(gpu="any")(dummy)
    with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_ANY


def test_gpu_string_config(client, servicer):
    app = App()

    # Invalid enum value.
    with pytest.raises(InvalidError):
        app.function(gpu="foo")(dummy)

    app.function(gpu="A100")(dummy)
    with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_A100


def test_gpu_string_count_config(client, servicer):
    app = App()

    # Invalid count values.
    with pytest.raises(InvalidError):
        app.function(gpu="A10G:hello")(dummy)
    with pytest.raises(InvalidError):
        app.function(gpu="Nonexistent:2")(dummy)

    app.function(gpu="A10G:4")(dummy)
    with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu_config.count == 4
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_A10G


def test_gpu_config_function(client, servicer):
    import modal

    app = App()

    app.function(gpu=modal.gpu.A100())(dummy)
    with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_A100


def test_cloud_provider_selection(client, servicer):
    import modal

    app = App()

    app.function(gpu=modal.gpu.A100(), cloud="gcp")(dummy)
    with app.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.cloud_provider == api_pb2.CLOUD_PROVIDER_GCP

    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_A100

    # Invalid enum value.
    with pytest.raises(InvalidError):
        app.function(cloud="foo")(dummy)


@pytest.mark.parametrize(
    "memory_arg,gpu_type,memory_gb",
    [
        ("40GB", api_pb2.GPU_TYPE_A100, 40),
        ("80GB", api_pb2.GPU_TYPE_A100_80GB, 80),
    ],
)
def test_memory_selection_gpu_variant(client, servicer, memory_arg, gpu_type, memory_gb):
    import modal

    app = App()
    if isinstance(memory_arg, str):
        app.function(gpu=modal.gpu.A100(size=memory_arg))(dummy)
    else:
        raise RuntimeError(f"Unexpected test parameterization arg type {type(memory_arg)}")

    with app.run(client=client):
        pass

    func_def = next(iter(servicer.app_functions.values()))

    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == gpu_type
    assert func_def.resources.gpu_config.memory == memory_gb


def test_gpu_unsupported_config():
    import modal

    app = App()

    with pytest.raises(ValueError, match="size='20GB' is invalid"):
        app.function(gpu=modal.gpu.A100(size="20GB"))(dummy)

    with pytest.warns(match="size='80GB'"):
        app.function(gpu=modal.gpu.A100(memory=80))(dummy)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_gpu_type_selection_from_count(client, servicer, count):
    import modal

    app = App()

    # Task type does not change when user asks more than 1 GPU on an A100.
    app.function(gpu=modal.gpu.A100(count=count))(dummy)
    with app.run(client=client):
        pass

    func_def = next(iter(servicer.app_functions.values()))

    assert func_def.resources.gpu_config.count == count
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_A100
