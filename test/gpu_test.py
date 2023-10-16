# Copyright Modal Labs 2022
import pytest

from modal import Stub
from modal.exception import DeprecationError, InvalidError
from modal_proto import api_pb2


def dummy():
    pass  # not actually used in test (servicer returns sum of square of all args)


def test_gpu_true_function(client, servicer):
    stub = Stub()

    with pytest.raises(DeprecationError):
        stub.function(gpu=True)(dummy)


def test_gpu_any_function(client, servicer):
    stub = Stub()

    stub.function(gpu="any")(dummy)
    with stub.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_ANY


def test_gpu_string_config(client, servicer):
    stub = Stub()

    # Invalid enum value.
    with pytest.raises(InvalidError):
        stub.function(gpu="foo")(dummy)

    stub.function(gpu="A100")(dummy)
    with stub.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_A100


def test_gpu_string_count_config(client, servicer):
    stub = Stub()

    # Invalid count values.
    with pytest.raises(InvalidError):
        stub.function(gpu="A10G:hello")(dummy)
    with pytest.raises(InvalidError):
        stub.function(gpu="Nonexistent:2")(dummy)

    stub.function(gpu="A10G:4")(dummy)
    with stub.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu_config.count == 4
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_A10G


def test_gpu_config_function(client, servicer):
    import modal

    stub = Stub()

    stub.function(gpu=modal.gpu.A100())(dummy)
    with stub.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_A100


def test_cloud_provider_selection(client, servicer):
    import modal

    stub = Stub()

    stub.function(gpu=modal.gpu.A100(), cloud="gcp")(dummy)
    with stub.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.cloud_provider == api_pb2.CLOUD_PROVIDER_GCP

    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_A100

    # Invalid enum value.
    with pytest.raises(InvalidError):
        stub.function(cloud="foo")(dummy)


A100_GPU_MEMORY_MAPPING = {0: api_pb2.GPU_TYPE_A100, 40: api_pb2.GPU_TYPE_A100}


@pytest.mark.parametrize("memory,gpu_type", A100_GPU_MEMORY_MAPPING.items())
def test_memory_selection_gpu_variant(client, servicer, memory, gpu_type):
    import modal

    stub = Stub()
    stub.function(gpu=modal.gpu.A100(memory=memory))(dummy)

    with stub.run(client=client):
        pass

    func_def = next(iter(servicer.app_functions.values()))

    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == gpu_type
    assert func_def.resources.gpu_config.memory == memory


def test_a100_20gb_gpu_unsupported():
    import modal

    stub = Stub()

    with pytest.raises(ValueError) as err:
        stub.function(gpu=modal.gpu.A100(memory=20))(dummy)
    assert err.value.args == ("A100 20GB is unsupported, consider A10 or A100 40GB instead",)


A100_GPU_COUNT_MAPPING = {1: api_pb2.GPU_TYPE_A100, **{i: api_pb2.GPU_TYPE_A100 for i in range(2, 5)}}


@pytest.mark.parametrize("count,gpu_type", A100_GPU_COUNT_MAPPING.items())
def test_gpu_type_selection_from_count(client, servicer, count, gpu_type):
    import modal

    stub = Stub()

    # Functions that use A100 20GB can only request one GPU
    # at a time.
    with pytest.raises(ValueError):
        with pytest.warns(DeprecationError):
            stub.function(gpu=modal.gpu.A100(count=2, memory=20))(dummy)
        with stub.run(client=client):
            pass

    # Task type changes whenever user asks more than 1 GPU on
    # an A100.
    stub.function(gpu=modal.gpu.A100(count=count))(dummy)
    with stub.run(client=client):
        pass

    func_def = next(iter(servicer.app_functions.values()))

    assert func_def.resources.gpu_config.count == count
    assert func_def.resources.gpu_config.type == gpu_type
