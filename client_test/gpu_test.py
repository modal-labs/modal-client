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
        stub.function(dummy, gpu=True)


def test_gpu_any_function(client, servicer):
    stub = Stub()

    stub.function(dummy, gpu="any")
    with stub.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu == 0
    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_ANY


def test_gpu_string_config(client, servicer):
    stub = Stub()

    # Invalid enum value.
    with pytest.raises(InvalidError):
        stub.function(dummy, gpu="foo")

    stub.function(dummy, gpu="A100")
    with stub.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu == 0
    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_A100


def test_gpu_config_function(client, servicer):
    import modal

    stub = Stub()

    stub.function(dummy, gpu=modal.gpu.A100())
    with stub.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.resources.gpu == 0
    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_A100


def test_cloud_provider_selection(client, servicer):
    import modal

    stub = Stub()

    stub.function(dummy, gpu=modal.gpu.A100(), cloud="gcp")
    with stub.run(client=client):
        pass

    assert len(servicer.app_functions) == 1
    func_def = next(iter(servicer.app_functions.values()))
    assert func_def.cloud_provider == api_pb2.CLOUD_PROVIDER_GCP

    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == api_pb2.GPU_TYPE_A100

    # Invalid enum value.
    with pytest.raises(InvalidError):
        stub.function(dummy, cloud="foo")


A100_GPU_MEMORY_MAPPING = {0: api_pb2.GPU_TYPE_A100, 20: api_pb2.GPU_TYPE_A100_20G, 40: api_pb2.GPU_TYPE_A100}


@pytest.mark.parametrize("memory,gpu_type", A100_GPU_MEMORY_MAPPING.items())
def test_memory_selection_gpu_variant(client, servicer, memory, gpu_type):
    import modal

    stub = Stub()

    stub.function(dummy, gpu=modal.gpu.A100(memory=memory))
    with stub.run(client=client):
        pass

    func_def = next(iter(servicer.app_functions.values()))

    assert func_def.resources.gpu_config.count == 1
    assert func_def.resources.gpu_config.type == gpu_type
    assert func_def.resources.gpu_config.memory == memory


A100_GPU_COUNT_MAPPING = {1: api_pb2.GPU_TYPE_A100, **{i: api_pb2.GPU_TYPE_A100_40GB_MANY for i in range(2, 5)}}


@pytest.mark.parametrize("count,gpu_type", A100_GPU_COUNT_MAPPING.items())
def test_gpu_type_selection_from_count(client, servicer, count, gpu_type):
    import modal

    stub = Stub()

    # Functions that use A100 20GB can only request one GPU
    # at a time.
    with pytest.raises(ValueError):
        stub.function(dummy, gpu=modal.gpu.A100(count=2, memory=20))
        with stub.run(client=client):
            pass

    # Task type changes whenever user asks more than 1 GPU on
    # an A100.
    stub.function(dummy, gpu=modal.gpu.A100(count=count))
    with stub.run(client=client):
        pass

    func_def = next(iter(servicer.app_functions.values()))

    assert func_def.resources.gpu_config.count == count
    assert func_def.resources.gpu_config.type == gpu_type
