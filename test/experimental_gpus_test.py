# Copyright Modal Labs 2024
import modal
from modal import App, Sandbox
from modal_proto import api_pb2

from .sandbox_test import skip_non_linux

app = App()


@app.function(_experimental_gpus=["a10g"])
def f1():
    pass


@app.function(_experimental_gpus=["a10g", "t4:2"])
def f2():
    pass


@app.function(_experimental_gpus=["h100:2", modal.gpu.A100(count=2, size="80GB")])
def f3():
    pass


def test_experimental_resources(servicer, client):
    with app.run(client=client):
        assert len(servicer.app_functions) == 3

        a10_1 = api_pb2.Resources(
            gpu_config=api_pb2.GPUConfig(
                type=api_pb2.GPU_TYPE_A10G,
                count=1,
            )
        )
        t4_2 = api_pb2.Resources(
            gpu_config=api_pb2.GPUConfig(
                type=api_pb2.GPU_TYPE_T4,
                count=2,
            )
        )
        h100_2 = api_pb2.Resources(
            gpu_config=api_pb2.GPUConfig(
                type=api_pb2.GPU_TYPE_H100,
                count=2,
            )
        )
        a100_80gb_2 = api_pb2.Resources(
            gpu_config=api_pb2.GPUConfig(
                type=api_pb2.GPU_TYPE_A100_80GB,
                count=2,
                memory=80,
            )
        )

        fn1 = servicer.app_functions["fu-1"]  # f1
        assert len(fn1._experimental_resources) == 1
        assert fn1._experimental_resources[0].gpu_config.type == a10_1.gpu_config.type
        assert fn1._experimental_resources[0].gpu_config.count == a10_1.gpu_config.count

        fn2 = servicer.app_functions["fu-2"]  # f2
        assert len(fn2._experimental_resources) == 2
        assert fn2._experimental_resources[0].gpu_config.type == a10_1.gpu_config.type
        assert fn2._experimental_resources[0].gpu_config.count == a10_1.gpu_config.count
        assert fn2._experimental_resources[1].gpu_config.type == t4_2.gpu_config.type
        assert fn2._experimental_resources[1].gpu_config.count == t4_2.gpu_config.count

        fn3 = servicer.app_functions["fu-3"]  # f3
        assert len(fn3._experimental_resources) == 2
        assert fn3._experimental_resources[0].gpu_config.type == h100_2.gpu_config.type
        assert fn3._experimental_resources[0].gpu_config.count == h100_2.gpu_config.count
        assert fn3._experimental_resources[1].gpu_config.type == a100_80gb_2.gpu_config.type
        assert fn3._experimental_resources[1].gpu_config.count == a100_80gb_2.gpu_config.count
        assert fn3._experimental_resources[1].gpu_config.memory == a100_80gb_2.gpu_config.memory


@skip_non_linux
def test_sandbox_experimental_resources(client, servicer):
    Sandbox.create(
        "bash",
        "-c",
        "echo bye >&2 && sleep 1 && echo hi && exit 42",
        timeout=600,
        _experimental_gpus=["a10g:2", "t4:4"],
        client=client,
    )

    a10_2 = api_pb2.Resources(
        gpu_config=api_pb2.GPUConfig(
            type=api_pb2.GPU_TYPE_A10G,
            count=2,
        )
    )
    t4_4 = api_pb2.Resources(
        gpu_config=api_pb2.GPUConfig(
            type=api_pb2.GPU_TYPE_T4,
            count=4,
        )
    )

    assert len(servicer.sandbox_defs) == 1
    sb_def = servicer.sandbox_defs[0]
    assert len(sb_def._experimental_resources) == 2
    assert sb_def._experimental_resources[0].gpu_config.type == a10_2.gpu_config.type
    assert sb_def._experimental_resources[0].gpu_config.count == a10_2.gpu_config.count
    assert sb_def._experimental_resources[1].gpu_config.type == t4_4.gpu_config.type
    assert sb_def._experimental_resources[1].gpu_config.count == t4_4.gpu_config.count
