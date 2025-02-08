# Copyright Modal Labs 2024
from modal import App
from modal_proto import api_pb2

app = App()


@app.function(gpu=["a10g"])
def f1():
    pass


@app.function(gpu=["a10g", "t4:2"])
def f2():
    pass


@app.function(gpu=["h100:2", "a100-80gb:2"])
def f3():
    pass


def test_gpu_fallback(servicer, client):
    with app.run(client=client):
        assert len(servicer.app_functions) == 3

        a10_1 = api_pb2.Resources(
            gpu_config=api_pb2.GPUConfig(
                gpu_type="A10G",
                count=1,
            )
        )
        t4_2 = api_pb2.Resources(
            gpu_config=api_pb2.GPUConfig(
                gpu_type="T4",
                count=2,
            )
        )
        h100_2 = api_pb2.Resources(
            gpu_config=api_pb2.GPUConfig(
                gpu_type="H100",
                count=2,
            )
        )
        a100_80gb_2 = api_pb2.Resources(
            gpu_config=api_pb2.GPUConfig(type=api_pb2.GPU_TYPE_A100_80GB, count=2, gpu_type="A100-80GB")
        )

        fn1 = servicer.app_functions["fu-1"]  # f1
        assert len(fn1.ranked_functions) == 1
        assert fn1.ranked_functions[0].function.resources.gpu_config.gpu_type == a10_1.gpu_config.gpu_type
        assert fn1.ranked_functions[0].function.resources.gpu_config.count == a10_1.gpu_config.count

        fn2 = servicer.app_functions["fu-2"]  # f2
        assert len(fn2.ranked_functions) == 2
        assert fn2.ranked_functions[0].function.resources.gpu_config.gpu_type == a10_1.gpu_config.gpu_type
        assert fn2.ranked_functions[0].function.resources.gpu_config.count == a10_1.gpu_config.count
        assert fn2.ranked_functions[1].function.resources.gpu_config.gpu_type == t4_2.gpu_config.gpu_type
        assert fn2.ranked_functions[1].function.resources.gpu_config.count == t4_2.gpu_config.count

        fn3 = servicer.app_functions["fu-3"]  # f3
        assert len(fn3.ranked_functions) == 2
        assert fn3.ranked_functions[0].function.resources.gpu_config.gpu_type == h100_2.gpu_config.gpu_type
        assert fn3.ranked_functions[0].function.resources.gpu_config.count == h100_2.gpu_config.count
        assert fn3.ranked_functions[1].function.resources.gpu_config.gpu_type == a100_80gb_2.gpu_config.gpu_type
        assert fn3.ranked_functions[1].function.resources.gpu_config.count == a100_80gb_2.gpu_config.count
        assert fn3.ranked_functions[1].function.resources.gpu_config.gpu_type == a100_80gb_2.gpu_config.gpu_type
