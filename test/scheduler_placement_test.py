# Copyright Modal Labs 2024
from modal import App, SchedulerPlacement
from modal_proto import api_pb2

from .sandbox_test import skip_non_linux

app = App()


@app.function(
    _experimental_scheduler_placement=SchedulerPlacement(
        region="us-east-1",
        zone="us-east-1a",
        spot=False,
    ),
)
def f1():
    pass


@app.function(
    region="us-east-1",
)
def f2():
    pass


@app.function(
    region=["us-east-1", "us-west-2"],
)
def f3():
    pass


def test_fn_scheduler_placement(servicer, client):
    with app.run(client=client):
        assert len(servicer.app_functions) == 3
        fn1 = servicer.app_functions["fu-1"]  # f1
        assert fn1.scheduler_placement == api_pb2.SchedulerPlacement(
            regions=["us-east-1"],
            _zone="us-east-1a",
            _lifecycle="on-demand",
        )

        fn2 = servicer.app_functions["fu-2"]  # f2
        assert fn2.scheduler_placement == api_pb2.SchedulerPlacement(
            regions=["us-east-1"],
        )

        fn3 = servicer.app_functions["fu-3"]  # f3
        assert fn3.scheduler_placement == api_pb2.SchedulerPlacement(
            regions=["us-east-1", "us-west-2"],
        )


@skip_non_linux
def test_sandbox_scheduler_placement(client, servicer):
    with app.run(client=client):
        _ = app.spawn_sandbox(
            "bash",
            "-c",
            "echo bye >&2 && sleep 1 && echo hi && exit 42",
            timeout=600,
            region="us-east-1",
        )

        assert len(servicer.sandbox_defs) == 1
        sb_def = servicer.sandbox_defs[0]
        assert sb_def.scheduler_placement == api_pb2.SchedulerPlacement(
            regions=["us-east-1"],
        )
