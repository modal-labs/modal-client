# Copyright Modal Labs 2024
from modal import App, SchedulerPlacement
from modal_proto import api_pb2

from .sandbox_test import skip_non_linux

app = App()


@app.function(
    _experimental_scheduler=True,
    _experimental_scheduler_placement=SchedulerPlacement(
        region="us-east-1",
        zone="us-east-1a",
        spot=False,
    ),
)
def f():
    pass


def test_fn_scheduler_placement(servicer, client):
    with app.run(client=client):
        assert len(servicer.app_functions) == 1
        fn = servicer.app_functions["fu-1"]
        assert fn._experimental_scheduler
        assert fn._experimental_scheduler_placement == api_pb2.SchedulerPlacement(
            _region="us-east-1",
            _zone="us-east-1a",
            _lifecycle="on-demand",
        )


@skip_non_linux
def test_sandbox_scheduler_placement(client, servicer):
    with app.run(client=client):
        _ = app.spawn_sandbox(
            "bash",
            "-c",
            "echo bye >&2 && sleep 1 && echo hi && exit 42",
            timeout=600,
            _experimental_scheduler=True,
            _experimental_scheduler_placement=SchedulerPlacement(
                region="us-east-1",
                zone="us-east-1a",
                spot=False,
            ),
        )

        assert len(servicer.sandbox_defs) == 1
        sb_def = servicer.sandbox_defs[0]
        assert sb_def._experimental_scheduler
        assert sb_def._experimental_scheduler_placement == api_pb2.SchedulerPlacement(
            _region="us-east-1",
            _zone="us-east-1a",
            _lifecycle="on-demand",
        )
