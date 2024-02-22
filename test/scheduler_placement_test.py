# Copyright Modal Labs 2024
from modal import SchedulerPlacement, Stub
from modal_proto import api_pb2

stub = Stub()


@stub.function(
    _experimental_scheduler=True,
    _experimental_scheduler_placement=SchedulerPlacement(
        region="us-east-1",
        zone="us-east-1a",
        spot=False,
    ),
)
def f():
    pass


def test_scheduler_placement(servicer, client):
    with stub.run(client=client):
        assert len(servicer.app_functions) == 1
        fn = servicer.app_functions["fu-1"]
        assert fn._experimental_scheduler
        assert fn._experimental_scheduler_placement == api_pb2.SchedulerPlacement(
            _region="us-east-1",
            _zone="us-east-1a",
            _lifecycle="on-demand",
        )
